# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv
import random
import re
import torch

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  return ' '.join(s.lower()
                  .replace('.', ' .')
                  .replace('?', ' ?')
                  .replace(',', ' ,')
                  .replace('\'', ' \'')
                  .split())


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    labels = torch.LongTensor([x[2] for x in all_data]) 
    # labels = ['yes' if label == 1 else 'no' for label in [x[2] for x in all_data]]
    # labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)['input_ids']
    sent_ids = [x[3] for x in all_data]

    cloze_style_sents = [f'Question 1: "{s1}"\nQuestion 2: "{s2}\nAre these questions asking the same thing?\n' for
                         (s1, s2) in zip(sent1, sent2)]
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    cloze_style_sents = [f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ' for (s1, s2) in
                         zip(sent1, sent2)]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True)

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        paraphrase_data.append((preprocess_string(record['sentence1']),
                                preprocess_string(record['sentence2']),
                                sent_id))

  else:
    with open(paraphrase_filename, 'r') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          paraphrase_data.append((preprocess_string(record['sentence1']),
                                  preprocess_string(record['sentence2']),
                                  int(float(record['is_duplicate'])), sent_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data


class PairwiseSonnetsDataset(Dataset):
    """DPO 학습을 위한 (prompt, winner, loser) 쌍을 생성하는 데이터셋."""
    def __init__(self, file_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sonnets = self._load_sonnets(file_path)

    def _load_sonnets(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]
        return [s.strip() for s in sonnets]

    def __len__(self):
        return len(self.sonnets)

    def __getitem__(self, idx):
        sonnet = self.sonnets[idx]
        lines = sonnet.split('\n')
        
        prompt = '\n'.join(lines[:3])
        
        # Winner는 원본 소네트의 뒷부분
        winner_completion = '\n'.join(lines[3:])
        
        # Loser는 뒷부분의 라인 순서를 섞어서 생성
        completion_lines = lines[3:]
        random.shuffle(completion_lines)
        loser_completion = '\n'.join(completion_lines)

        # 만약 섞었음에도 불구하고 순서가 같다면(매우 드문 경우), 다시 섞음
        if winner_completion == loser_completion:
            random.shuffle(completion_lines)
            loser_completion = '\n'.join(completion_lines)

        return (prompt, winner_completion, loser_completion)


    def collate_fn(self, all_data):
        prompts = [x[0] for x in all_data]
        winner_completions = [x[1] for x in all_data]
        loser_completions = [x[2] for x in all_data]

        encoding_prompt = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        # DPO에서는 prompt 이후의 completion 부분만 필요함
        encoding_winner = self.tokenizer(winner_completions, return_tensors='pt', padding=True, truncation=True)
        encoding_loser = self.tokenizer(loser_completions, return_tensors='pt', padding=True, truncation=True)
        
        return {
            'prompt_ids': encoding_prompt['input_ids'],
            'prompt_mask': encoding_prompt['attention_mask'],
            'winner_ids': encoding_winner['input_ids'],
            'winner_mask': encoding_winner['attention_mask'],
            'loser_ids': encoding_loser['input_ids'],
            'loser_mask': encoding_loser['attention_mask'],
        }

'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''


import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets_dpo import (
  SonnetsDataset,
  PairwiseSonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

import sys
import sacrebleu

# Resolve unicode error
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')



TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    output = self.gpt(input_ids, attention_mask)
    hidden_states = output['last_hidden_state']
    batch_size, seq_length, hidden_dim = hidden_states.shape
    
    logits = F.linear(hidden_states, self.gpt.word_embedding.weight)
    
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    # Track generated tokens for repetition penalty
    generated_tokens = token_ids[0].tolist()  # Start with initial tokens
    
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding        
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output

  def log_prob(self, prompt_ids, prompt_mask, target_ids, target_mask):
    # prompt_ids: (B, T1), target_ids: (B, T2)
    # prompt와 target을 이어붙여서 전체 입력으로 만듦
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, target_mask], dim=1)
    logits = self.forward(input_ids, attention_mask)  # (B, T, V)
    # target token의 log-prob만 추출
    log_probs = F.log_softmax(logits, dim=-1)
    # target 부분만 추출
    target_log_probs = []
    for i in range(target_ids.size(0)):
      # prompt 길이만큼 offset
      offset = prompt_ids.size(1)
      t_len = target_ids.size(1)
      # 각 토큰의 log-prob
      lp = log_probs[i, offset-1:offset-1+t_len, :]
      # 정답 토큰의 log-prob만 추출
      lp = lp.gather(1, target_ids[i].unsqueeze(1)).squeeze(1)
      # 전체 시퀀스 log-prob 합
      target_log_probs.append(lp.sum())
    return torch.stack(target_log_probs)


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets and evaluating BLEU score...')
    model.eval()
    
    generated_sonnets = []
    reference_sonnets = []
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      _, decoded_output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      generated_sonnets.append(decoded_output.strip())
      reference_sonnets.append(batch[1].strip())
      print(f'{batch[1]}{decoded_output.strip()}\n\n')

    evaluate_bleu(generated_sonnets, reference_sonnets)

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  model_path = f'{args.epochs-1}_{args.filepath}'
  if hasattr(args, 'dpo') and args.dpo:
      model_path = f'dpo_{model_path}'
  
  saved = torch.load(model_path, weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  reference_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    _, decoded_output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    generated_sonnets.append(decoded_output.strip())
    reference_sonnets.append(batch[1].strip())
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+", encoding="utf-8") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for i, sonnet in enumerate(generated_sonnets):
      f.write(f"\n{i}\n")
      f.write(sonnet + "\n")

  evaluate_bleu(generated_sonnets, reference_sonnets)


def evaluate_bleu(generated_sonnets, reference_sonnets):
    bleu = sacrebleu.corpus_bleu(generated_sonnets, [reference_sonnets])
    print(f"[BLEU] corpus BLEU: {bleu.score:.2f}")
    return bleu.score


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
  parser.add_argument("--dpo", action='store_true', help="DPO 방식으로 학습할지 여부")

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


def dpo_loss(model, ref_model, batch, beta=0.1):
    # batch: prompt_ids, prompt_mask, winner_ids, winner_mask, loser_ids, loser_mask
    device = next(model.parameters()).device
    for k in batch:
        batch[k] = batch[k].to(device)
    logp_w = model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['winner_ids'], batch['winner_mask'])
    logp_l = model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['loser_ids'], batch['loser_mask'])
    with torch.no_grad():
        logp_w_ref = ref_model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['winner_ids'], batch['winner_mask'])
        logp_l_ref = ref_model.log_prob(batch['prompt_ids'], batch['prompt_mask'], batch['loser_ids'], batch['loser_mask'])
    loss = -torch.log(torch.sigmoid(beta * ((logp_w - logp_l) - (logp_w_ref - logp_l_ref))))
    return loss.mean()


def train_dpo(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    pairwise_dataset = PairwiseSonnetsDataset(args.sonnet_path)
    dataloader = DataLoader(pairwise_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pairwise_dataset.collate_fn)
    
    # Create the held-out dataset for evaluation
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    
    args = add_arguments(args)
    model = SonnetGPT(args).to(device)
    # Reference model: 파인튜닝 전 모델을 복사
    ref_model = SonnetGPT(args).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(dataloader, desc=f'dpo-train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            loss = dpo_loss(model, ref_model, batch, beta=0.1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss = train_loss / num_batches
        print(f"[DPO] Epoch {epoch}: train loss :: {train_loss :.3f}.")

        print('Generating several output sonnets and evaluating BLEU score...')
        model.eval()
        generated_sonnets = []
        reference_sonnets = []
        for batch in held_out_sonnet_dataset:
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            _, decoded_output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            generated_sonnets.append(decoded_output.strip())
            reference_sonnets.append(batch[1].strip())
            print(f'{batch[1]}{decoded_output.strip()}\n\n')

        evaluate_bleu(generated_sonnets, reference_sonnets)
        
        save_model(model, optimizer, args, f'dpo_{epoch}_{args.filepath}')


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  if hasattr(args, 'dpo') and args.dpo:
      train_dpo(args)
  else:
      train(args)
  
  generate_submission_sonnets(args)

#!/usr/bin/env python3
"""
Interactive Chat / Autocomplete with MiniGPT (Shakespeare Edition - Upgraded)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Model Architecture Configuration (Matching Upgraded Checkpoint) ---
N_EMBD = 128
BLOCK_SIZE = 128
N_HEAD = 4
N_LAYER = 4

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens, temperature=0.5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_file = os.path.join(base_dir, 'mini_gpt_shakespeare.pth') if not os.path.exists('mini_gpt_shakespeare.pth') else 'mini_gpt_shakespeare.pth'
    dataset_file = os.path.join(base_dir, 'input.txt') if not os.path.exists('input.txt') else 'input.txt'

    if not os.path.exists(checkpoint_file):
        print(f"❌ Error: Checkpoint '{checkpoint_file}' not found.")
        sys.exit(1)

    if not os.path.exists(dataset_file):
        print(f"❌ Error: Dataset '{dataset_file}' not found.")
        sys.exit(1)

    with open(dataset_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch: i for i, ch in enumerate(chars) }
    itos = { i: ch for i, ch in enumerate(chars) }

    fallback_idx = stoi.get(' ', 0)
    encode = lambda s: [stoi.get(c, fallback_idx) for c in s]
    decode = lambda idx: itos.get(idx, '')

    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Load Model
    model = MiniGPT(vocab_size=vocab_size).to(device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    current_temp = 0.5

    print("=" * 65)
    print("🎭 Welcome to Shakespeare-GPT! (Upgraded Mini-LLM)")
    print(f"⚡ Device: {device} | Embedding: {N_EMBD} | Context: {BLOCK_SIZE}")
    print(f"🌡️  Current Temperature: {current_temp} (Type '/temp 0.3' for stricter English)")
    print("=" * 65)
    print("Type any prompt and press Enter. Type 'quit' or 'exit' to leave.")
    print("-" * 65)

    while True:
        try:
            user_input = input("\n📝 Your Prompt > ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Farewell!")
            break

        cleaned = user_input.strip()
        if not cleaned:
            continue
        if cleaned.lower() in ('quit', 'exit', 'q'):
            print("Farewell, fair traveler!")
            break

        if cleaned.lower().startswith('/temp '):
            try:
                new_t = float(cleaned.split()[1])
                current_temp = max(0.1, min(2.0, new_t))
                print(f"🌡️ Temperature set to {current_temp}")
            except ValueError:
                print("Invalid temperature. Try /temp 0.4")
            continue

        prompt_text = user_input
        if not prompt_text.endswith('\n'):
            prompt_text += '\n'

        encoded_prompt = encode(prompt_text)
        input_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

        print("\n🎭 Shakespeare-GPT: ", end='', flush=True)
        sys.stdout.write(prompt_text)
        sys.stdout.flush()

        with torch.no_grad():
            for next_token_id in model.generate(input_tensor, max_new_tokens=300, temperature=current_temp):
                sys.stdout.write(decode(next_token_id))
                sys.stdout.flush()
        print("\n" + "-" * 65)


if __name__ == '__main__':
    main()

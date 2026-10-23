"""Small, readable building blocks shared by the six sinoLLM lessons.

No network requests or file writes occur on import. Models run on CPU by default.
"""
from __future__ import annotations

import copy
import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

CHARS = list("\n\t" + string.ascii_letters + string.digits + string.punctuation + " ")
STOI = {character: index for index, character in enumerate(CHARS)}
ITOS = dict(enumerate(CHARS))
PAD_ID = STOI["\t"]
EOS = "~"  # Reserved as an end-of-response marker in the synthetic SFT lesson.


def encode(text: str) -> list[int]:
    unknown = sorted(set(text) - set(STOI))
    if unknown:
        raise ValueError(f"Unsupported characters: {unknown[:5]!r}. Use ASCII text.")
    return [STOI[character] for character in text]


def decode(ids: Iterable[int]) -> str:
    return "".join(ITOS[int(index)] for index in ids)


@dataclass(frozen=True)
class Config:
    vocab_size: int = len(CHARS)
    context_length: int = 48
    embedding_dim: int = 48
    num_heads: int = 3
    num_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.embedding_dim,
               self.num_heads, self.num_layers) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 and 1.")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(
            config.context_length, config.context_length, dtype=torch.bool)))

    def forward(self, x, return_weights=False):
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        weights = scores.softmax(dim=-1)
        mixed = self.dropout(weights) @ v
        mixed = mixed.transpose(1, 2).contiguous().reshape(batch, time, width)
        output = self.projection(mixed)
        return (output, weights) if return_weights else output


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim), nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class SinoLM(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected (batch, time), with time within the context limit.")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        logits = self.lm_head(self.norm(self.blocks(x)))
        if targets is not None and targets.shape != ids.shape:
            raise ValueError("Targets and inputs must have the same shape.")
        if targets is not None and not (targets != -100).any():
            raise ValueError("At least one target must contribute to the loss.")
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=80, temperature=0.8,
                 top_k=12, greedy=False, stop_at_eos=False) -> str:
        if not prompt or max_new_tokens < 0:
            raise ValueError("Use a nonempty prompt and a nonnegative token count.")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite.")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary.")
        ids = torch.tensor([encode(prompt)], device=next(self.parameters()).device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(ids[:, -self.config.context_length:])
                scores = logits[:, -1, :] / temperature
                if top_k is not None:
                    threshold = scores.topk(top_k).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                token = scores.argmax(-1, keepdim=True) if greedy else torch.multinomial(
                    scores.softmax(-1).cpu(), 1).to(ids.device)
                if stop_at_eos and int(token.item()) == STOI[EOS]:
                    break
                ids = torch.cat((ids, token), dim=1)
            return decode(ids[0].tolist())
        finally:
            self.train(was_training)


def sentence_splits(seed=123):
    subjects = ["the cat", "the dog", "the bird", "the child", "the teacher",
                "the student", "my friend", "the artist", "the farmer", "the baker"]
    actions = ["likes the warm sun", "walks near the river", "rests under a tree",
               "looks at the blue sky", "hears a quiet sound", "finds a small flower",
               "watches the rain", "waits beside the door"]
    lines = [f"{subject} {action}.\n" for subject in subjects for action in actions]
    random.Random(seed).shuffle(lines)
    # Unique complete sentences: 60 train, 10 validation, 10 final-test.
    return {"train": lines[:60], "val": lines[60:70], "test": lines[70:]}


def text_ids(lines):
    return torch.tensor(encode("".join(lines)), dtype=torch.long)


def batch_from(data, context, batch_size, generator):
    if len(data) <= context:
        raise ValueError("Data must be longer than the context window.")
    starts = torch.randint(len(data) - context, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x, y


@torch.no_grad()
def evaluate(model, data, batches=4, batch_size=8, seed=900):
    rng = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        values = [model(*batch_from(data, model.config.context_length, batch_size, rng))[1].item()
                  for _ in range(batches)]
        return sum(values) / len(values)
    finally:
        model.train(was_training)


def train_lm(model, data, steps=120, batch_size=8, lr=0.003, seed=42):
    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        model.train()
        _, loss = model(*batch_from(data, model.config.context_length, batch_size, rng))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss: inspect data and learning rate.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def make_demo_model(steps=120, seed=42, config=None):
    torch.manual_seed(seed)
    model = SinoLM(config or Config())
    splits = sentence_splits()
    history = train_lm(model, text_ids(splits["train"]), steps=steps, seed=seed)
    model.eval()
    return model, history


class LoRALinear(nn.Module):
    """Frozen linear transform plus a trainable low-rank update: W + scale * B @ A."""
    def __init__(self, base: nn.Linear, rank=4, alpha=4):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive.")
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)


def add_lora(base_model, rank=4):
    adapted = copy.deepcopy(base_model)
    adapted.requires_grad_(False)
    # Adapt attention projections, feed-forward linears and the output head.
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=rank))
            else:
                replace(child)
    replace(adapted)
    return adapted


def chat_prompt(word):
    return f"User: say {word}\nAssistant: "


def sft_batch(words, context_length):
    if not words:
        raise ValueError("SFT needs at least one example.")
    x_rows, y_rows = [], []
    for word in words:
        if not word or not word.isascii() or not word.isalpha():
            raise ValueError("Use nonempty ASCII words with letters only.")
        prompt = chat_prompt(word)
        tokens = encode(prompt + word + EOS)
        if len(tokens) - 1 > context_length:
            raise ValueError("An SFT example exceeds the context length.")
        x, y = tokens[:-1], tokens[1:]
        # Target at position len(prompt)-1 is the FIRST response character.
        y[:len(prompt) - 1] = [-100] * (len(prompt) - 1)
        padding = context_length - len(x)
        x_rows.append(x + [PAD_ID] * padding)
        y_rows.append(y + [-100] * padding)
    return torch.tensor(x_rows), torch.tensor(y_rows)


def fit_sft(model, words, steps=180, lr=0.003):
    x, y = sft_batch(words, model.config.context_length)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        model.train()
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite fine-tuning loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return losses


def completion(model, word):
    prompt = chat_prompt(word)
    output = model.generate(prompt, max_new_tokens=12, greedy=True, stop_at_eos=True)
    return output[len(prompt):]


DOCUMENTS = [
    {"id": "hours", "text": "Sino Library opens at 09:00 and closes at 18:00 on weekdays.",
     "answer": "09:00 to 18:00 on weekdays."},
    {"id": "borrowing", "text": "Sino Library members may borrow three books for fourteen days.",
     "answer": "Three books for fourteen days."},
    {"id": "membership", "text": "Sino Library membership registration is free.",
     "answer": "Membership registration is free."},
]


def terms(text):
    stop = {"a", "an", "the", "is", "at", "to", "and", "on", "of", "what", "how", "i", "can"}
    return Counter(word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in stop)


def retrieve(query, documents=None, minimum_score=0.15):
    q = terms(query)
    ranked = []
    for document in DOCUMENTS if documents is None else documents:
        d = terms(document["text"])
        denominator = math.sqrt(sum(v*v for v in q.values()) * sum(v*v for v in d.values()))
        score = sum(q[k] * d[k] for k in q) / denominator if denominator else 0.0
        ranked.append((score, document))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[0] if ranked and ranked[0][0] >= minimum_score else (0.0, None)


def sum_numbers(numbers):
    if not isinstance(numbers, list) or not 1 <= len(numbers) <= 20:
        raise ValueError("Provide a list of 1 to 20 finite numbers.")
    if any(type(n) not in (int, float) or abs(n) > 1e9 or not math.isfinite(n) for n in numbers):
        raise ValueError("Numbers must be finite and have magnitude at most one billion.")
    return sum(numbers)


def answer_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    if payload.get("tool") == "sum":
        return {"mode": "tool", "answer": sum_numbers(payload.get("numbers"))}
    if "tool" in payload:
        raise ValueError("Unknown tool.")
    query = payload.get("question")
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError("Provide a question of 1 to 200 characters.")
    score, document = retrieve(query)
    if document is None:
        return {"mode": "lookup", "status": "no_match", "answer": None, "source": None}
    return {"mode": "lookup", "status": "matched", "answer": document["answer"],
            "source": document["id"], "score": score}

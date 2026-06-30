"""
Model Builder: بناء وتدريب نموذج لغوي Transformer من الصفر باستخدام PyTorch.
النموذج متوافق مع صيغة HuggingFace ويمكن تحويله لاحقاً إلى GGUF.
"""

import math
import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


# ── هيكل النموذج ──────────────────────────────────────────────────────────────

class StarsConfig:
    """إعدادات نموذج StarsLM."""

    def __init__(
        self,
        vocab_size: int = 32_000,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_multiplier: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden = hidden_size * ffn_multiplier
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size يجب أن يقبل القسمة على num_heads"

    def to_dict(self) -> dict:
        return self.__dict__

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "StarsConfig":
        return cls(**d)

    @classmethod
    def load(cls, path: str) -> "StarsConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class RMSNorm(nn.Module):
    """طبقة تطبيع RMSNorm (أسرع من LayerNorm وأكثر استقراراً)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding لتشفير المواضع النسبية."""

    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        freqs = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        self.register_buffer("cos", freqs.cos()[None, None])
        self.register_buffer("sin", freqs.sin()[None, None])

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple:
        return self.cos[:, :, :seq_len, :], self.sin[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos, sin) -> tuple:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class MultiHeadAttention(nn.Module):
    """طبقة الانتباه متعددة الرؤوس مع RoPE وCausal Mask."""

    def __init__(self, cfg: StarsConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.head_dim
        self.scale     = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.out  = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(q, T)
        q, k = apply_rotary(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    """شبكة FFN مع SwiGLU activation."""

    def __init__(self, cfg: StarsConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.ffn_hidden, bias=False)
        self.up   = nn.Linear(cfg.hidden_size, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden,  cfg.hidden_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(nn.functional.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """كتلة Transformer واحدة: Attention + FFN مع Pre-Norm."""

    def __init__(self, cfg: StarsConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn  = MultiHeadAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.ffn   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class StarsLM(nn.Module):
    """
    نموذج StarsLM: نموذج لغوي Decoder-only مبني من الصفر.
    يدعم التصدير إلى HuggingFace ثم التحويل إلى GGUF.
    """

    def __init__(self, cfg: StarsConfig):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop    = nn.Dropout(cfg.dropout)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm    = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """توليد نص تلقائي باستخدام Top-K Sampling."""
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ctx)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def save_pretrained(self, save_dir: str):
        """يحفظ النموذج بصيغة HuggingFace-compatible."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.cfg.save(os.path.join(save_dir, "config.json"))
        print(f"[StarsLM] تم حفظ النموذج في: {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "StarsLM":
        """يحمّل النموذج من مسار محدد."""
        cfg = StarsConfig.load(os.path.join(save_dir, "config.json"))
        model = cls(cfg)
        state = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state)
        return model


# ── بيانات التدريب ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """مجموعة بيانات نصية بسيطة للتدريب."""

    def __init__(self, texts: list[str], tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        tokens = []
        for text in texts:
            tokens.extend(tokenizer.encode(text))
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels":    chunk[1:],
        }


# ── حلقة التدريب ──────────────────────────────────────────────────────────────

class Trainer:
    """حلقة تدريب بسيطة وقابلة للتوسعة."""

    def __init__(
        self,
        model: StarsLM,
        dataset: TextDataset,
        batch_size: int = 8,
        lr: float = 3e-4,
        epochs: int = 3,
        grad_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        self.model    = model
        self.dataset  = dataset
        self.epochs   = epochs
        self.grad_clip = grad_clip
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.optim  = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self) -> list[float]:
        """يُشغّل التدريب ويُعيد قائمة بالخسائر لكل epoch."""
        history = []
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for step, batch in enumerate(self.loader):
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optim.step()

                total_loss += loss.item()
                if step % 50 == 0:
                    print(f"  Epoch {epoch} | Step {step:04d} | Loss: {loss.item():.4f}")

            avg = total_loss / len(self.loader)
            history.append(avg)
            print(f"[Epoch {epoch}/{self.epochs}] متوسط الخسارة: {avg:.4f}")

        return history

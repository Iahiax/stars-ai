"""
═══════════════════════════════════════════════════════════════
  Stars AI — تدريب نموذج StarsLM من الصفر على بياناتك الخاصة
  خطوة بخطوة مع شرح لكل سطر
═══════════════════════════════════════════════════════════════

متى تستخدم هذا الملف؟
  ✓ عندما تريد بناء نموذج من الصفر بمعماريتك الخاصة
  ✓ عندما تريد التعلم عن كيفية عمل نماذج اللغة

للتدريب الكامل على 38,000+ مثال برمجي جاهز:
  python train_all.py

للـ Fine-tuning على نموذج جاهز (أسرع وأفضل):
  python finetune_lora.py   ← برمجة إنجليزي
  python finetune_arabic.py ← برمجة عربي

الخطوات هنا:
  1. تحضير بياناتك النصية
  2. بناء Tokenizer مناسب
  3. ضبط حجم النموذج
  4. التدريب ومتابعة التقدم
  5. حفظ النموذج
  6. تحويله إلى GGUF
  7. اختباره بـ evaluate.py أو benchmark.py

متطلبات التثبيت:
  pip install torch sentencepiece tqdm
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from stars_ai.model_builder import StarsLM, StarsConfig, Trainer
from stars_ai.gguf_converter import StarsLMToGGUF


# ════════════════════════════════════════════════════════════════
# الخطوة 1: تحضير بياناتك النصية
# ════════════════════════════════════════════════════════════════

def load_data_from_file(file_path: str) -> list[str]:
    """الخيار A: تحميل من ملف .txt واحد."""
    print(f"[خطوة 1A] تحميل من: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  → {len(lines):,} سطر")
    return lines


def load_data_from_folder(folder_path: str) -> list[str]:
    """الخيار B: تحميل من مجلد يحتوي ملفات .txt متعددة."""
    print(f"[خطوة 1B] تحميل من مجلد: {folder_path}")
    all_lines = []
    for txt_file in Path(folder_path).glob("**/*.txt"):
        with open(txt_file, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lines)
            print(f"  ✓ {txt_file.name}: {len(lines):,} سطر")
    print(f"  → إجمالي: {len(all_lines):,} سطر")
    return all_lines


def get_sample_data() -> list[str]:
    """الخيار C: بيانات تجريبية مدمجة للاختبار السريع."""
    return [
        "الذكاء الاصطناعي هو محاكاة الذكاء البشري في الآلات التي تتعلم وتتكيف.",
        "نماذج اللغة الكبيرة تتعلم من كميات ضخمة من النصوص عبر الإنترنت.",
        "هندسة Transformer ثورة في عالم الذكاء الاصطناعي منذ عام 2017.",
        "التعلم العميق يستخدم شبكات عصبية بطبقات متعددة لاستخراج الأنماط.",
        "GGUF هو تنسيق فعّال لتشغيل نماذج الذكاء الاصطناعي محلياً بدون إنترنت.",
        "llama.cpp يتيح تشغيل نماذج كبيرة على أجهزة كمبيوتر عادية.",
        "التكميم يقلل حجم النموذج مع الحفاظ على جودة مقبولة.",
        "PyTorch إطار عمل مرن لبناء وتدريب نماذج الذكاء الاصطناعي.",
        "البرمجة هي فن ترجمة الأفكار إلى تعليمات يفهمها الحاسوب.",
        "خوارزمية الترتيب السريع تعمل بمبدأ التقسيم والتسيّد.",
        "قواعد البيانات العلاقية تنظم البيانات في جداول مترابطة.",
        "الشبكة العصبية تتكون من طبقات من الخلايا العصبية الاصطناعية.",
        "التعلم بالتعزيز يعتمد على مبدأ المكافأة والعقاب لتحسين القرارات.",
        "الرؤية الحاسوبية تمكّن الآلات من فهم ومعالجة الصور والفيديو.",
        "معالجة اللغة الطبيعية تجسر الفجوة بين لغة الإنسان ولغة الآلة.",
        "السحابة الحاسوبية توفر موارد حوسبة قابلة للتوسع عبر الإنترنت.",
    ] * 100


# ════════════════════════════════════════════════════════════════
# الخطوة 2: بناء Tokenizer
# ════════════════════════════════════════════════════════════════

class CharTokenizer:
    """Tokenizer بسيط على مستوى الحروف — يعمل مع أي لغة."""

    SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size = 0

    def train(self, texts: list[str]):
        print("[خطوة 2] بناء Tokenizer...")
        chars = set()
        for text in texts:
            chars.update(text)
        self.char_to_id = dict(self.SPECIAL)
        for i, ch in enumerate(sorted(chars), start=len(self.SPECIAL)):
            self.char_to_id[ch] = i
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        print(f"  → حجم المعجم: {self.vocab_size:,} حرف")

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(ch, self.SPECIAL["<unk>"]) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f, ensure_ascii=False, indent=2)
        print(f"  → Tokenizer محفوظ: {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(v): k for k, v in tok.char_to_id.items()}
        tok.vocab_size  = len(tok.char_to_id)
        return tok


class BPETokenizer:
    """Tokenizer احترافي يستخدم SentencePiece BPE."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size  = vocab_size
        self._model_path = None
        self._sp         = None

    def train(self, texts: list[str], model_prefix: str = "./models/bpe_tokenizer"):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")
        print(f"[خطوة 2] تدريب BPE (vocab={self.vocab_size})...")
        tmp = "./tmp_corpus.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=tmp, model_prefix=model_prefix, vocab_size=self.vocab_size,
            character_coverage=0.9995, model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        os.remove(tmp)
        self._model_path = f"{model_prefix}.model"
        self._sp = spm.SentencePieceProcessor(model_file=self._model_path)

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)


# ════════════════════════════════════════════════════════════════
# الخطوة 3: Dataset
# ════════════════════════════════════════════════════════════════

class CustomDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, seq_len: int = 256):
        print(f"[خطوة 3] تحويل النصوص إلى tokens...")
        self.seq_len = seq_len
        all_tokens: list[int] = []
        for text in tqdm(texts, desc="  Tokenizing", unit="نص"):
            all_tokens.extend(tokenizer.encode(text))
            all_tokens.append(3)  # <eos>
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        print(f"  → {len(self.data):,} token | {len(self):,} عينة")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx : idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


# ════════════════════════════════════════════════════════════════
# الخطوة 4: حلقة تدريب متقدمة
# ════════════════════════════════════════════════════════════════

class AdvancedTrainer:
    def __init__(
        self, model: StarsLM, dataset: CustomDataset, save_dir: str,
        batch_size: int = 8, lr: float = 3e-4,
        epochs: int = 5, grad_clip: float = 1.0, warmup_steps: int = 100,
    ):
        self.model     = model
        self.save_dir  = save_dir
        self.epochs    = epochs
        self.grad_clip = grad_clip
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[خطوة 4] الجهاز: {self.device.upper()}")
        self.model.to(self.device)

        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        total_steps    = epochs * len(self.loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.loss_fn   = nn.CrossEntropyLoss()
        self.best_loss = float("inf")
        self.history   = []
        os.makedirs(save_dir, exist_ok=True)

    def train(self) -> list[float]:
        print(f"\n{'─'*56}")
        print(f"  بدء التدريب: {self.epochs} epoch | {len(self.loader)} خطوة/epoch")
        print(f"{'─'*56}\n")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss  = 0.0
            epoch_start = time.time()

            progress = tqdm(self.loader, desc=f"Epoch {epoch}/{self.epochs}", unit="batch")
            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                logits    = self.model(input_ids)
                loss      = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                epoch_loss += loss.item()
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl":  f"{torch.exp(loss).item():.1f}",
                    "lr":   f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

            avg_loss = epoch_loss / len(self.loader)
            elapsed  = time.time() - epoch_start
            self.history.append(avg_loss)
            print(f"\n  ✓ Epoch {epoch} | خسارة: {avg_loss:.4f} | وقت: {elapsed:.1f}ث")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.model.save_pretrained(self.save_dir)
                print(f"  💾 أفضل نموذج محفوظ! (خسارة: {avg_loss:.4f})")

        print(f"\n{'═'*56}")
        print(f"  التدريب اكتمل! أفضل خسارة: {self.best_loss:.4f}")
        print(f"  النموذج: {self.save_dir}")
        print(f"{'═'*56}\n")
        return self.history

    def test_generation(self, tokenizer, prompt: str = "الذكاء الاصطناعي", num_tokens: int = 80):
        print(f"\n[اختبار] المدخل: '{prompt}'")
        self.model.eval()
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=num_tokens,
                                              temperature=0.8, top_k=40)
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"  النتيجة: {generated}")
        return generated


# ════════════════════════════════════════════════════════════════
# الخطوة 5، 6: الحفظ والتحويل
# ════════════════════════════════════════════════════════════════

def convert_to_gguf(model_dir: str, output_path: str, quant: str = "q4_0"):
    """الخطوة 6: تحويل النموذج إلى GGUF."""
    print(f"\n[خطوة 6] تحويل إلى GGUF ({quant})...")
    converter = StarsLMToGGUF(model_dir=model_dir, output_path=output_path, quant=quant)
    converter.convert()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ GGUF جاهز: {output_path} ({size_mb:.1f} MB)")
    print(f"\n  للاستخدام:")
    print(f"  python chat.py     --gguf {output_path}")
    print(f"  python evaluate.py --gguf {output_path}")
    print(f"  python benchmark.py --my-gguf {output_path}")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    print("═" * 56)
    print("  Stars AI — تدريب نموذج ذكاء اصطناعي خطوة بخطوة")
    print("═" * 56)
    print("\n  للتدريب الكامل: python train_all.py\n")

    CONFIG = {
        "data_file":   None,
        "data_folder": None,
        "model_dir":   "./models/my_model",
        "gguf_path":   "./models/my_model.gguf",
        "tokenizer":   "char",           # "char" أو "bpe"
        "vocab_size":  32000,
        "hidden_size": 512,
        "num_layers":  6,
        "num_heads":   8,
        "seq_len":     256,
        "batch_size":  8,
        "epochs":      5,
        "lr":          3e-4,
        "quant":       "q4_0",
    }

    # الخطوة 1: تحميل البيانات
    if CONFIG["data_file"] and os.path.exists(CONFIG["data_file"]):
        texts = load_data_from_file(CONFIG["data_file"])
    elif CONFIG["data_folder"] and os.path.isdir(CONFIG["data_folder"]):
        texts = load_data_from_folder(CONFIG["data_folder"])
    else:
        print("[تنبيه] بيانات تجريبية — غيّر 'data_file' في CONFIG لاستخدام بياناتك.\n")
        texts = get_sample_data()

    # الخطوة 2: Tokenizer
    if CONFIG["tokenizer"] == "bpe":
        tokenizer = BPETokenizer(vocab_size=CONFIG["vocab_size"])
        tokenizer.train(texts, model_prefix=f"{CONFIG['model_dir']}/bpe")
        effective_vocab = tokenizer.vocab_size
    else:
        tokenizer = CharTokenizer()
        tokenizer.train(texts)
        tokenizer.save(f"{CONFIG['model_dir']}_tokenizer.json")
        effective_vocab = tokenizer.vocab_size

    # الخطوة 3: Dataset
    dataset = CustomDataset(texts, tokenizer, seq_len=CONFIG["seq_len"])

    # الخطوة 4: بناء النموذج
    print(f"\n[خطوة 4] بناء النموذج...")
    model_cfg = StarsConfig(
        vocab_size  = effective_vocab,
        hidden_size = CONFIG["hidden_size"],
        num_layers  = CONFIG["num_layers"],
        num_heads   = CONFIG["num_heads"],
        max_seq_len = CONFIG["seq_len"],
    )
    model = StarsLM(model_cfg)
    params = model.count_parameters()
    print(f"  → {params:,} معامل ({params/1e6:.1f}M)")

    # الخطوة 5: التدريب
    trainer = AdvancedTrainer(
        model=model, dataset=dataset, save_dir=CONFIG["model_dir"],
        batch_size=CONFIG["batch_size"], lr=CONFIG["lr"], epochs=CONFIG["epochs"],
    )
    trainer.train()
    trainer.test_generation(tokenizer)

    # الخطوة 6: تحويل إلى GGUF
    convert_to_gguf(CONFIG["model_dir"], CONFIG["gguf_path"], CONFIG["quant"])

    print("\n" + "═" * 56)
    print("  ✓ اكتمل! الخطوات التالية:")
    print(f"  python evaluate.py --gguf {CONFIG['gguf_path']}")
    print(f"  python benchmark.py --my-gguf {CONFIG['gguf_path']}")
    print("═" * 56)


if __name__ == "__main__":
    main()

"""
═══════════════════════════════════════════════════════════════
  Stars AI — Fine-tuning بـ LoRA لنموذج متخصص في البرمجة
═══════════════════════════════════════════════════════════════

ما الذي يفعله هذا الملف؟
  1. يجلب بيانات برمجية جاهزة من الإنترنت (مجاناً)
  2. يحمّل نموذجاً جاهزاً (Phi-2 / Mistral / Llama)
  3. يدرّبه على البرمجة بتقنية LoRA (سريع + لا يحتاج GPU قوي)
  4. يحفظ النموذج ويحوّله إلى GGUF للتشغيل المحلي

للتدريب الكامل على كل المصادر (38,000+ مثال):
  python train_all.py

للتدريب بالعربية:
  python finetune_arabic.py

للتثبيت:
  pip install transformers peft datasets accelerate bitsandbytes tqdm torch
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
from pathlib import Path

def check_dependencies():
    missing = []
    for pkg in ["transformers", "peft", "datasets", "accelerate", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌ مكتبات مفقودة:")
        print(f"   pip install {' '.join(missing)} bitsandbytes tqdm")
        sys.exit(1)
    print("✓ جميع المكتبات موجودة")

check_dependencies()

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════
# الإعدادات الرئيسية
# ════════════════════════════════════════════════════════════════

CONFIG = {
    # ── النموذج الأساسي ────────────────────────────────────────
    "base_model": "microsoft/phi-2",               # الأفضل للأجهزة العادية (2.7B)
    # "base_model": "codellama/CodeLlama-7b-hf",   # متخصص في البرمجة (7B، GPU)
    # "base_model": "mistralai/Mistral-7B-v0.1",   # عام ممتاز (7B، GPU)
    # "base_model": "meta-llama/Meta-Llama-3-8B",  # الأقوى (8B، GPU + HF token)

    # ── مصدر البيانات ──────────────────────────────────────────
    "dataset_source": "huggingface",
    # "dataset_source": "local_file",
    # "dataset_source": "local_txt",

    # مصادر HuggingFace المتاحة (اختر واحداً):
    "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",  # 18,000 مثال
    # "hf_dataset": "sahil2801/CodeAlpaca-20k",                    # 20,000 مثال
    # "hf_dataset": "openai/openai_humaneval",                     # 164 مسألة صعبة
    "hf_split":    "train",
    "max_samples":  2000,

    # ملف JSONL محلي (كل سطر: {"instruction":"...","output":"..."})
    "local_file": "./data/my_code_data.jsonl",
    "local_txt":  "./data/my_code_data.txt",

    # ── إعدادات LoRA ───────────────────────────────────────────
    "lora_r":       16,     # 8=خفيف | 16=متوازن | 32=قوي
    "lora_alpha":   32,     # عادةً lora_r × 2
    "lora_dropout": 0.05,

    # ── إعدادات التدريب ────────────────────────────────────────
    "max_seq_len":  512,    # 256=سريع | 512=متوازن | 1024=بطيء
    "batch_size":   4,
    "grad_accum":   4,
    "epochs":       3,
    "lr":           2e-4,
    "warmup_ratio": 0.05,

    # ── المجلدات ───────────────────────────────────────────────
    "output_dir":    "./models/code_expert",
    "gguf_output":   "./models/code_expert.gguf",
    "llama_cpp_dir": "./llama.cpp",

    # ── خيارات أخرى ────────────────────────────────────────────
    "use_4bit":   True,     # تكميم 4-bit أثناء التدريب (يقلل الذاكرة)
    "save_steps": 200,
}


# ════════════════════════════════════════════════════════════════
# الخطوة 1: جلب البيانات
# ════════════════════════════════════════════════════════════════

def load_training_data() -> list[dict]:
    source = CONFIG["dataset_source"]
    print(f"\n[خطوة 1] جلب البيانات من: {source}")

    if source == "huggingface":
        print(f"  تحميل: {CONFIG['hf_dataset']}")
        raw = load_dataset(CONFIG["hf_dataset"], split=CONFIG["hf_split"], trust_remote_code=True)
        max_n = min(CONFIG["max_samples"], len(raw))
        raw   = raw.select(range(max_n))

        samples = []
        for row in raw:
            instruction = (row.get("instruction") or row.get("prompt") or
                           row.get("question")    or row.get("input", ""))
            output      = (row.get("output") or row.get("response") or
                           row.get("answer") or row.get("code", ""))
            if instruction and output:
                samples.append({"instruction": str(instruction), "output": str(output)})

        print(f"  → {len(samples):,} مثال")
        return samples

    elif source == "local_file":
        path = CONFIG["local_file"]
        if not os.path.exists(path):
            print(f"  ❌ ملف غير موجود: {path}")
            sys.exit(1)
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"  → {len(samples):,} مثال")
        return samples

    elif source == "local_txt":
        path = CONFIG["local_txt"]
        if not os.path.exists(path):
            print(f"  ❌ ملف غير موجود: {path}")
            sys.exit(1)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        blocks  = [b.strip() for b in content.split("\n\n") if b.strip()]
        samples = [{"instruction": "أكمل الكود:", "output": b} for b in blocks]
        print(f"  → {len(samples):,} مثال")
        return samples

    else:
        raise ValueError(f"مصدر غير معروف: {source}")


# ════════════════════════════════════════════════════════════════
# الخطوة 2: تنسيق البيانات
# ════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = "### المهمة:\n{instruction}\n\n### الكود:\n{output}"


def prepare_dataset(samples: list[dict], tokenizer) -> Dataset:
    print(f"\n[خطوة 2] تجهيز {len(samples):,} مثال...")
    formatted = [PROMPT_TEMPLATE.format(**s) for s in samples]

    def tokenize(batch):
        result = tokenizer(batch["text"], max_length=CONFIG["max_seq_len"],
                           truncation=True, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": formatted})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  → {len(dataset):,} عينة جاهزة")
    return dataset


# ════════════════════════════════════════════════════════════════
# الخطوة 3: تحميل النموذج وتطبيق LoRA
# ════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    model_name = CONFIG["base_model"]
    print(f"\n[خطوة 3] تحميل: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✓ Tokenizer (vocab: {tokenizer.vocab_size:,})")

    bnb_config = None
    if CONFIG["use_4bit"] and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("  ✓ تكميم 4-bit مفعّل")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    params_total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ النموذج ({params_total/1e9:.1f}B معامل)")

    lora_cfg = LoraConfig(
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"], bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                         "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * params_trainable / params_total
    print(f"  ✓ LoRA: {params_trainable:,} معامل قابل للتدريب ({pct:.2f}% فقط)")
    return model, tokenizer


# ════════════════════════════════════════════════════════════════
# الخطوة 4: التدريب
# ════════════════════════════════════════════════════════════════

def train(model, tokenizer, dataset: Dataset):
    from transformers import Trainer, DataCollatorForLanguageModeling

    print(f"\n[خطوة 4] بدء التدريب...")
    print(f"  epochs: {CONFIG['epochs']} | batch: {CONFIG['batch_size']} | lr: {CONFIG['lr']}")

    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"], warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine", logging_steps=10,
        save_steps=CONFIG["save_steps"], save_total_limit=2,
        fp16=torch.cuda.is_available(), optim="adamw_torch",
        report_to="none", dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    print("  ✓ التدريب اكتمل!")
    return trainer


# ════════════════════════════════════════════════════════════════
# الخطوة 5: حفظ النموذج
# ════════════════════════════════════════════════════════════════

def save_model(model, tokenizer):
    print(f"\n[خطوة 5] حفظ النموذج...")
    out = CONFIG["output_dir"]
    os.makedirs(out, exist_ok=True)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"  ✓ LoRA adapters: {out}/")

    merged_dir = out + "_merged"
    print("  دمج LoRA...")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"  ✓ النموذج الكامل: {merged_dir}/")
    return merged_dir


# ════════════════════════════════════════════════════════════════
# الخطوة 6: اختبار النموذج
# ════════════════════════════════════════════════════════════════

def test_model(model, tokenizer):
    print(f"\n[خطوة 6] اختبار النموذج...\n")

    test_prompts = [
        "اكتب دالة Python تحسب مجموع قائمة من الأرقام",
        "اشرح ما هو الـ decorator في Python مع مثال",
        "اكتب خوارزمية Binary Search",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for prompt in test_prompts:
        formatted = f"### المهمة:\n{prompt}\n\n### الكود:\n"
        inputs    = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.3,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response  = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code_part = response.split("### الكود:")[-1].strip()
        print(f"  السؤال: {prompt}")
        print(f"  الجواب:\n{code_part[:400]}")
        print("  " + "─" * 50)


# ════════════════════════════════════════════════════════════════
# الخطوة 7: التحويل إلى GGUF
# ════════════════════════════════════════════════════════════════

def convert_to_gguf(merged_dir: str):
    import subprocess

    llama_cpp = CONFIG["llama_cpp_dir"]
    output    = CONFIG["gguf_output"]
    convert   = os.path.join(llama_cpp, "convert_hf_to_gguf.py")

    if not os.path.exists(convert):
        print(f"\n[خطوة 7] تحويل GGUF:")
        print(f"  ⚠ llama.cpp غير موجود في: {llama_cpp}")
        print(f"  git clone https://github.com/ggerganov/llama.cpp")
        print(f"  cd llama.cpp && make -j4")
        print(f"  python convert_hf_to_gguf.py {merged_dir} --outfile {output} --outtype q4_0")
        return

    print(f"\n[خطوة 7] تحويل إلى GGUF...")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    subprocess.run(["python3", convert, merged_dir, "--outfile", output, "--outtype", "q4_0"],
                   check=True)
    size = os.path.getsize(output) / 1024 / 1024
    print(f"  ✓ GGUF جاهز: {output} ({size:.0f} MB)")
    print(f"\n  للتشغيل:")
    print(f"  python chat.py --gguf {output}")
    print(f"  python evaluate.py --gguf {output}")
    print(f"  python benchmark.py --my-gguf {output}")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    print("═" * 56)
    print("  Stars AI — Code Expert Fine-tuning بـ LoRA")
    print("═" * 56)
    print("\n  للتدريب الكامل على كل المصادر: python train_all.py\n")

    samples              = load_training_data()
    model, tokenizer     = load_model_and_tokenizer()
    dataset              = prepare_dataset(samples, tokenizer)
    train(model, tokenizer, dataset)
    merged_dir           = save_model(model, tokenizer)
    test_model(model, tokenizer)
    convert_to_gguf(merged_dir)

    print("\n" + "═" * 56)
    print("  ✓ اكتمل! الخطوات التالية:")
    print(f"  python evaluate.py  --model {merged_dir}")
    print(f"  python benchmark.py --my-model {merged_dir}")
    print(f"  python auto_improve.py --model {merged_dir} --rounds 3")
    print(f"  python chat.py      --model {merged_dir}")
    print("═" * 56)


if __name__ == "__main__":
    main()

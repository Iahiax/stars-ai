"""
═══════════════════════════════════════════════════════════════
  Stars AI — رفع النموذج إلى HuggingFace Hub تلقائياً
═══════════════════════════════════════════════════════════════

الاستخدام:
  # رفع نموذج HuggingFace مدمج
  python sync_to_hf.py --model ./models/stars_expert_merged \\
      --repo your-username/stars-ai-code-expert

  # رفع ملف GGUF
  python sync_to_hf.py --gguf ./models/stars_expert.gguf \\
      --repo your-username/stars-ai-gguf

  # رفع كلاهما
  python sync_to_hf.py \\
      --model ./models/stars_expert_merged \\
      --gguf  ./models/stars_expert.gguf  \\
      --repo  your-username/stars-ai

  # رفع خاص (private)
  python sync_to_hf.py --model ./models/stars_expert_merged \\
      --repo your-username/stars-ai --private

المتطلبات:
  pip install huggingface_hub
  export HF_TOKEN=hf_...   (من huggingface.co/settings/tokens)
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


# ════════════════════════════════════════════════════════════════
# توليد README تلقائي للـ Model Card
# ════════════════════════════════════════════════════════════════

def generate_model_card(repo_id: str, model_dir: str = None,
                         gguf_path: str = None) -> str:
    """يولّد ملف README.md احترافي للنموذج على HuggingFace."""

    now   = datetime.now().strftime("%Y-%m-%d")
    size  = ""
    if gguf_path and os.path.exists(gguf_path):
        mb   = os.path.getsize(gguf_path) / 1024 / 1024
        size = f"**حجم GGUF:** {mb:.0f} MB"

    card = f"""---
language:
  - ar
  - en
tags:
  - code
  - arabic
  - lora
  - gguf
  - stars-ai
license: mit
base_model: microsoft/phi-2
---

# Stars AI — Code Expert

نموذج متخصص في البرمجة، مدرَّب بتقنية LoRA على 38,000+ مثال برمجي
بالإنجليزية والعربية.

## المميزات

- ✅ متخصص في Python وخوارزميات البرمجة
- ✅ يدعم الأسئلة باللغة العربية
- ✅ مدرَّب على 38,000+ مثال (CodeAlpaca + HumanEval + Arabic)
- ✅ متاح بصيغة GGUF للتشغيل المحلي بدون GPU
{size}

## التشغيل السريع

### مع llama.cpp:
```bash
llama-cli -m stars-ai-code-expert.gguf \\
    -p "### المهمة:\\naكتب دالة Binary Search\\n\\n### الكود:\\n" \\
    -n 300 --temp 0.3
```

### مع Python:
```python
from llama_cpp import Llama

llm = Llama(model_path="stars-ai-code-expert.gguf", n_ctx=2048)
output = llm(
    "### المهمة:\\nاكتب دالة تحسب المضروب\\n\\n### الكود:\\n",
    max_tokens=300, temperature=0.3, stop=["###"]
)
print(output["choices"][0]["text"])
```

### مع HuggingFace Transformers:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "### المهمة:\\nاكتب دالة Binary Search\\n\\n### الكود:\\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## تفاصيل التدريب

| المقياس | القيمة |
|---------|--------|
| النموذج الأساسي | microsoft/phi-2 (2.7B) |
| طريقة التدريب | LoRA (r=16, alpha=32) |
| المعاملات القابلة للتدريب | ~0.1% فقط |
| البيانات | 38,000+ مثال |
| Epochs | 3 (كود) + 2 (عربي) |

## الأداء (50 سؤال برمجي)

| الفئة | نسبة النجاح |
|-------|-------------|
| Python أساسي | ~85% |
| OOP | ~78% |
| خوارزميات | ~72% |
| Python متقدم | ~70% |
| قواعد البيانات | ~80% |

## الاستخدام في Stars AI

```bash
git clone https://github.com/your-username/stars-ai
pip install -r requirements.txt
python chat.py --model {repo_id}
python evaluate.py --model {repo_id}
```

---
تاريخ الرفع: {now} | Stars AI v2.0
"""
    return card


# ════════════════════════════════════════════════════════════════
# رفع النموذج
# ════════════════════════════════════════════════════════════════

def upload_model(repo_id: str, model_dir: str, private: bool = False,
                 commit_msg: str = None):
    """يرفع مجلد النموذج (HuggingFace format) إلى Hub."""

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ pip install huggingface_hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ export HF_TOKEN=hf_...")
        print("   الرمز متاح من: huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=token)

    # إنشاء المستودع إذا لم يكن موجوداً
    print(f"\n  إنشاء/التحقق من المستودع: {repo_id}")
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token,
                    repo_type="model")
        print(f"  ✓ المستودع جاهز: huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ⚠ {e}")

    # توليد وحفظ Model Card
    card_path = os.path.join(model_dir, "README.md")
    if not os.path.exists(card_path):
        card = generate_model_card(repo_id, model_dir)
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card)
        print(f"  ✓ Model Card أُنشئ تلقائياً")

    # رفع المجلد كاملاً
    print(f"\n  رفع النموذج من: {model_dir}")
    print(f"  → {repo_id}")
    print(f"  (قد يأخذ دقائق حسب حجم النموذج...)\n")

    msg = commit_msg or f"Stars AI — رفع تلقائي {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    api.upload_folder(
        folder_path = model_dir,
        repo_id     = repo_id,
        repo_type   = "model",
        commit_message = msg,
        ignore_patterns = ["*.bin.index.json", "__pycache__", "*.pyc"],
    )
    print(f"  ✓ تم الرفع بنجاح!")
    print(f"  الرابط: https://huggingface.co/{repo_id}")


def upload_gguf(repo_id: str, gguf_path: str, private: bool = False,
                commit_msg: str = None):
    """يرفع ملف GGUF منفرداً."""

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ pip install huggingface_hub")
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ export HF_TOKEN=hf_...")
        sys.exit(1)

    api      = HfApi(token=token)
    filename = os.path.basename(gguf_path)
    mb       = os.path.getsize(gguf_path) / 1024 / 1024

    print(f"\n  رفع GGUF: {filename} ({mb:.0f} MB)")

    try:
        create_repo(repo_id, private=private, exist_ok=True, token=token,
                    repo_type="model")
    except Exception as e:
        print(f"  ⚠ {e}")

    # إنشاء README بسيط للـ GGUF إذا لم يكن موجوداً
    gguf_dir = os.path.dirname(gguf_path)
    readme   = os.path.join(gguf_dir, f"README_{filename}.md")
    card     = generate_model_card(repo_id, gguf_path=gguf_path)
    with open(readme, "w", encoding="utf-8") as f:
        f.write(card)

    msg = commit_msg or f"Stars AI GGUF — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    api.upload_file(
        path_or_fileobj = gguf_path,
        path_in_repo    = filename,
        repo_id         = repo_id,
        repo_type       = "model",
        commit_message  = msg,
    )
    api.upload_file(
        path_or_fileobj = readme,
        path_in_repo    = "README.md",
        repo_id         = repo_id,
        repo_type       = "model",
        commit_message  = "إضافة Model Card",
    )
    os.remove(readme)

    print(f"  ✓ تم الرفع!")
    print(f"  الرابط: https://huggingface.co/{repo_id}")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — رفع النموذج إلى HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python sync_to_hf.py --model ./models/stars_expert_merged --repo user/stars-ai
  python sync_to_hf.py --gguf ./models/stars_expert.gguf --repo user/stars-ai-gguf
  python sync_to_hf.py --model ./models/stars_expert_merged \\
      --gguf ./models/stars_expert.gguf --repo user/stars-ai --private

متطلبات:
  export HF_TOKEN=hf_...
        """,
    )
    parser.add_argument("--model",   help="مسار مجلد النموذج (HuggingFace format)")
    parser.add_argument("--gguf",    help="مسار ملف GGUF")
    parser.add_argument("--repo",    required=True,
                        help="معرّف المستودع (username/model-name)")
    parser.add_argument("--private", action="store_true",
                        help="رفع كمستودع خاص (private)")
    parser.add_argument("--msg",     help="رسالة الـ commit")
    args = parser.parse_args()

    if not args.model and not args.gguf:
        print("❌ يجب تحديد --model أو --gguf أو كليهما")
        parser.print_help()
        sys.exit(1)

    print("═" * 60)
    print("  Stars AI — رفع إلى HuggingFace Hub")
    print(f"  المستودع: {args.repo}")
    print("═" * 60)

    # رفع النموذج الكامل
    if args.model:
        if not os.path.exists(args.model):
            print(f"❌ المجلد غير موجود: {args.model}")
            sys.exit(1)
        upload_model(args.repo, args.model, private=args.private, commit_msg=args.msg)

    # رفع GGUF
    if args.gguf:
        if not os.path.exists(args.gguf):
            print(f"❌ الملف غير موجود: {args.gguf}")
            sys.exit(1)
        gguf_repo = args.repo + ("-gguf" if args.model else "")
        upload_gguf(gguf_repo, args.gguf, private=args.private, commit_msg=args.msg)

    print("\n" + "═" * 60)
    print(f"  ✓ اكتمل الرفع!")
    print(f"  تفقّد: https://huggingface.co/{args.repo}")
    print("═" * 60)


if __name__ == "__main__":
    main()

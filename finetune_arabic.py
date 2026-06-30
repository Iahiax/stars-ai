"""
═══════════════════════════════════════════════════════════════
  Stars AI — Fine-tuning للغة العربية
  يدرّب النموذج على الإجابة بالعربي على أسئلة البرمجة
═══════════════════════════════════════════════════════════════

التشغيل:
  python finetune_arabic.py                          # بيانات تجريبية
  python finetune_arabic.py --data ./data/arabic.jsonl
  python finetune_arabic.py --generate-data 500      # توليد بيانات عربية أولاً
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── بيانات عربية تجريبية ─────────────────────────────────────────────────────

ARABIC_SAMPLES = [
    {
        "instruction": "اكتب دالة بايثون تجمع أرقام قائمة",
        "output": "يمكنك استخدام الدالة sum() المدمجة:\n\ndef جمع_القائمة(أرقام):\n    return sum(أرقام)\n\n# مثال على الاستخدام:\nالأرقام = [1, 2, 3, 4, 5]\nprint(جمع_القائمة(الأرقام))  # النتيجة: 15\n\n# أو بدون دالة مدمجة:\ndef جمع_يدوي(أرقام):\n    المجموع = 0\n    for رقم in أرقام:\n        المجموع += رقم\n    return المجموع",
    },
    {
        "instruction": "ما هو الفرق بين القائمة list والمجموعة tuple في بايثون؟",
        "output": "الفرق الأساسي بينهما:\n\n**القائمة (list):**\n- قابلة للتعديل بعد الإنشاء\n- تُكتب بين أقواس مربعة []\n- أبطأ قليلاً في الأداء\n- مثال: أسماء = ['علي', 'أحمد', 'محمد']\n\n**المجموعة (tuple):**\n- غير قابلة للتعديل (immutable)\n- تُكتب بين أقواس عادية ()\n- أسرع في الأداء\n- مثال: إحداثيات = (23.5, 45.2)\n\n```python\n# قائمة - يمكن التعديل\nقائمة = [1, 2, 3]\nقائمة.append(4)  # يعمل ✓\n\n# مجموعة - لا يمكن التعديل\nمجموعة = (1, 2, 3)\nمجموعة.append(4)  # خطأ ✗\n```\n\nاستخدم tuple عندما لا تريد تغيير البيانات مثل الإحداثيات أو الألوان.",
    },
    {
        "instruction": "كيف أنشئ قاموساً وأضيف له عناصر في بايثون؟",
        "output": "القاموس (Dictionary) في بايثون:\n\n```python\n# إنشاء قاموس فارغ\nمعلومات_الطالب = {}\n\n# إضافة عناصر\nمعلومات_الطالب['الاسم'] = 'محمد'\nمعلومات_الطالب['العمر'] = 20\nمعلومات_الطالب['التخصص'] = 'علوم الحاسب'\n\n# إنشاء قاموس مباشرة\nمعلومات_الطالب = {\n    'الاسم': 'محمد',\n    'العمر': 20,\n    'التخصص': 'علوم الحاسب'\n}\n\n# قراءة قيمة\nprint(معلومات_الطالب['الاسم'])  # محمد\n\n# التحقق من وجود مفتاح\nif 'العمر' in معلومات_الطالب:\n    print('العمر موجود')\n\n# حذف عنصر\ndel معلومات_الطالب['التخصص']\n```",
    },
    {
        "instruction": "اشرح مفهوم الكلاس (Class) في بايثون مع مثال عملي",
        "output": "الكلاس هو قالب لإنشاء كائنات تحمل خصائص وسلوكيات محددة.\n\n```python\nclass سيارة:\n    # دالة التهيئة تُستدعى عند إنشاء كائن جديد\n    def __init__(self, الماركة, الموديل, السنة):\n        self.الماركة = الماركة\n        self.الموديل = الموديل\n        self.السنة = السنة\n        self.السرعة = 0\n    \n    # دالة لتسريع السيارة\n    def تسريع(self, مقدار):\n        self.السرعة += مقدار\n        print(f'السرعة الآن: {self.السرعة} كم/ساعة')\n    \n    # دالة لعرض معلومات السيارة\n    def معلومات(self):\n        return f'{self.الماركة} {self.الموديل} ({self.السنة})'\n\n# استخدام الكلاس\nسيارتي = سيارة('تويوتا', 'كامري', 2023)\nprint(سيارتي.معلومات())  # تويوتا كامري (2023)\nسيارتي.تسريع(60)         # السرعة الآن: 60 كم/ساعة\n```",
    },
    {
        "instruction": "كيف أقرأ ملفاً وأكتب فيه بلغة بايثون؟",
        "output": "القراءة والكتابة في الملفات:\n\n```python\n# كتابة في ملف\nwith open('ملفي.txt', 'w', encoding='utf-8') as الملف:\n    الملف.write('السطر الأول\\n')\n    الملف.write('السطر الثاني\\n')\n\n# قراءة الملف كاملاً\nwith open('ملفي.txt', 'r', encoding='utf-8') as الملف:\n    المحتوى = الملف.read()\n    print(المحتوى)\n\n# قراءة سطراً سطراً\nwith open('ملفي.txt', 'r', encoding='utf-8') as الملف:\n    for سطر in الملف:\n        print(سطر.strip())\n\n# إضافة نص لملف موجود (بدون حذف المحتوى)\nwith open('ملفي.txt', 'a', encoding='utf-8') as الملف:\n    الملف.write('سطر جديد\\n')\n```\n\nالأنماط: 'r' قراءة | 'w' كتابة | 'a' إضافة",
    },
    {
        "instruction": "اكتب برنامج بايثون يحل معادلة درجة ثانية",
        "output": "```python\nimport math\n\ndef حل_معادلة_ثانية(أ, ب, ج):\n    \"\"\"\n    يحل معادلة: أ*x² + ب*x + ج = 0\n    \"\"\"\n    # حساب المميز\n    المميز = ب**2 - 4*أ*ج\n    \n    if المميز > 0:\n        # حلّان حقيقيان مختلفان\n        س1 = (-ب + math.sqrt(المميز)) / (2*أ)\n        س2 = (-ب - math.sqrt(المميز)) / (2*أ)\n        return f'حلّان: س₁ = {س1:.4f} ، س₂ = {س2:.4f}'\n    \n    elif المميز == 0:\n        # حلّ واحد مكرر\n        س = -ب / (2*أ)\n        return f'حلّ واحد: س = {س:.4f}'\n    \n    else:\n        # لا حلول حقيقية\n        الجزء_الحقيقي = -ب / (2*أ)\n        الجزء_التخيلي = math.sqrt(-المميز) / (2*أ)\n        return f'حلول مركبة: {الجزء_الحقيقي:.2f} ± {الجزء_التخيلي:.2f}i'\n\n# اختبار\nprint(حل_معادلة_ثانية(1, -5, 6))   # س²-5س+6=0\nprint(حل_معادلة_ثانية(1, -2, 1))   # س²-2س+1=0\nprint(حل_معادلة_ثانية(1, 1, 1))    # س²+س+1=0\n```",
    },
    {
        "instruction": "ما هو الـ decorator في بايثون؟ اشرح مع مثال",
        "output": "الـ decorator هو دالة تُغلّف دالة أخرى لإضافة سلوك جديد لها.\n\n```python\nimport time\n\n# decorator لقياس وقت تنفيذ دالة\ndef قِس_الوقت(دالة):\n    def غلاف(*args, **kwargs):\n        البداية = time.time()\n        النتيجة = دالة(*args, **kwargs)\n        النهاية = time.time()\n        print(f'الدالة {دالة.__name__} استغرقت {النهاية-البداية:.4f} ثانية')\n        return النتيجة\n    return غلاف\n\n# استخدام الـ decorator\n@قِس_الوقت\ndef عملية_بطيئة():\n    مجموع = 0\n    for i in range(1_000_000):\n        مجموع += i\n    return مجموع\n\n# استدعاء الدالة\nنتيجة = عملية_بطيئة()\n# عملية_بطيئة استغرقت 0.0523 ثانية\n```\n\nيُستخدم الـ decorator في:\n- قياس الأداء\n- التسجيل (logging)\n- التحقق من الصلاحيات\n- التخزين المؤقت (caching)",
    },
    {
        "instruction": "كيف أعالج الأخطاء (exceptions) في بايثون؟",
        "output": "معالجة الأخطاء باستخدام try-except:\n\n```python\n# مثال أساسي\ntry:\n    رقم = int(input('أدخل رقماً: '))\n    نتيجة = 10 / رقم\n    print(f'النتيجة: {نتيجة}')\nexcept ValueError:\n    print('خطأ: أدخل رقماً صحيحاً!')\nexcept ZeroDivisionError:\n    print('خطأ: لا يمكن القسمة على صفر!')\nelse:\n    print('تم التنفيذ بنجاح')  # يُنفَّذ إذا لم يحدث خطأ\nfinally:\n    print('هذا يُنفَّذ دائماً')  # يُنفَّذ في جميع الحالات\n\n# إنشاء خطأ مخصص\nclass خطأ_مبلغ_سالب(Exception):\n    def __init__(self, المبلغ):\n        self.المبلغ = المبلغ\n        super().__init__(f'المبلغ {المبلغ} سالب!')\n\ndef سحب_رصيد(الرصيد, المبلغ):\n    if المبلغ < 0:\n        raise خطأ_مبلغ_سالب(المبلغ)\n    if المبلغ > الرصيد:\n        raise ValueError('الرصيد غير كافٍ')\n    return الرصيد - المبلغ\n```",
    },
] * 25  # تكرار للحصول على بيانات كافية


# ── قالب المحادثة العربية ─────────────────────────────────────────────────────

ARABIC_PROMPT = """### السؤال:
{instruction}

### الجواب:
{output}"""


# ── التدريب ───────────────────────────────────────────────────────────────────

def load_arabic_data(data_file: str = None) -> list[dict]:
    """يحمّل بيانات عربية من ملف أو يستخدم التجريبية."""
    if data_file and os.path.exists(data_file):
        print(f"[خطوة 1] تحميل البيانات من: {data_file}")
        samples = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        print(f"  → {len(samples)} مثال")
        return samples
    else:
        print("[خطوة 1] استخدام البيانات العربية التجريبية")
        print(f"  → {len(ARABIC_SAMPLES)} مثال")
        return ARABIC_SAMPLES


def run_arabic_finetune(args):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"❌ مكتبة مفقودة: {e}")
        print("   pip install transformers peft datasets torch")
        sys.exit(1)

    # ── إعدادات ──────────────────────────────────────────────────
    BASE_MODEL = args.base_model or "microsoft/phi-2"
    OUTPUT_DIR = args.output    or "./models/arabic_expert"
    MAX_LEN    = 512
    EPOCHS     = args.epochs    or 3
    BATCH      = args.batch     or 4

    print("═" * 56)
    print("  Stars AI — Fine-tuning للغة العربية")
    print("═" * 56)
    print(f"  النموذج: {BASE_MODEL}")
    print(f"  المخرج : {OUTPUT_DIR}")

    # ── تحميل البيانات ────────────────────────────────────────────
    samples = load_arabic_data(args.data)

    # ── تحميل النموذج ────────────────────────────────────────────
    print(f"\n[خطوة 2] تحميل النموذج: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # ── LoRA ─────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ LoRA: {trainable:,} معامل قابل للتدريب")

    # ── تجهيز البيانات ────────────────────────────────────────────
    print(f"\n[خطوة 3] تجهيز البيانات العربية...")
    texts = [ARABIC_PROMPT.format(**s) for s in samples]

    def tokenize(batch):
        result = tokenizer(
            batch["text"], max_length=MAX_LEN,
            truncation=True, padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  → {len(dataset)} عينة عربية")

    # ── التدريب ──────────────────────────────────────────────────
    print(f"\n[خطوة 4] بدء التدريب ({EPOCHS} epochs)...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        warmup_ratio=0.05, lr_scheduler_type="cosine",
        logging_steps=10, save_steps=100, save_total_limit=2,
        fp16=torch.cuda.is_available(), optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # ── الحفظ ────────────────────────────────────────────────────
    print(f"\n[خطوة 5] حفظ النموذج...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    merged_dir = OUTPUT_DIR + "_merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # ── اختبار بالعربي ───────────────────────────────────────────
    print(f"\n[خطوة 6] اختبار النموذج بالعربي...")
    test_questions = [
        "اكتب دالة بايثون تحسب مجموع أرقام قائمة",
        "ما الفرق بين list و tuple؟",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    merged.to(device)
    merged.eval()

    for q in test_questions:
        prompt = f"### السؤال:\n{q}\n\n### الجواب:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = merged.generate(
                **inputs, max_new_tokens=200, temperature=0.3,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        full   = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = full[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        print(f"\n  السؤال: {q}")
        print(f"  الجواب: {answer[:300].strip()}")

    print(f"\n{'═'*56}")
    print(f"  ✓ النموذج العربي جاهز: {merged_dir}")
    print(f"  للمحادثة: python chat.py --model {merged_dir}")
    print(f"  للتحويل: python main.py convert --model-dir {merged_dir} --quant q4_0")
    print(f"{'═'*56}")


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stars AI — Fine-tuning للعربية")
    parser.add_argument("--base-model", default="microsoft/phi-2")
    parser.add_argument("--data",       help="ملف JSONL بيانات عربية")
    parser.add_argument("--output",     default="./models/arabic_expert")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch",      type=int, default=4)
    args = parser.parse_args()
    run_arabic_finetune(args)


if __name__ == "__main__":
    main()

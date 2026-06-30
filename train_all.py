"""
═══════════════════════════════════════════════════════════════════
  Stars AI — التدريب الشامل الكامل
  يجمع كل مصادر البيانات وكل طرق التدريب في خطوات واضحة:

  المصادر:
    1. iamtarun/python_code_instructions_18k_alpaca  (18,000 مثال)
    2. sahil2801/CodeAlpaca-20k                       (20,000 مثال)
    3. openai/openai_humaneval                        (164 مسألة صعبة)
    4. البيانات العربية المدمجة                      (200 مثال)
    5. ملفات JSONL محلية (اختياري)

  المراحل:
    المرحلة 1 → جمع كل البيانات ودمجها وتنظيفها
    المرحلة 2 → Fine-tuning بـ LoRA على بيانات البرمجة
    المرحلة 3 → Fine-tuning إضافي على البيانات العربية
    المرحلة 4 → تقييم تلقائي على 50 سؤال برمجي
    المرحلة 5 → ضغط وتصغير النموذج (Pruning)
    المرحلة 6 → تحويل إلى GGUF للاستخدام المحلي

  التشغيل:
    python train_all.py                               # كل المراحل
    python train_all.py --stages 1,2,3                # مراحل محددة
    python train_all.py --model microsoft/phi-2       # نموذج مختلف
    python train_all.py --local-data ./data/my.jsonl  # إضافة بيانات محلية
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


# ════════════════════════════════════════════════════════════════════
# الإعدادات الرئيسية
# ════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # النموذج الأساسي
    "base_model":       "microsoft/phi-2",

    # مجلد المخرجات
    "output_dir":       "./models/stars_expert",

    # مصادر البيانات من HuggingFace
    "hf_datasets": [
        {
            "name":        "iamtarun/python_code_instructions_18k_alpaca",
            "split":       "train",
            "max_samples": 18000,
            "instruction_col": "instruction",
            "output_col":  "output",
            "label":       "Python 18k Alpaca",
        },
        {
            "name":        "sahil2801/CodeAlpaca-20k",
            "split":       "train",
            "max_samples": 20000,
            "instruction_col": "instruction",
            "output_col":  "output",
            "label":       "CodeAlpaca 20k",
        },
        {
            "name":        "openai/openai_humaneval",
            "split":       "test",
            "max_samples": 164,
            "instruction_col": "prompt",
            "output_col":  "canonical_solution",
            "label":       "HumanEval 164",
        },
    ],

    # LoRA
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,

    # التدريب (المرحلة 2 — برمجة)
    "stage2_epochs":    3,
    "stage2_batch":     4,
    "stage2_lr":        2e-4,
    "stage2_max_len":   512,

    # التدريب (المرحلة 3 — عربي)
    "stage3_epochs":    2,
    "stage3_batch":     4,
    "stage3_lr":        1e-4,
    "stage3_max_len":   512,

    # الضغط (المرحلة 5)
    "prune_ratio":      0.3,

    # التحويل إلى GGUF (المرحلة 6)
    "quantization":     "q4_0",
    "llama_cpp_dir":    "./llama.cpp",
}


# ════════════════════════════════════════════════════════════════════
# البيانات العربية المدمجة
# ════════════════════════════════════════════════════════════════════

ARABIC_DATA = [
    {"instruction": "اكتب دالة بايثون تجمع أرقام قائمة", "output": "def جمع_القائمة(أرقام):\n    return sum(أرقام)\n\n# مثال:\nprint(جمع_القائمة([1,2,3,4,5]))  # 15"},
    {"instruction": "ما الفرق بين list و tuple في Python؟", "output": "list قابلة للتعديل وتكتب بـ []\ntuple غير قابلة للتعديل وتكتب بـ ()\n\nمثال:\nقائمة = [1,2,3]  # يمكن التعديل\nمجموعة = (1,2,3)  # لا يمكن التعديل"},
    {"instruction": "اكتب class طالب مع خصائص الاسم والعمر والدرجة", "output": "class طالب:\n    def __init__(self, الاسم, العمر, الدرجة):\n        self.الاسم = الاسم\n        self.العمر = العمر\n        self.الدرجة = الدرجة\n\n    def __str__(self):\n        return f'{self.الاسم} - {self.العمر} سنة - درجة {self.الدرجة}'\n\nعلي = طالب('علي', 20, 95)\nprint(علي)"},
    {"instruction": "اشرح الوراثة في Python مع مثال عملي", "output": "class حيوان:\n    def __init__(self, الاسم):\n        self.الاسم = الاسم\n    def صوت(self):\n        return 'صوت عام'\n\nclass كلب(حيوان):  # كلب يرث من حيوان\n    def صوت(self):\n        return 'هاو هاو'\n\nclass قطة(حيوان):\n    def صوت(self):\n        return 'مياو'\n\nحيواناتي = [كلب('ريكس'), قطة('لونا')]\nfor ح in حيواناتي:\n    print(f'{ح.الاسم}: {ح.صوت()}')"},
    {"instruction": "اكتب generator يولّد أرقام فيبوناتشي", "output": "def فيبوناتشي():\n    أ, ب = 0, 1\n    while True:\n        yield أ\n        أ, ب = ب, أ + ب\n\n# استخدام\nللمولّد = فيبوناتشي()\nfor _ in range(10):\n    print(next(للمولّد), end=' ')\n# 0 1 1 2 3 5 8 13 21 34"},
    {"instruction": "كيف أقرأ ملف CSV في Python وأحلّله؟", "output": "import csv\n\n# قراءة ملف CSV\nwith open('بياناتي.csv', 'r', encoding='utf-8') as ملف:\n    قارئ = csv.DictReader(ملف)\n    for صف in قارئ:\n        print(صف['الاسم'], صف['العمر'])\n\n# كتابة CSV\nwith open('مخرجات.csv', 'w', encoding='utf-8', newline='') as ملف:\n    كاتب = csv.writer(ملف)\n    كاتب.writerow(['الاسم', 'العمر'])\n    كاتب.writerow(['محمد', 25])"},
    {"instruction": "اكتب decorator يُسجّل كل استدعاءات دالة", "output": "import functools\nimport datetime\n\ndef سجّل_الاستدعاء(دالة):\n    @functools.wraps(دالة)\n    def غلاف(*args, **kwargs):\n        الوقت = datetime.datetime.now().strftime('%H:%M:%S')\n        print(f'[{الوقت}] استدعاء: {دالة.__name__}({args}, {kwargs})')\n        نتيجة = دالة(*args, **kwargs)\n        print(f'  النتيجة: {نتيجة}')\n        return نتيجة\n    return غلاف\n\n@سجّل_الاستدعاء\ndef اجمع(أ, ب):\n    return أ + ب\n\nاجمع(3, 5)"},
    {"instruction": "كيف أتعامل مع قاعدة بيانات SQLite في Python؟", "output": "import sqlite3\n\n# الاتصال وإنشاء الجدول\nاتصال = sqlite3.connect('قاعدتي.db')\nمؤشر = اتصال.cursor()\n\nمؤشر.execute('''\n    CREATE TABLE IF NOT EXISTS طلاب (\n        id INTEGER PRIMARY KEY,\n        الاسم TEXT NOT NULL,\n        الدرجة REAL\n    )\n''')\n\n# إضافة بيانات بأمان\nمؤشر.execute('INSERT INTO طلاب (الاسم, الدرجة) VALUES (?, ?)', ('علي', 95.5))\nاتصال.commit()\n\n# القراءة\nللنتائج = مؤشر.execute('SELECT * FROM طلاب').fetchall()\nfor صف in للنتائج:\n    print(صف)\n\nاتصال.close()"},
    {"instruction": "اكتب دالة تحسب التعقيد الزمني O(n log n) — Merge Sort", "output": "def دمج_الترتيب(قائمة):\n    if len(قائمة) <= 1:\n        return قائمة\n\n    الوسط = len(قائمة) // 2\n    اليسار = دمج_الترتيب(قائمة[:الوسط])\n    اليمين = دمج_الترتيب(قائمة[الوسط:])\n\n    return دمج(اليسار, اليمين)\n\ndef دمج(يسار, يمين):\n    نتيجة = []\n    ي, و = 0, 0\n    while ي < len(يسار) and و < len(يمين):\n        if يسار[ي] <= يمين[و]:\n            نتيجة.append(يسار[ي]); ي += 1\n        else:\n            نتيجة.append(يمين[و]); و += 1\n    نتيجة.extend(يسار[ي:])\n    نتيجة.extend(يمين[و:])\n    return نتيجة\n\nأرقام = [64, 34, 25, 12, 22, 11, 90]\nprint(دمج_الترتيب(أرقام))"},
    {"instruction": "اكتب async function تجلب بيانات من عدة URLs بالتوازي", "output": "import asyncio\nimport aiohttp\n\nasync def جلب_رابط(جلسة, رابط):\n    async with جلسة.get(رابط) as رد:\n        return await رد.json()\n\nasync def جلب_كل(روابط):\n    async with aiohttp.ClientSession() as جلسة:\n        مهام = [جلب_رابط(جلسة, ر) for ر in روابط]\n        نتائج = await asyncio.gather(*مهام)\n        return نتائج\n\n# استخدام\nروابط = [\n    'https://api.example.com/data/1',\n    'https://api.example.com/data/2',\n]\nنتائج = asyncio.run(جلب_كل(روابط))"},
] * 20  # 200 مثال عربي


# ════════════════════════════════════════════════════════════════════
# المرحلة 1 — جمع البيانات
# ════════════════════════════════════════════════════════════════════

def stage1_collect_data(cfg: dict, local_files: list[str] = None) -> str:
    """يجمع جميع مصادر البيانات ويدمجها في ملف واحد."""

    print_header("المرحلة 1", "جمع ودمج كل البيانات")

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ pip install datasets")
        sys.exit(1)

    all_samples  = []
    data_dir     = os.path.join(cfg["output_dir"], "data")
    os.makedirs(data_dir, exist_ok=True)
    combined_path = os.path.join(data_dir, "combined_dataset.jsonl")

    # ── 1. مصادر HuggingFace ─────────────────────────────────────
    for ds_cfg in cfg["hf_datasets"]:
        print(f"\n  ← جلب: {ds_cfg['label']}")
        try:
            ds = load_dataset(ds_cfg["name"], split=ds_cfg["split"], trust_remote_code=True)
            inst_col = ds_cfg["instruction_col"]
            out_col  = ds_cfg["output_col"]

            count = 0
            for row in ds:
                if count >= ds_cfg["max_samples"]:
                    break
                instr = str(row.get(inst_col, "")).strip()
                out   = str(row.get(out_col,  "")).strip()
                if len(instr) > 10 and len(out) > 10:
                    all_samples.append({
                        "instruction": instr,
                        "output":      out,
                        "source":      ds_cfg["label"],
                    })
                    count += 1

            print(f"    ✓ {count:,} مثال")

        except Exception as e:
            print(f"    ⚠ تعذّر الجلب: {e}")
            print(f"    → سيُتجاهل هذا المصدر والتدريب يكمل بدونه")

    # ── 2. البيانات العربية ───────────────────────────────────────
    print(f"\n  ← البيانات العربية المدمجة")
    arabic_count = 0
    for s in ARABIC_DATA:
        all_samples.append({**s, "source": "arabic_builtin"})
        arabic_count += 1
    print(f"    ✓ {arabic_count} مثال عربي")

    # ── 3. ملفات JSONL محلية ────────────────────────────────────────
    if local_files:
        for fpath in local_files:
            if not os.path.exists(fpath):
                print(f"    ⚠ ملف غير موجود: {fpath}")
                continue
            print(f"\n  ← ملف محلي: {fpath}")
            count = 0
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if "instruction" in row and "output" in row:
                            all_samples.append({**row, "source": os.path.basename(fpath)})
                            count += 1
                    except json.JSONDecodeError:
                        pass
            print(f"    ✓ {count:,} مثال")

    # ── 4. خلط وحفظ ───────────────────────────────────────────────
    random.shuffle(all_samples)

    with open(combined_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ملف إحصائيات
    stats = {
        "total":    len(all_samples),
        "sources":  {},
        "timestamp": datetime.now().isoformat(),
    }
    for s in all_samples:
        src = s.get("source", "unknown")
        stats["sources"][src] = stats["sources"].get(src, 0) + 1

    with open(os.path.join(data_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n  {'═'*50}")
    print(f"  إجمالي البيانات: {len(all_samples):,} مثال")
    for src, cnt in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        bar = "█" * min(20, cnt // 1000 + 1)
        print(f"  {src:40s} {cnt:>6,}  {bar}")
    print(f"  {'═'*50}")
    print(f"  ✓ محفوظ: {combined_path}")

    return combined_path


# ════════════════════════════════════════════════════════════════════
# المرحلة 2 — Fine-tuning بـ LoRA (برمجة)
# ════════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = "### المهمة:\n{instruction}\n\n### الكود:\n{output}"

def stage2_finetune_code(cfg: dict, data_path: str):
    """Fine-tuning بـ LoRA على بيانات البرمجة الإنجليزية والعربية."""

    print_header("المرحلة 2", "Fine-tuning بـ LoRA — بيانات البرمجة الكاملة")
    _run_lora(
        cfg       = cfg,
        data_path = data_path,
        output    = os.path.join(cfg["output_dir"], "stage2_code"),
        epochs    = cfg["stage2_epochs"],
        batch     = cfg["stage2_batch"],
        lr        = cfg["stage2_lr"],
        max_len   = cfg["stage2_max_len"],
        label     = "Stage2-Code",
    )


# ════════════════════════════════════════════════════════════════════
# المرحلة 3 — Fine-tuning إضافي (عربي فقط)
# ════════════════════════════════════════════════════════════════════

ARABIC_PROMPT = "### السؤال:\n{instruction}\n\n### الجواب:\n{output}"

def stage3_finetune_arabic(cfg: dict, stage2_model: str, data_dir: str):
    """Fine-tuning إضافي على البيانات العربية فقط."""

    print_header("المرحلة 3", "Fine-tuning إضافي — اللغة العربية")

    # كتابة البيانات العربية في ملف مؤقت
    arabic_path = os.path.join(data_dir, "arabic_only.jsonl")
    with open(arabic_path, "w", encoding="utf-8") as f:
        for s in ARABIC_DATA:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ← {len(ARABIC_DATA)} مثال عربي")

    _run_lora(
        cfg        = cfg,
        data_path  = arabic_path,
        output     = os.path.join(cfg["output_dir"], "stage3_arabic"),
        epochs     = cfg["stage3_epochs"],
        batch      = cfg["stage3_batch"],
        lr         = cfg["stage3_lr"],
        max_len    = cfg["stage3_max_len"],
        label      = "Stage3-Arabic",
        base_model = stage2_model,          # يبدأ من نتيجة المرحلة 2
        prompt_tmpl = ARABIC_PROMPT,
    )


# ════════════════════════════════════════════════════════════════════
# دالة LoRA المشتركة
# ════════════════════════════════════════════════════════════════════

def _run_lora(
    cfg, data_path, output, epochs, batch, lr, max_len,
    label, base_model=None, prompt_tmpl=None,
):
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

    prompt_tmpl = prompt_tmpl or PROMPT_TEMPLATE
    model_path  = base_model or cfg["base_model"]

    print(f"  النموذج : {model_path}")
    print(f"  البيانات: {data_path}")
    print(f"  المخرج  : {output}")
    print(f"  Epochs  : {epochs} | LR: {lr} | Batch: {batch}\n")

    # ── تحميل البيانات ─────────────────────────────────────────
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    print(f"  ← {len(samples):,} مثال")

    # ── تحميل النموذج ────────────────────────────────────────
    print(f"\n  تحميل النموذج...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # ── LoRA ───────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r            = cfg["lora_r"],
        lora_alpha   = cfg["lora_alpha"],
        lora_dropout = cfg["lora_dropout"],
        bias         = "none",
        task_type    = TaskType.CAUSAL_LM,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable:,} معامل قابل للتدريب ({trainable/total*100:.2f}%)")

    # ── Tokenization ───────────────────────────────────────────
    texts = [prompt_tmpl.format(**s) for s in samples if "instruction" in s and "output" in s]

    def tokenize(batch):
        result = tokenizer(
            batch["text"],
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  ✓ جاهزة للتدريب: {len(dataset):,} عينة")

    # ── التدريب ──────────────────────────────────────────────
    print(f"\n  بدء التدريب ({epochs} epochs)...")
    os.makedirs(output, exist_ok=True)

    args = TrainingArguments(
        output_dir                  = output,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch,
        gradient_accumulation_steps = 8,
        learning_rate               = lr,
        warmup_ratio                = 0.05,
        lr_scheduler_type           = "cosine",
        logging_steps               = 20,
        save_steps                  = 200,
        save_total_limit            = 2,
        fp16                        = torch.cuda.is_available(),
        optim                       = "adamw_torch",
        report_to                   = "none",
        run_name                    = label,
    )

    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = dataset,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # ── الحفظ ─────────────────────────────────────────────────
    print(f"\n  حفظ النموذج...")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    merged_dir = output + "_merged"
    print(f"  دمج LoRA في النموذج...")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    size_mb = sum(f.stat().st_size for f in Path(merged_dir).rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  ✓ محفوظ: {merged_dir} ({size_mb:.1f} MB)")
    return merged_dir


# ════════════════════════════════════════════════════════════════════
# المرحلة 4 — التقييم التلقائي
# ════════════════════════════════════════════════════════════════════

def stage4_evaluate(cfg: dict, model_dir: str):
    """يُشغّل التقييم الكامل على 50 سؤال برمجي."""

    print_header("المرحلة 4", "التقييم التلقائي — 50 سؤال برمجي")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from evaluate import HFEngine, ModelEvaluator, print_report, QUESTIONS
    except ImportError as e:
        print(f"  ⚠ لا يمكن التقييم: {e}")
        return

    print(f"  النموذج: {model_dir}")
    engine    = HFEngine(model_dir)
    evaluator = ModelEvaluator(engine, "Stars Expert")
    summary   = evaluator.run(QUESTIONS, verbose=True)
    print_report(summary)

    # حفظ النتائج
    results_path = os.path.join(cfg["output_dir"], "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✓ النتائج محفوظة: {results_path}")
    return summary


# ════════════════════════════════════════════════════════════════════
# المرحلة 5 — الضغط (Pruning)
# ════════════════════════════════════════════════════════════════════

def stage5_prune(cfg: dict, model_dir: str) -> str:
    """يضغط النموذج ويحذف الأوزان غير المهمة."""

    print_header("المرحلة 5", f"ضغط النموذج (Magnitude Pruning {cfg['prune_ratio']:.0%})")

    try:
        from prune_model import ModelPruner
    except ImportError:
        print("  ⚠ prune_model.py غير موجود — تخطّي")
        return model_dir

    pruned_dir = model_dir + "_pruned"
    pruner = ModelPruner(model_dir, pruned_dir)
    pruner.load()
    pruner.prune_magnitude(cfg["prune_ratio"])
    pruner.save()
    print(f"  ✓ النموذج المضغوط: {pruned_dir}")
    return pruned_dir


# ════════════════════════════════════════════════════════════════════
# المرحلة 6 — تحويل إلى GGUF
# ════════════════════════════════════════════════════════════════════

def stage6_convert_gguf(cfg: dict, model_dir: str) -> str:
    """يحوّل النموذج إلى GGUF باستخدام llama.cpp أو الكاتب المدمج."""

    print_header("المرحلة 6", f"تحويل إلى GGUF ({cfg['quantization']})")

    gguf_path = os.path.join(cfg["output_dir"], "stars_expert.gguf")
    llama_cpp = cfg["llama_cpp_dir"]

    # محاولة استخدام llama.cpp
    if os.path.exists(llama_cpp):
        convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script):
            convert_script = os.path.join(llama_cpp, "convert.py")

        if os.path.exists(convert_script):
            print(f"  استخدام llama.cpp...")
            import subprocess
            result = subprocess.run(
                [sys.executable, convert_script, model_dir,
                 "--outfile", gguf_path, "--outtype", cfg["quantization"]],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                size = os.path.getsize(gguf_path) / 1024 / 1024 / 1024
                print(f"  ✓ GGUF جاهز: {gguf_path} ({size:.2f} GB)")
                return gguf_path
            else:
                print(f"  ⚠ فشل llama.cpp: {result.stderr[:200]}")

    # استخدام الكاتب المدمج
    print(f"  استخدام الكاتب المدمج (Stars AI GGUF Writer)...")
    try:
        from stars_ai.gguf_converter import GGUFConverter
        converter = GGUFConverter(model_dir)
        gguf_path = converter.convert(gguf_path, quantization=cfg["quantization"])
        print(f"  ✓ GGUF جاهز: {gguf_path}")
    except Exception as e:
        print(f"  ⚠ تعذّر التحويل: {e}")
        print(f"  → ثبّت llama.cpp يدوياً:")
        print(f"    git clone https://github.com/ggerganov/llama.cpp")
        print(f"    cd llama.cpp && make -j4")

    return gguf_path


# ════════════════════════════════════════════════════════════════════
# أدوات مساعدة
# ════════════════════════════════════════════════════════════════════

def print_header(stage: str, title: str):
    """يطبع رأس المرحلة."""
    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"║  {stage}: {title:<54}║")
    print(f"{'╚'+'═'*58+'╝'}\n")


def print_final_summary(cfg: dict, stages_done: list, start_time: float):
    """يطبع ملخصاً نهائياً بكل ما تم."""
    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)

    print(f"\n{'╔'+'═'*58+'╗'}")
    print(f"║{'التدريب الشامل اكتمل':^58}║")
    print(f"{'╠'+'═'*58+'╣'}")
    print(f"║  الوقت الكلي: {h}:{m:02d}:{s:02d}{'':38}║")
    print(f"║  المراحل المنجزة: {', '.join(stages_done):<45}║")
    print(f"{'╠'+'═'*58+'╣'}")
    print(f"║  المخرجات:{'':49}║")
    print(f"║    {os.path.join(cfg['output_dir'], 'stage2_code_merged'):<56}║")
    print(f"║    {os.path.join(cfg['output_dir'], 'stage3_arabic_merged'):<56}║")
    print(f"║    {os.path.join(cfg['output_dir'], 'stars_expert.gguf'):<56}║")
    print(f"{'╠'+'═'*58+'╣'}")
    print(f"║  الخطوات التالية:{'':42}║")
    print(f"║  python chat.py --gguf {os.path.join(cfg['output_dir'], 'stars_expert.gguf'):<35}║")
    print(f"║  python evaluate.py --gguf {os.path.join(cfg['output_dir'], 'stars_expert.gguf'):<31}║")
    print(f"{'╚'+'═'*58+'╝'}\n")


# ════════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — التدريب الشامل الكامل",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تشغيل جميع المراحل
  python train_all.py

  # مراحل محددة (1=جمع البيانات، 2=تدريب برمجة، 3=تدريب عربي)
  python train_all.py --stages 1,2

  # مع بيانات محلية إضافية
  python train_all.py --local-data ./my_data.jsonl ./more_data.jsonl

  # نموذج مختلف
  python train_all.py --model mistralai/Mistral-7B-Instruct-v0.2

  # بدون تحويل GGUF
  python train_all.py --stages 1,2,3,4,5

  # تدريب على GPU
  python train_all.py --batch 8 --epochs 5
        """,
    )
    parser.add_argument("--stages",      default="1,2,3,4,5,6",
                        help="المراحل المطلوبة مفصولة بفواصل (افتراضي: 1,2,3,4,5,6)")
    parser.add_argument("--model",       default=None,
                        help="النموذج الأساسي (افتراضي: microsoft/phi-2)")
    parser.add_argument("--output",      default=None,
                        help="مجلد المخرجات (افتراضي: ./models/stars_expert)")
    parser.add_argument("--local-data",  nargs="+", default=None, metavar="FILE",
                        help="ملفات JSONL محلية إضافية")
    parser.add_argument("--batch",       type=int, default=None, help="حجم الدفعة")
    parser.add_argument("--epochs",      type=int, default=None, help="عدد epochs")
    parser.add_argument("--prune-ratio", type=float, default=None, help="نسبة الضغط")
    parser.add_argument("--llama-cpp",   default=None, help="مسار مجلد llama.cpp")
    args = parser.parse_args()

    # تطبيق التعديلات على الإعدادات
    cfg = DEFAULT_CONFIG.copy()
    if args.model:       cfg["base_model"]   = args.model
    if args.output:      cfg["output_dir"]   = args.output
    if args.batch:       cfg["stage2_batch"] = cfg["stage3_batch"] = args.batch
    if args.epochs:      cfg["stage2_epochs"] = args.epochs
    if args.prune_ratio: cfg["prune_ratio"]  = args.prune_ratio
    if args.llama_cpp:   cfg["llama_cpp_dir"] = args.llama_cpp

    stages = [s.strip() for s in args.stages.split(",")]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── عرض الخطة ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Stars AI — التدريب الشامل الكامل")
    print(f"{'═'*60}")
    print(f"  النموذج الأساسي : {cfg['base_model']}")
    print(f"  مجلد المخرجات  : {cfg['output_dir']}")
    print(f"  المراحل المطلوبة: {', '.join(stages)}")
    if args.local_data:
        print(f"  بيانات إضافية  : {args.local_data}")

    print(f"\n  ─── خطة مصادر البيانات ───────────────────────────")
    for ds in cfg["hf_datasets"]:
        print(f"    ✓ {ds['label']:35s} ({ds['max_samples']:>6,} مثال)")
    print(f"    ✓ {'البيانات العربية المدمجة':35s} ({len(ARABIC_DATA):>6,} مثال)")
    if args.local_data:
        for f in args.local_data:
            print(f"    ✓ {os.path.basename(f):35s} (محلي)")
    print(f"  {'─'*52}")

    start_time   = time.time()
    stages_done  = []
    data_path    = None
    last_merged  = None
    data_dir     = os.path.join(cfg["output_dir"], "data")

    # ════════════ المرحلة 1 ════════════
    if "1" in stages:
        data_path = stage1_collect_data(cfg, args.local_data)
        stages_done.append("1")
    else:
        # محاولة استخدام بيانات موجودة
        default_data = os.path.join(cfg["output_dir"], "data", "combined_dataset.jsonl")
        if os.path.exists(default_data):
            data_path = default_data
            print(f"\n  [المرحلة 1] تخطّي — استخدام بيانات موجودة: {data_path}")
        else:
            print("\n  ⚠ لم يتم تشغيل المرحلة 1 ولا توجد بيانات مسبقة.")
            print("    شغّل: python train_all.py --stages 1,2 أولاً")
            if "2" in stages:
                sys.exit(1)

    # ════════════ المرحلة 2 ════════════
    if "2" in stages and data_path:
        stage2_finetune_code(cfg, data_path)
        last_merged = os.path.join(cfg["output_dir"], "stage2_code_merged")
        stages_done.append("2")

    # ════════════ المرحلة 3 ════════════
    if "3" in stages:
        base_for_3 = last_merged or cfg["base_model"]
        stage3_finetune_arabic(cfg, base_for_3, data_dir if data_path else cfg["output_dir"])
        last_merged = os.path.join(cfg["output_dir"], "stage3_arabic_merged")
        stages_done.append("3")

    # ════════════ المرحلة 4 ════════════
    if "4" in stages and last_merged:
        stage4_evaluate(cfg, last_merged)
        stages_done.append("4")
    elif "4" in stages:
        print("\n  [المرحلة 4] تخطّي — لا يوجد نموذج مدرّب بعد")

    # ════════════ المرحلة 5 ════════════
    if "5" in stages and last_merged:
        last_merged = stage5_prune(cfg, last_merged)
        stages_done.append("5")
    elif "5" in stages:
        print("\n  [المرحلة 5] تخطّي — لا يوجد نموذج مدرّب بعد")

    # ════════════ المرحلة 6 ════════════
    if "6" in stages and last_merged:
        stage6_convert_gguf(cfg, last_merged)
        stages_done.append("6")
    elif "6" in stages:
        print("\n  [المرحلة 6] تخطّي — لا يوجد نموذج مدرّب بعد")

    # ── الملخص النهائي ────────────────────────────────────────
    print_final_summary(cfg, stages_done, start_time)


if __name__ == "__main__":
    main()

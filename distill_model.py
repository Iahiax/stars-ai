"""
═══════════════════════════════════════════════════════════════
  Stars AI — تقطير المعرفة (Knowledge Distillation)
  يدرّب نموذجاً صغيراً ليتعلم من نموذج كبير (Teacher → Student)
═══════════════════════════════════════════════════════════════

الفكرة:
  النموذج الكبير (Teacher): GPT-4 / Llama-70B / Mistral-7B
  النموذج الصغير (Student): Phi-2 / TinyLlama / DistilGPT-2
  
  بدلاً من التعلم من البيانات فقط، يتعلم النموذج الصغير
  من التوزيع الاحتمالي لمخرجات النموذج الكبير.
  النتيجة: نموذج صغير أذكى بكثير من تدريبه المباشر.

الاستخدام:
  # تقطير من Mistral-7B إلى Phi-2 (بدون API خارجي)
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 --student microsoft/phi-2

  # تقطير باستخدام GPT-4 عبر OpenAI API
  python distill_model.py --teacher gpt-4o --student microsoft/phi-2 --use-api

  # تقطير مع بيانات خاصة
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 \\
      --student microsoft/phi-2 --data ./data/my_data.jsonl

  # استئناف من نقطة تفتيش
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 \\
      --student microsoft/phi-2 --resume
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# بيانات التقطير الافتراضية (أسئلة برمجية)
# ════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    "اكتب دالة Python تحسب المضروب بالتعاود",
    "اكتب خوارزمية Binary Search",
    "اشرح الـ decorator في Python مع مثال",
    "اكتب class Stack يدعم push وpop",
    "اكتب generator يولّد أرقام فيبوناتشي",
    "ما الفرق بين list وtuple وset؟",
    "اكتب async function تجلب بيانات من URL",
    "اكتب decorator يقيس وقت تنفيذ الدالة",
    "اكتب دالة تجد أقصر مسار BFS في graph",
    "اكتب مثال على metaclass في Python",
    "اكتب خوارزمية Merge Sort",
    "اكتب class Queue يدعم enqueue وdequeue",
    "ما هو الـ GIL في Python؟",
    "اكتب دالة Two Sum بتعقيد O(n)",
    "اكتب context manager باستخدام __enter__",
    "اكتب regex يستخرج الإيميلات من نص",
    "اكتب دالة تحسب N من فيبوناتشي بـ memoization",
    "اكتب مثال على الـ polymorphism",
    "اكتب كود يقرأ CSV ويحلّله",
    "اكتب unit test لدالة حساب المساحة",
] * 5  # 100 مثال


PROMPT_TEMPLATE = "### المهمة:\n{prompt}\n\n### الكود:\n"


# ════════════════════════════════════════════════════════════════
# جلب إجابات Teacher عبر API (GPT-4 / Claude)
# ════════════════════════════════════════════════════════════════

class APITeacher:
    """يستخدم GPT-4o أو Claude كـ Teacher عبر API."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ pip install openai")
            sys.exit(1)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ export OPENAI_API_KEY=sk-...")
            sys.exit(1)

        self.client = __import__("openai").OpenAI(api_key=api_key)
        self.model  = model
        print(f"  ✓ Teacher API: {model}")

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "أنت مساعد برمجة خبير. أجب بكود Python واضح ومشروح."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=600,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    ⚠ خطأ API: {e}")
            return ""


# ════════════════════════════════════════════════════════════════
# جلب إجابات Teacher من نموذج HuggingFace محلي
# ════════════════════════════════════════════════════════════════

class LocalTeacher:
    """يستخدم نموذج HuggingFace محلياً كـ Teacher."""

    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("❌ pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"  تحميل Teacher: {model_name}")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
        print(f"  ✓ Teacher جاهز على {self.device.upper()}")

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=400, temperature=0.3,
                top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full   = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full[len(self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)):]
        return answer.strip()

    def get_logits(self, input_ids):
        """يُعيد logits النموذج (للتقطير الناعم)."""
        import torch
        with torch.no_grad():
            output = self.model(input_ids.to(self.device))
        return output.logits


# ════════════════════════════════════════════════════════════════
# الخطوة 1: توليد بيانات Teacher
# ════════════════════════════════════════════════════════════════

def generate_teacher_data(teacher, prompts: list[str], output_path: str,
                           resume: bool = False) -> str:
    """يجمع إجابات Teacher ويحفظها كبيانات تدريب."""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # استئناف — تحميل ما تم توليده مسبقاً
    existing = set()
    if resume and os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["instruction"])
                except Exception:
                    pass
        print(f"  ← استئناف: {len(existing)} مثال موجود مسبقاً")

    print(f"\n[خطوة 1] توليد {len(prompts):,} إجابة من Teacher...")
    print(f"  الحفظ في: {output_path}\n")

    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            if prompt in existing:
                continue

            print(f"  [{i:03d}/{len(prompts)}] {prompt[:50]}...", end=" ", flush=True)
            t0     = time.time()
            answer = teacher.generate(PROMPT_TEMPLATE.format(prompt=prompt))
            elapsed = time.time() - t0

            if answer and len(answer.strip()) > 10:
                record = {"instruction": prompt, "output": answer, "source": "teacher_distill"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                count += 1
                print(f"✓ ({elapsed:.1f}ث)")
            else:
                print(f"✗ فارغ")

    print(f"\n  ✓ تم توليد {count} إجابة جديدة")
    return output_path


# ════════════════════════════════════════════════════════════════
# الخطوة 2: تدريب Student بـ LoRA على بيانات Teacher
# ════════════════════════════════════════════════════════════════

def train_student(student_model: str, data_path: str, output_dir: str,
                  epochs: int = 3, resume: bool = False):
    """يدرّب النموذج الصغير (Student) على إجابات Teacher."""

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
        print(f"❌ {e}\npip install transformers peft datasets torch accelerate")
        sys.exit(1)

    print(f"\n[خطوة 2] تدريب Student: {student_model}")

    # ── تحميل البيانات ──────────────────────────────────────────
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    print(f"  ← {len(samples):,} مثال من Teacher")

    # ── تحميل Student ────────────────────────────────────────────
    resume_from = output_dir if (resume and os.path.exists(output_dir)) else student_model
    print(f"  النموذج: {resume_from}")

    tokenizer = AutoTokenizer.from_pretrained(resume_from, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resume_from, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # ── LoRA ─────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                         "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA: {trainable:,} معامل ({trainable/sum(p.numel() for p in model.parameters())*100:.2f}%)")

    # ── Tokenization ─────────────────────────────────────────────
    template = "### المهمة:\n{instruction}\n\n### الكود:\n{output}"
    texts    = [template.format(**s) for s in samples]

    def tokenize(batch):
        result = tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"  ✓ {len(dataset):,} عينة جاهزة")

    # ── التدريب ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Weights & Biases إذا كان متاحاً
    report_to = "none"
    try:
        import wandb
        if os.getenv("WANDB_API_KEY"):
            wandb.init(project="stars-ai-distill", name="student-training")
            report_to = "wandb"
    except ImportError:
        pass

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to=report_to,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # ── الحفظ ────────────────────────────────────────────────────
    print(f"\n  حفظ Student المدرّب...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    merged_dir = output_dir + "_merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"  ✓ محفوظ: {merged_dir}")
    return merged_dir


# ════════════════════════════════════════════════════════════════
# الخطوة 3: مقارنة Teacher vs Student
# ════════════════════════════════════════════════════════════════

def compare_teacher_student(teacher, student_dir: str, n_samples: int = 5):
    """يقارن جودة إجابات Teacher مقابل Student."""

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        return

    print(f"\n[خطوة 3] مقارنة Teacher vs Student ({n_samples} أسئلة)\n")

    tokenizer = AutoTokenizer.from_pretrained(student_dir, trust_remote_code=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    student_model = AutoModelForCausalLM.from_pretrained(
        student_dir, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    student_model.eval()

    test_prompts = DEFAULT_PROMPTS[:n_samples]

    for i, prompt in enumerate(test_prompts, 1):
        formatted = PROMPT_TEMPLATE.format(prompt=prompt)
        print(f"  ── السؤال {i}: {prompt}")

        # Teacher
        t_ans = teacher.generate(formatted)
        print(f"  Teacher  : {t_ans[:200].strip()}")

        # Student
        inputs = tokenizer(formatted, return_tensors="pt", max_length=256,
                           truncation=True).to(device)
        with torch.no_grad():
            outputs = student_model.generate(
                **inputs, max_new_tokens=200, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        full  = tokenizer.decode(outputs[0], skip_special_tokens=True)
        s_ans = full[len(tokenizer.decode(inputs["input_ids"][0],
                                           skip_special_tokens=True)):].strip()
        print(f"  Student  : {s_ans[:200].strip()}")
        print(f"  {'─'*60}")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — تقطير المعرفة (Knowledge Distillation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 --student microsoft/phi-2
  python distill_model.py --teacher gpt-4o --student microsoft/phi-2 --use-api
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 --student microsoft/phi-2 --resume
  python distill_model.py --teacher mistralai/Mistral-7B-v0.1 --student microsoft/phi-2 \\
      --data ./data/my_prompts.txt --epochs 5
        """,
    )
    parser.add_argument("--teacher",    default="mistralai/Mistral-7B-v0.1",
                        help="النموذج الكبير (Teacher)")
    parser.add_argument("--student",    default="microsoft/phi-2",
                        help="النموذج الصغير (Student)")
    parser.add_argument("--use-api",    action="store_true",
                        help="استخدام OpenAI API للـ Teacher (يحتاج OPENAI_API_KEY)")
    parser.add_argument("--data",       help="ملف نصي بأسئلة إضافية (سطر = سؤال)")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--output",     default="./models/distilled_student")
    parser.add_argument("--resume",     action="store_true",
                        help="استئناف التوليد/التدريب من حيث توقف")
    parser.add_argument("--skip-gen",   action="store_true",
                        help="تخطي توليد بيانات Teacher (استخدم ملف موجود)")
    parser.add_argument("--compare",    action="store_true",
                        help="مقارنة Teacher vs Student بعد التدريب")
    args = parser.parse_args()

    print("═" * 60)
    print("  Stars AI — Knowledge Distillation")
    print(f"  Teacher : {args.teacher}")
    print(f"  Student : {args.student}")
    print("═" * 60)

    data_path = os.path.join(args.output, "teacher_data.jsonl")

    # ── تحضير الأسئلة ────────────────────────────────────────────
    prompts = list(DEFAULT_PROMPTS)
    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            extra = [line.strip() for line in f if line.strip()]
        prompts.extend(extra)
        print(f"  + {len(extra)} سؤال إضافي من {args.data}")
    print(f"  إجمالي الأسئلة: {len(prompts)}\n")

    # ── تهيئة Teacher ────────────────────────────────────────────
    if args.use_api:
        teacher = APITeacher(model=args.teacher)
    else:
        teacher = LocalTeacher(model_name=args.teacher)

    # ── الخطوة 1: توليد بيانات Teacher ────────────────────────────
    if not args.skip_gen:
        generate_teacher_data(teacher, prompts, data_path, resume=args.resume)
    else:
        print(f"[تخطي الخطوة 1] استخدام: {data_path}")

    # ── الخطوة 2: تدريب Student ────────────────────────────────────
    merged = train_student(
        student_model = args.student,
        data_path     = data_path,
        output_dir    = args.output,
        epochs        = args.epochs,
        resume        = args.resume,
    )

    # ── الخطوة 3: مقارنة (اختياري) ────────────────────────────────
    if args.compare:
        compare_teacher_student(teacher, merged)

    print("\n" + "═" * 60)
    print("  ✓ التقطير اكتمل!")
    print(f"  النموذج المدرّب: {merged}")
    print(f"\n  الخطوات التالية:")
    print(f"  python evaluate.py   --model {merged}")
    print(f"  python benchmark.py  --my-model {merged}")
    print(f"  python sync_to_hf.py --model {merged} --repo your-username/stars-distilled")
    print("═" * 60)


if __name__ == "__main__":
    main()

"""
═══════════════════════════════════════════════════════════════
  Stars AI — توليد بيانات التدريب تلقائياً
  يستخدم GPT-4 لإنشاء آلاف أسئلة وأجوبة برمجية
═══════════════════════════════════════════════════════════════

التشغيل:
  python generate_data.py --count 500 --topic python
  python generate_data.py --count 1000 --topic algorithms --output my_data.jsonl
  python generate_data.py --count 200 --topic arabic_python
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stars_ai.key_manager import KeyManager


# ── موضوعات التوليد ───────────────────────────────────────────────────────────

TOPICS = {
    "python": [
        "دوال Python الأساسية", "معالجة القوائم", "القواميس والمجموعات",
        "OOP والكلاسات", "decorators وgenerators", "معالجة الأخطاء (exceptions)",
        "الملفات والـ I/O", "المكتبات المعيارية", "list comprehension",
        "lambda والدوال العليا",
    ],
    "algorithms": [
        "خوارزميات الترتيب", "خوارزميات البحث", "هياكل البيانات",
        "البرمجة الديناميكية", "الخوارزميات الجشعة", "الرسوم البيانية (Graphs)",
        "المكدسات والطوابير", "الأشجار الثنائية", "خوارزميات التعقيد",
        "مسائل LeetCode الشائعة",
    ],
    "web": [
        "Flask API endpoints", "FastAPI مع Pydantic", "Django models",
        "REST API design", "Authentication و JWT", "WebSockets",
        "HTTP requests و responses", "middleware", "database ORM",
        "API testing",
    ],
    "arabic_python": [
        "اكتب كوداً لحل مسألة برمجية شائعة",
        "اشرح مفهوماً برمجياً بالعربي مع مثال",
        "كيف تحل هذه المشكلة في Python؟",
        "اكتب دالة تقوم بـ...",
        "ما الفرق بين ... و ... في Python؟",
    ],
    "mixed": [
        "python", "algorithms", "web", "arabic_python"
    ],
}

# ── قوالب الطلبات لـ GPT-4 ───────────────────────────────────────────────────

SYSTEM_PROMPT = """أنت خبير برمجة متخصص في Python وتوليد بيانات تدريب عالية الجودة.
مهمتك: توليد أسئلة وأجوبة برمجية دقيقة ومتنوعة.
القواعد:
- الكود يجب أن يكون صحيحاً ويعمل فعلاً
- الإجابات تكون مفصّلة مع شرح
- تنوّع في مستوى الصعوبة (سهل/متوسط/صعب)
- الإجابة بنفس لغة السؤال"""

GENERATION_PROMPTS = {
    "python": """أنشئ {n} سؤالاً وجواباً برمجياً في Python حول: {subtopic}
الصيغة المطلوبة (JSON array):
[
  {{"instruction": "السؤال هنا", "output": "الكود والشرح هنا"}},
  ...
]
تأكد: الكود صحيح، الإجابات متنوعة الصعوبة.""",

    "algorithms": """أنشئ {n} مسألة خوارزمية مع حلها الكامل في Python حول: {subtopic}
الصيغة:
[
  {{"instruction": "وصف المسألة", "output": "الكود الكامل مع شرح التعقيد"}},
  ...
]""",

    "arabic_python": """أنشئ {n} سؤالاً وجواباً برمجياً باللغة العربية.
السؤال والجواب كلاهما بالعربي، الكود بـ Python.
الصيغة:
[
  {{"instruction": "سؤال عربي", "output": "شرح عربي + كود Python"}},
  ...
]""",
}


# ── الموّلد الرئيسي ──────────────────────────────────────────────────────────

class DataGenerator:
    """يولّد بيانات التدريب تلقائياً باستخدام GPT-4."""

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ pip install openai")
            sys.exit(1)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate_batch(self, topic: str, subtopic: str, batch_size: int = 10) -> list[dict]:
        """يولّد دفعة من الأمثلة."""
        template = GENERATION_PROMPTS.get(topic, GENERATION_PROMPTS["python"])
        prompt   = template.format(n=batch_size, subtopic=subtopic)

        try:
            response = self.client.chat.completions.create(
                model    = "gpt-4o",
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature   = 0.8,
                max_tokens    = 3000,
                response_format = {"type": "json_object"},
            )
            text = response.choices[0].message.content
            data = json.loads(text)

            # استخراج القائمة من الـ JSON
            if isinstance(data, list):
                return data
            for val in data.values():
                if isinstance(val, list):
                    return val
            return []

        except json.JSONDecodeError:
            # محاولة تصحيح الـ JSON
            try:
                start = text.find("[")
                end   = text.rfind("]") + 1
                return json.loads(text[start:end])
            except Exception:
                return []
        except Exception as e:
            print(f"  ⚠ خطأ في الطلب: {e}")
            time.sleep(5)
            return []

    def generate(
        self,
        topic:     str,
        count:     int,
        output:    str,
        batch:     int = 10,
    ) -> int:
        """يولّد عدداً كاملاً من الأمثلة ويحفظها."""
        subtopics = TOPICS.get(topic, TOPICS["python"])
        if topic == "mixed":
            subtopics = TOPICS["python"] + TOPICS["algorithms"]

        total_generated = 0
        batches_needed  = (count + batch - 1) // batch

        print(f"\n[توليد البيانات]")
        print(f"  الموضوع  : {topic}")
        print(f"  العدد    : {count}")
        print(f"  الدفعات  : {batches_needed} × {batch}")
        print(f"  المخرج   : {output}\n")

        with open(output, "w", encoding="utf-8") as f:
            for i in range(batches_needed):
                if total_generated >= count:
                    break

                subtopic = subtopics[i % len(subtopics)]
                need     = min(batch, count - total_generated)

                print(f"  [{i+1:03d}/{batches_needed}] {subtopic[:40]}...", end=" ", flush=True)
                examples = self.generate_batch(topic, subtopic, need)

                saved = 0
                for ex in examples:
                    if isinstance(ex, dict) and "instruction" in ex and "output" in ex:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        saved += 1
                        total_generated += 1

                print(f"✓ {saved} مثال  (إجمالي: {total_generated})")

                # تأخير لتجنّب حدود الـ API
                if i < batches_needed - 1:
                    time.sleep(1)

        print(f"\n  ✓ تم حفظ {total_generated} مثال في: {output}")
        return total_generated


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stars AI — توليد بيانات التدريب")
    parser.add_argument("--count",  type=int, default=100,
                        help="عدد الأمثلة المطلوبة (افتراضي: 100)")
    parser.add_argument("--topic",  default="python",
                        choices=list(TOPICS.keys()),
                        help="موضوع البيانات (افتراضي: python)")
    parser.add_argument("--output", default=None,
                        help="ملف الحفظ .jsonl (افتراضي: data_<topic>.jsonl)")
    parser.add_argument("--batch",  type=int, default=10,
                        help="حجم كل دفعة من GPT-4 (افتراضي: 10)")
    args = parser.parse_args()

    output = args.output or f"./data/data_{args.topic}.jsonl"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # جلب مفتاح OpenAI
    km = KeyManager()
    try:
        api_key = km.get_key("openai")
    except ValueError:
        print("❌ مفتاح OpenAI غير موجود.")
        print("   export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    gen   = DataGenerator(api_key)
    total = gen.generate(args.topic, args.count, output, args.batch)

    print(f"\n  البيانات جاهزة للاستخدام في finetune_lora.py:")
    print(f'  "dataset_source": "local_file",')
    print(f'  "local_file": "{output}",')


if __name__ == "__main__":
    main()

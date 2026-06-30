"""
═══════════════════════════════════════════════════════════════
  Stars AI — تقييم النموذج تلقائياً
  يختبر النموذج على 50 سؤال برمجي ويعطيك نسبة الدقة
  قبل التدريب وبعده لترى الفرق بوضوح
═══════════════════════════════════════════════════════════════

طريقة التشغيل:

  تقييم نموذج GGUF:
    python evaluate.py --gguf ./models/stars_expert.gguf

  تقييم نموذج HuggingFace:
    python evaluate.py --model ./models/stars_expert_merged

  مقارنة قبل وبعد التدريب:
    python evaluate.py --before microsoft/phi-2 --after ./models/stars_expert_merged

  حفظ النتائج في ملف JSON:
    python evaluate.py --gguf ./models/stars_expert.gguf --save results.json

بعد التقييم يمكنك:
  python benchmark.py   --my-model ./models/stars_expert_merged
  python auto_improve.py --model   ./models/stars_expert_merged --rounds 3
  python compare_models.py --models phi-2 mistral --gguf ./models/stars_expert.gguf
"""

import os
import sys
import json
import time
import argparse
import datetime
from dataclasses import dataclass, field, asdict


# ════════════════════════════════════════════════════════════════
# قاعدة الأسئلة البرمجية (50 سؤال)
# ════════════════════════════════════════════════════════════════

QUESTIONS = [
    # ── Python الأساسي (10 أسئلة) ────────────────────────────────
    {
        "id": 1, "category": "Python أساسي",
        "question": "اكتب دالة Python تجمع كل أرقام قائمة",
        "keywords": ["def", "sum", "return"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 2, "category": "Python أساسي",
        "question": "اكتب دالة تتحقق إذا كان رقم زوجياً أم لا",
        "keywords": ["def", "return", "%", "2"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 3, "category": "Python أساسي",
        "question": "اكتب حلقة تطبع أرقام من 1 إلى 10",
        "keywords": ["for", "range", "print"],
        "must_contain": ["for", "print"],
    },
    {
        "id": 4, "category": "Python أساسي",
        "question": "اكتب دالة تعكس نص (string reversal)",
        "keywords": ["def", "return", "[::-1]"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 5, "category": "Python أساسي",
        "question": "اكتب دالة تحسب المضروب (factorial) لرقم",
        "keywords": ["def", "return", "factorial", "*"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 6, "category": "Python أساسي",
        "question": "كيف تنشئ قاموس (dictionary) في Python وتضيف له عنصراً؟",
        "keywords": ["{", "}", "=", ":"],
        "must_contain": ["{"],
    },
    {
        "id": 7, "category": "Python أساسي",
        "question": "اكتب list comprehension يُعيد مربعات الأرقام من 1 إلى 10",
        "keywords": ["[", "for", "in", "range"],
        "must_contain": ["for", "in", "range"],
    },
    {
        "id": 8, "category": "Python أساسي",
        "question": "اكتب دالة تجد أكبر عنصر في قائمة",
        "keywords": ["def", "max", "return"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 9, "category": "Python أساسي",
        "question": "كيف تفتح ملفاً وتقرأ محتواه في Python؟",
        "keywords": ["open", "read", "with"],
        "must_contain": ["open"],
    },
    {
        "id": 10, "category": "Python أساسي",
        "question": "اكتب دالة تتحقق إذا كانت كلمة palindrome (تُقرأ بنفس الطريقة من الاتجاهين)",
        "keywords": ["def", "return", "[::-1]"],
        "must_contain": ["def", "return"],
    },

    # ── الدوال والـ OOP (10 أسئلة) ────────────────────────────────
    {
        "id": 11, "category": "OOP",
        "question": "اكتب class للطالب (Student) مع خصائص الاسم والعمر",
        "keywords": ["class", "def", "__init__", "self"],
        "must_contain": ["class", "__init__", "self"],
    },
    {
        "id": 12, "category": "OOP",
        "question": "اشرح الوراثة (Inheritance) في Python مع مثال",
        "keywords": ["class", "def", "__init__", "super"],
        "must_contain": ["class"],
    },
    {
        "id": 13, "category": "OOP",
        "question": "ما هو الـ decorator في Python؟ اكتب مثالاً عليه",
        "keywords": ["def", "@", "return", "wrapper"],
        "must_contain": ["def", "@"],
    },
    {
        "id": 14, "category": "OOP",
        "question": "اكتب class Stack يدعم push وpop وisEmpty",
        "keywords": ["class", "def", "append", "pop"],
        "must_contain": ["class", "def"],
    },
    {
        "id": 15, "category": "OOP",
        "question": "اكتب property decorator للتحقق من صحة قيمة عند الإسناد",
        "keywords": ["@property", "def", "setter"],
        "must_contain": ["@property", "def"],
    },
    {
        "id": 16, "category": "OOP",
        "question": "ما الفرق بين __str__ و__repr__ في Python؟",
        "keywords": ["__str__", "__repr__", "return"],
        "must_contain": ["__str__"],
    },
    {
        "id": 17, "category": "OOP",
        "question": "اكتب context manager باستخدام __enter__ و__exit__",
        "keywords": ["def", "__enter__", "__exit__", "class"],
        "must_contain": ["__enter__", "__exit__"],
    },
    {
        "id": 18, "category": "OOP",
        "question": "اكتب مثالاً على الـ polymorphism في Python",
        "keywords": ["class", "def", "super"],
        "must_contain": ["class", "def"],
    },
    {
        "id": 19, "category": "OOP",
        "question": "ما هو الـ staticmethod وكيف يختلف عن classmethod؟",
        "keywords": ["@staticmethod", "@classmethod", "cls"],
        "must_contain": ["@staticmethod"],
    },
    {
        "id": 20, "category": "OOP",
        "question": "اكتب class Queue (طابور) يدعم enqueue وdequeue",
        "keywords": ["class", "def", "append", "pop"],
        "must_contain": ["class", "def"],
    },

    # ── الخوارزميات (10 أسئلة) ───────────────────────────────────
    {
        "id": 21, "category": "خوارزميات",
        "question": "اكتب خوارزمية Binary Search في Python",
        "keywords": ["def", "while", "mid", "return"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 22, "category": "خوارزميات",
        "question": "اكتب خوارزمية Bubble Sort",
        "keywords": ["def", "for", "swap", "return"],
        "must_contain": ["def", "for"],
    },
    {
        "id": 23, "category": "خوارزميات",
        "question": "اكتب دالة تحسب أرقام فيبوناتشي (Fibonacci) بشكل تعاودي",
        "keywords": ["def", "return", "fibonacci", "+"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 24, "category": "خوارزميات",
        "question": "اكتب دالة تجد الأرقام الأولية (Prime Numbers) حتى N",
        "keywords": ["def", "for", "return", "prime"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 25, "category": "خوارزميات",
        "question": "اكتب خوارزمية Merge Sort",
        "keywords": ["def", "merge", "return", "left", "right"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 26, "category": "خوارزميات",
        "question": "اكتب دالة تحسب أقصر مسار بين نقطتين (BFS على graph)",
        "keywords": ["def", "queue", "visited", "return"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 27, "category": "خوارزميات",
        "question": "اكتب دالة لحل مسألة Two Sum (إيجاد زوجين مجموعهما = target)",
        "keywords": ["def", "for", "return", "dict"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 28, "category": "خوارزميات",
        "question": "اكتب دالة تعكس Linked List",
        "keywords": ["def", "next", "return", "prev"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 29, "category": "خوارزميات",
        "question": "اكتب دالة تتحقق من توازن أقواس نص ما (Balanced Parentheses)",
        "keywords": ["def", "stack", "return", "append", "pop"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 30, "category": "خوارزميات",
        "question": "اكتب دالة تجد أطول تسلسل متصاعد (Longest Increasing Subsequence)",
        "keywords": ["def", "for", "return", "dp"],
        "must_contain": ["def", "return"],
    },

    # ── Python متقدم (10 أسئلة) ──────────────────────────────────
    {
        "id": 31, "category": "Python متقدم",
        "question": "اكتب generator يولّد أرقام فيبوناتشي بشكل لانهائي",
        "keywords": ["def", "yield", "while"],
        "must_contain": ["yield"],
    },
    {
        "id": 32, "category": "Python متقدم",
        "question": "اكتب async function تجلب بيانات من URL",
        "keywords": ["async", "await", "def"],
        "must_contain": ["async", "await"],
    },
    {
        "id": 33, "category": "Python متقدم",
        "question": "اشرح الـ GIL في Python وكيف تتعامل معه",
        "keywords": ["GIL", "thread", "multiprocessing"],
        "must_contain": ["GIL"],
    },
    {
        "id": 34, "category": "Python متقدم",
        "question": "اكتب decorator يُحفّظ نتائج دالة (memoization/cache)",
        "keywords": ["def", "@", "cache", "dict", "return"],
        "must_contain": ["def", "return"],
    },
    {
        "id": 35, "category": "Python متقدم",
        "question": "ما الفرق بين map وfilter وreduce؟ أعطِ أمثلة",
        "keywords": ["map", "filter", "lambda", "reduce"],
        "must_contain": ["map", "filter"],
    },
    {
        "id": 36, "category": "Python متقدم",
        "question": "اكتب metaclass بسيط في Python",
        "keywords": ["class", "type", "metaclass", "def"],
        "must_contain": ["class", "metaclass"],
    },
    {
        "id": 37, "category": "Python متقدم",
        "question": "اكتب كود يستخدم threading لتشغيل مهام بالتوازي",
        "keywords": ["import", "threading", "Thread", "start"],
        "must_contain": ["threading", "Thread"],
    },
    {
        "id": 38, "category": "Python متقدم",
        "question": "اكتب كود يقرأ ويكتب ملف JSON في Python",
        "keywords": ["import", "json", "open", "load", "dump"],
        "must_contain": ["json", "open"],
    },
    {
        "id": 39, "category": "Python متقدم",
        "question": "اشرح الـ descriptor protocol في Python",
        "keywords": ["__get__", "__set__", "def", "class"],
        "must_contain": ["__get__"],
    },
    {
        "id": 40, "category": "Python متقدم",
        "question": "اكتب type hints كاملة لدالة تأخذ قائمة وترجع قاموساً",
        "keywords": ["def", "List", "Dict", "->", ":"],
        "must_contain": ["->", ":"],
    },

    # ── قواعد البيانات والملفات (10 أسئلة) ──────────────────────
    {
        "id": 41, "category": "قواعد البيانات",
        "question": "اكتب كود Python يتصل بـ SQLite ويُنشئ جدولاً",
        "keywords": ["sqlite3", "connect", "cursor", "CREATE", "TABLE"],
        "must_contain": ["sqlite3", "connect"],
    },
    {
        "id": 42, "category": "قواعد البيانات",
        "question": "اكتب دالة تُدرج بيانات في قاعدة SQLite بأمان (parameterized query)",
        "keywords": ["cursor", "execute", "?", "commit"],
        "must_contain": ["execute", "commit"],
    },
    {
        "id": 43, "category": "قواعد البيانات",
        "question": "اكتب كود يقرأ ملف CSV ويحلّله باستخدام Python",
        "keywords": ["import", "csv", "open", "reader"],
        "must_contain": ["csv", "open"],
    },
    {
        "id": 44, "category": "قواعد البيانات",
        "question": "اكتب كود يستخدم requests لجلب بيانات من API",
        "keywords": ["import", "requests", "get", "json"],
        "must_contain": ["requests", "get"],
    },
    {
        "id": 45, "category": "قواعد البيانات",
        "question": "اكتب class يتعامل مع ملفات JSON (قراءة/كتابة/تحديث)",
        "keywords": ["class", "def", "json", "open"],
        "must_contain": ["class", "json"],
    },
    {
        "id": 46, "category": "قواعد البيانات",
        "question": "اكتب regex يستخرج كل عناوين البريد الإلكتروني من نص",
        "keywords": ["import", "re", "findall", "@"],
        "must_contain": ["re", "findall"],
    },
    {
        "id": 47, "category": "قواعد البيانات",
        "question": "اكتب كود يحذف ملفات قديمة (أكثر من 7 أيام) من مجلد",
        "keywords": ["import", "os", "datetime", "remove"],
        "must_contain": ["os", "datetime"],
    },
    {
        "id": 48, "category": "قواعد البيانات",
        "question": "اكتب كود يضغط مجلداً إلى ملف ZIP",
        "keywords": ["import", "zipfile", "ZipFile", "write"],
        "must_contain": ["zipfile"],
    },
    {
        "id": 49, "category": "قواعد البيانات",
        "question": "اكتب كود يرسل بريداً إلكترونياً باستخدام smtplib",
        "keywords": ["import", "smtplib", "SMTP", "sendmail"],
        "must_contain": ["smtplib", "SMTP"],
    },
    {
        "id": 50, "category": "قواعد البيانات",
        "question": "اكتب unit test لدالة تحسب المساحة",
        "keywords": ["import", "unittest", "TestCase", "assertEqual"],
        "must_contain": ["unittest", "assertEqual"],
    },
]


# ════════════════════════════════════════════════════════════════
# نتيجة سؤال واحد
# ════════════════════════════════════════════════════════════════

@dataclass
class QuestionResult:
    question_id:   int
    category:      str
    question:      str
    answer:        str
    score:         float        # 0.0 إلى 1.0
    passed:        bool
    keywords_found: list[str]   = field(default_factory=list)
    keywords_missed: list[str]  = field(default_factory=list)
    response_time: float        = 0.0


# ════════════════════════════════════════════════════════════════
# نظام التقييم
# ════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    يقيّم النموذج على 50 سؤال برمجي ويحسب نسبة الدقة.

    معيار التقييم لكل سؤال:
      - الكلمات الأساسية (must_contain): 60% من الدرجة
      - الكلمات المساعدة (keywords):    40% من الدرجة
      - الجواب لا يكون فارغاً أو عشوائياً
    """

    PROMPT_TEMPLATE = "### المهمة:\n{question}\n\n### الكود:\n"

    def __init__(self, engine, model_name: str = "النموذج"):
        self.engine     = engine
        self.model_name = model_name
        self.results:   list[QuestionResult] = []

    def evaluate_answer(self, q: dict, answer: str) -> QuestionResult:
        """يقيّم إجابة واحدة ويعطيها درجة."""
        answer_lower = answer.lower()

        # فحص الكلمات الأساسية (must_contain) — وزن 60%
        must     = q.get("must_contain", [])
        found_m  = [kw for kw in must if kw.lower() in answer_lower]
        missed_m = [kw for kw in must if kw.lower() not in answer_lower]
        must_score = len(found_m) / len(must) if must else 1.0

        # فحص الكلمات المساعدة (keywords) — وزن 40%
        kws      = q.get("keywords", [])
        found_k  = [kw for kw in kws if kw.lower() in answer_lower]
        kw_score = len(found_k) / len(kws) if kws else 1.0

        # الدرجة الإجمالية
        score = 0.6 * must_score + 0.4 * kw_score

        # عقوبة إذا كان الجواب قصيراً جداً
        if len(answer.strip()) < 20:
            score *= 0.3

        passed = score >= 0.5

        return QuestionResult(
            question_id    = q["id"],
            category       = q["category"],
            question       = q["question"],
            answer         = answer[:500],
            score          = round(score, 3),
            passed         = passed,
            keywords_found = found_m + found_k,
            keywords_missed= missed_m,
        )

    def run(self, questions: list[dict] = None, verbose: bool = True) -> dict:
        """
        يُشغّل التقييم الكامل على كل الأسئلة.
        يُعيد ملخصاً بالنتائج.
        """
        questions = questions or QUESTIONS
        self.results.clear()
        total = len(questions)

        print(f"\n{'═'*56}")
        print(f"  تقييم: {self.model_name}")
        print(f"  الأسئلة: {total} | المعيار: 50% للنجاح")
        print(f"{'═'*56}\n")

        for i, q in enumerate(questions, 1):
            prompt = self.PROMPT_TEMPLATE.format(question=q["question"])

            t0 = time.time()
            try:
                answer = self.engine.generate(prompt)
            except Exception as e:
                answer = f"[خطأ: {e}]"
            elapsed = time.time() - t0

            result = self.evaluate_answer(q, answer)
            result.response_time = round(elapsed, 2)
            self.results.append(result)

            if verbose:
                icon   = "✓" if result.passed else "✗"
                bar    = "█" * int(result.score * 10) + "░" * (10 - int(result.score * 10))
                print(
                    f"  [{i:02d}/50] {icon} "
                    f"[{bar}] {result.score*100:5.1f}%  "
                    f"({elapsed:.1f}ث)  "
                    f"{q['category']:15s}  "
                    f"{q['question'][:35]}..."
                )

        return self._summarize()

    def _summarize(self) -> dict:
        """يحسب ملخص النتائج الإجمالية."""
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r.passed)
        avg_score = sum(r.score for r in self.results) / total if total else 0
        avg_time  = sum(r.response_time for r in self.results) / total if total else 0

        # نتائج حسب الفئة
        by_category: dict[str, dict] = {}
        for r in self.results:
            cat = r.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "score_sum": 0.0}
            by_category[cat]["total"]     += 1
            by_category[cat]["passed"]    += 1 if r.passed else 0
            by_category[cat]["score_sum"] += r.score

        categories = {
            cat: {
                "pass_rate": round(v["passed"] / v["total"] * 100, 1),
                "avg_score": round(v["score_sum"] / v["total"] * 100, 1),
            }
            for cat, v in by_category.items()
        }

        return {
            "model_name":  self.model_name,
            "total":       total,
            "passed":      passed,
            "failed":      total - passed,
            "pass_rate":   round(passed / total * 100, 1),
            "avg_score":   round(avg_score * 100, 1),
            "avg_time_s":  round(avg_time, 2),
            "by_category": categories,
            "results":     [asdict(r) for r in self.results],
            "timestamp":   datetime.datetime.now().isoformat(),
        }


# ════════════════════════════════════════════════════════════════
# عرض النتائج
# ════════════════════════════════════════════════════════════════

def print_report(summary: dict):
    """يطبع تقريراً منسقاً بالنتائج."""
    print(f"\n{'═'*56}")
    print(f"  تقرير التقييم: {summary['model_name']}")
    print(f"{'═'*56}")
    print(f"  الأسئلة     : {summary['total']}")
    print(f"  ناجح        : {summary['passed']} ✓")
    print(f"  راسب        : {summary['failed']} ✗")
    print(f"  نسبة النجاح : {summary['pass_rate']}%")
    print(f"  متوسط الدقة : {summary['avg_score']}%")
    print(f"  متوسط الوقت : {summary['avg_time_s']}ث / سؤال")

    print(f"\n  {'─'*50}")
    print(f"  النتائج حسب الفئة:")
    print(f"  {'─'*50}")
    print(f"  {'الفئة':20s} {'نسبة النجاح':15s} {'متوسط الدقة'}")
    print(f"  {'─'*50}")
    for cat, data in summary["by_category"].items():
        bar = "█" * int(data["pass_rate"] / 10) + "░" * (10 - int(data["pass_rate"] / 10))
        print(f"  {cat:20s} [{bar}] {data['pass_rate']:5.1f}%   {data['avg_score']:5.1f}%")

    # تقييم عام
    rate = summary["pass_rate"]
    print(f"\n  {'─'*50}")
    if rate >= 85:
        verdict = "ممتاز — النموذج محترف في البرمجة"
    elif rate >= 70:
        verdict = "جيد جداً — النموذج كفء للمهام العملية"
    elif rate >= 55:
        verdict = "جيد — النموذج يحتاج مزيداً من التدريب"
    elif rate >= 40:
        verdict = "مقبول — يُنصح بإعادة التدريب على بيانات أكثر"
    else:
        verdict = "ضعيف — النموذج يحتاج تدريباً أعمق"
    print(f"  التقييم النهائي: {verdict}")
    print(f"{'═'*56}\n")


def print_comparison(before: dict, after: dict):
    """يطبع مقارنة قبل وبعد التدريب."""
    diff_rate  = after["pass_rate"]  - before["pass_rate"]
    diff_score = after["avg_score"]  - before["avg_score"]
    diff_time  = after["avg_time_s"] - before["avg_time_s"]

    print(f"\n{'╔'+'═'*54+'╗'}")
    print(f"║{'مقارنة قبل وبعد التدريب':^54}║")
    print(f"{'╠'+'═'*54+'╣'}")
    print(f"║  {'المقياس':20s} {'قبل':>10s} {'بعد':>10s} {'الفرق':>10s}  ║")
    print(f"{'╠'+'─'*54+'╣'}")
    print(f"║  {'نسبة النجاح':20s} {before['pass_rate']:>9.1f}% {after['pass_rate']:>9.1f}% {diff_rate:>+9.1f}%  ║")
    print(f"║  {'متوسط الدقة':20s} {before['avg_score']:>9.1f}% {after['avg_score']:>9.1f}% {diff_score:>+9.1f}%  ║")
    print(f"║  {'سرعة الإجابة':20s} {before['avg_time_s']:>9.1f}ث {after['avg_time_s']:>9.1f}ث {diff_time:>+9.1f}ث  ║")
    print(f"{'╠'+'═'*54+'╣'}")

    # مقارنة الفئات
    print(f"║  {'الفئة':22s} {'قبل':>8s}  {'بعد':>8s}  {'الفرق':>8s}   ║")
    print(f"║  {'─'*50}   ║")
    all_cats = set(list(before["by_category"].keys()) + list(after["by_category"].keys()))
    for cat in sorted(all_cats):
        b = before["by_category"].get(cat, {}).get("pass_rate", 0)
        a = after["by_category"].get(cat, {}).get("pass_rate", 0)
        d = a - b
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        print(f"║  {cat:22s} {b:>7.1f}%  {a:>7.1f}%  {arrow}{abs(d):>6.1f}%   ║")

    print(f"{'╠'+'═'*54+'╣'}")

    # الخلاصة
    if diff_rate >= 20:
        conclusion = "تحسّن كبير جداً! التدريب ناجح"
    elif diff_rate >= 10:
        conclusion = "تحسّن ملحوظ! التدريب مفيد"
    elif diff_rate >= 0:
        conclusion = "تحسّن بسيط — زِد من البيانات"
    else:
        conclusion = "لم يتحسّن — راجع إعدادات التدريب"

    print(f"║  الخلاصة: {conclusion:44s}║")
    print(f"{'╚'+'═'*54+'╝'}\n")


# ════════════════════════════════════════════════════════════════
# محركات التوليد (مشتركة مع chat.py)
# ════════════════════════════════════════════════════════════════

class GGUFEngine:
    def __init__(self, path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("❌ pip install llama-cpp-python")
            sys.exit(1)
        print(f"  تحميل GGUF: {path}")
        self.llm = Llama(model_path=path, n_ctx=1024, n_threads=os.cpu_count(), verbose=False)
        print("  ✓ جاهز")

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        out = self.llm(prompt, max_tokens=max_tokens, temperature=temperature,
                       stop=["### المهمة:", "###"], echo=False)
        return out["choices"][0]["text"].strip()


class HFEngine:
    def __init__(self, path: str):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("❌ pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  تحميل: {path} على {self.device.upper()}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
        print("  ✓ جاهز")

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.3) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def make_engine(args_gguf, args_model):
    if args_gguf:
        return GGUFEngine(args_gguf), os.path.basename(args_gguf)
    return HFEngine(args_model), os.path.basename(args_model.rstrip("/"))


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — تقييم نموذج البرمجة",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # تقييم نموذج GGUF
  python evaluate.py --gguf ./models/code_expert.gguf

  # تقييم نموذج HuggingFace
  python evaluate.py --model ./models/code_expert

  # مقارنة قبل وبعد التدريب
  python evaluate.py --before microsoft/phi-2 --after ./models/code_expert

  # حفظ النتائج
  python evaluate.py --gguf ./models/code_expert.gguf --save results.json

  # فئة واحدة فقط (للاختبار السريع)
  python evaluate.py --model ./models/code_expert --category "Python أساسي"
        """,
    )
    parser.add_argument("--gguf",     help="تقييم ملف GGUF")
    parser.add_argument("--model",    help="تقييم نموذج HuggingFace")
    parser.add_argument("--before",   help="النموذج قبل التدريب (للمقارنة)")
    parser.add_argument("--after",    help="النموذج بعد التدريب (للمقارنة)")
    parser.add_argument("--save",     help="حفظ النتائج في ملف JSON")
    parser.add_argument("--category", help="تقييم فئة واحدة فقط")
    parser.add_argument("--quiet",    action="store_true", help="إخفاء تفاصيل كل سؤال")
    args = parser.parse_args()

    # تصفية الأسئلة حسب الفئة
    questions = QUESTIONS
    if args.category:
        questions = [q for q in QUESTIONS if args.category in q["category"]]
        if not questions:
            cats = list({q["category"] for q in QUESTIONS})
            print(f"❌ الفئة غير موجودة. الفئات المتاحة:\n  " + "\n  ".join(cats))
            sys.exit(1)
        print(f"  تقييم فئة: {args.category} ({len(questions)} سؤال)")

    all_summaries = {}

    # ── وضع المقارنة قبل/بعد ──────────────────────────────────────
    if args.before and args.after:
        print("\n[وضع المقارنة] قبل وبعد التدريب\n")

        print("[ النموذج قبل التدريب ]")
        engine_b, name_b = make_engine(None, args.before)
        ev_b    = ModelEvaluator(engine_b, name_b)
        before  = ev_b.run(questions, verbose=not args.quiet)
        print_report(before)

        print("\n[ النموذج بعد التدريب ]")
        engine_a, name_a = make_engine(None, args.after)
        ev_a    = ModelEvaluator(engine_a, name_a)
        after   = ev_a.run(questions, verbose=not args.quiet)
        print_report(after)

        print_comparison(before, after)

        all_summaries = {"before": before, "after": after}

    # ── وضع التقييم العادي ────────────────────────────────────────
    elif args.gguf or args.model:
        engine, name = make_engine(args.gguf, args.model)
        evaluator    = ModelEvaluator(engine, name)
        summary      = evaluator.run(questions, verbose=not args.quiet)
        print_report(summary)
        all_summaries = summary

    else:
        print("❌ حدّد نموذجاً للتقييم.")
        print("   python evaluate.py --gguf ./models/code_expert.gguf")
        print("   python evaluate.py --model microsoft/phi-2")
        print("   python evaluate.py --before microsoft/phi-2 --after ./models/code_expert")
        sys.exit(1)

    # حفظ النتائج
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"  ✓ النتائج محفوظة: {args.save}")


if __name__ == "__main__":
    main()

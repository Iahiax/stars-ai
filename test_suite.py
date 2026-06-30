"""
═══════════════════════════════════════════════════════════════
  Stars AI — اختبارات تلقائية شاملة
  يتحقق من صحة كل مكونات المشروع قبل التدريب أو الرفع
═══════════════════════════════════════════════════════════════

الاستخدام:
  python test_suite.py              # كل الاختبارات
  python test_suite.py --fast       # اختبارات سريعة فقط (بدون نماذج)
  python test_suite.py --module imports   # نوع محدد
  python test_suite.py --module model
  python test_suite.py --module data
  python test_suite.py --module eval
  python test_suite.py --module agents
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import argparse
import traceback
import importlib
from pathlib import Path
from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════════
# نتيجة اختبار
# ════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name:    str
    passed:  bool
    message: str = ""
    time_s:  float = 0.0


class TestSuite:
    def __init__(self):
        self.results: list[TestResult] = []
        self.total   = 0
        self.passed  = 0

    def run(self, name: str, fn):
        self.total += 1
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            self.results.append(TestResult(name, True, "✓", round(elapsed, 2)))
            self.passed += 1
            print(f"  ✓  {name:55s} ({elapsed:.2f}ث)")
        except Exception as e:
            elapsed = time.time() - t0
            msg = str(e)[:120]
            self.results.append(TestResult(name, False, msg, round(elapsed, 2)))
            print(f"  ✗  {name:55s} ({elapsed:.2f}ث)")
            print(f"     → {msg}")

    def summary(self) -> bool:
        failed = self.total - self.passed
        print(f"\n{'═'*65}")
        print(f"  النتائج: {self.passed}/{self.total} اختبار ناجح | {failed} فاشل")
        print(f"{'═'*65}")

        if failed:
            print("\n  الاختبارات الفاشلة:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}")
                    print(f"    {r.message}")

        return failed == 0


# ════════════════════════════════════════════════════════════════
# 1. اختبارات المكتبات (imports)
# ════════════════════════════════════════════════════════════════

def test_imports(suite: TestSuite):
    print("\n[1] اختبار المكتبات الأساسية")
    print("─" * 65)

    core_libs = [
        ("torch",           "PyTorch"),
        ("transformers",    "HuggingFace Transformers"),
        ("datasets",        "HuggingFace Datasets"),
        ("peft",            "PEFT / LoRA"),
        ("tqdm",            "TQDM"),
        ("numpy",           "NumPy"),
    ]

    optional_libs = [
        ("accelerate",      "Accelerate (تدريب أسرع)"),
        ("bitsandbytes",    "BitsAndBytes (تكميم 4-bit)"),
        ("llama_cpp",       "llama-cpp-python (GGUF)"),
        ("openai",          "OpenAI SDK"),
        ("aiohttp",         "aiohttp (async HTTP)"),
        ("wandb",           "Weights & Biases"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("sentencepiece",   "SentencePiece"),
        ("dotenv",          "python-dotenv"),
    ]

    for lib, label in core_libs:
        suite.run(f"import {label}", lambda l=lib: __import__(l))

    for lib, label in optional_libs:
        try:
            __import__(lib)
            print(f"  ○  {f'import {label} (اختياري)':55s} موجود")
        except ImportError:
            print(f"  -  {f'import {label} (اختياري)':55s} غير مثبّت")


# ════════════════════════════════════════════════════════════════
# 2. اختبارات حزمة stars_ai
# ════════════════════════════════════════════════════════════════

def test_stars_ai_package(suite: TestSuite):
    print("\n[2] اختبار حزمة stars_ai")
    print("─" * 65)

    suite.run("استيراد stars_ai", lambda: __import__("stars_ai"))
    suite.run("استيراد KeyManager", lambda: __import__(
        "stars_ai.key_manager", fromlist=["KeyManager"]))
    suite.run("استيراد ModelRegistry", lambda: __import__(
        "stars_ai.model_registry", fromlist=["ModelRegistry"]))
    suite.run("استيراد StarsLM", lambda: __import__(
        "stars_ai.model_builder", fromlist=["StarsLM"]))
    suite.run("استيراد GGUFConverter", lambda: __import__(
        "stars_ai.gguf_converter", fromlist=["StarsLMToGGUF"]))

    def test_key_manager():
        from stars_ai.key_manager import KeyManager
        km = KeyManager()
        result = km.get_key_safe("non_existent_provider_xyz")
        assert result is None

    def test_registry():
        from stars_ai.model_registry import ModelRegistry
        reg     = ModelRegistry()
        summary = reg.summary()
        assert summary["total_providers"] > 0
        assert summary["total_models"]    > 0
        assert summary["gguf_compatible"] > 0

    def test_registry_search():
        from stars_ai.model_registry import ModelRegistry
        reg     = ModelRegistry()
        results = reg.search("llama")
        assert len(results) > 0

    suite.run("KeyManager — get_key_safe", test_key_manager)
    suite.run("ModelRegistry — summary()", test_registry)
    suite.run("ModelRegistry — search(llama)", test_registry_search)


# ════════════════════════════════════════════════════════════════
# 3. اختبارات النموذج StarsLM
# ════════════════════════════════════════════════════════════════

def test_model(suite: TestSuite):
    print("\n[3] اختبار نموذج StarsLM")
    print("─" * 65)

    def test_starslm_small():
        import torch
        from stars_ai.model_builder import StarsLM, StarsConfig

        cfg = StarsConfig(
            vocab_size=1000, hidden_size=64,
            num_layers=2, num_heads=4, max_seq_len=32,
        )
        model = StarsLM(cfg)
        params = model.count_parameters()
        assert params > 0, f"عدد المعاملات صفر!"

        x   = torch.randint(0, 1000, (1, 16))
        out = model(x)
        assert out.shape == (1, 16, 1000), f"شكل الخرج خاطئ: {out.shape}"

    def test_generate():
        import torch
        from stars_ai.model_builder import StarsLM, StarsConfig

        cfg = StarsConfig(
            vocab_size=1000, hidden_size=64,
            num_layers=2, num_heads=4, max_seq_len=32,
        )
        model = StarsLM(cfg)
        model.eval()
        x   = torch.randint(0, 1000, (1, 5))
        out = model.generate(x, max_new_tokens=10)
        assert out.shape[1] > 5, "التوليد لم يُنتج tokens جديدة"

    def test_save_load(tmp_path="/tmp/test_starslm"):
        import torch
        from stars_ai.model_builder import StarsLM, StarsConfig

        cfg = StarsConfig(
            vocab_size=1000, hidden_size=64,
            num_layers=2, num_heads=4, max_seq_len=32,
        )
        model = StarsLM(cfg)
        os.makedirs(tmp_path, exist_ok=True)
        model.save_pretrained(tmp_path)

        assert os.path.exists(os.path.join(tmp_path, "config.json"))
        assert os.path.exists(os.path.join(tmp_path, "model.pt"))

        loaded = StarsLM.from_pretrained(tmp_path)
        assert loaded.count_parameters() == model.count_parameters()

    suite.run("StarsLM — بناء نموذج صغير + forward pass", test_starslm_small)
    suite.run("StarsLM — generate()", test_generate)
    suite.run("StarsLM — save + load", test_save_load)


# ════════════════════════════════════════════════════════════════
# 4. اختبارات معالجة البيانات
# ════════════════════════════════════════════════════════════════

def test_data(suite: TestSuite):
    print("\n[4] اختبار معالجة البيانات")
    print("─" * 65)

    def test_char_tokenizer():
        from train_custom import CharTokenizer
        tok = CharTokenizer()
        tok.train(["مرحباً بالعالم", "Hello World"])
        ids  = tok.encode("مرحبا")
        text = tok.decode(ids)
        assert len(ids) == len("مرحبا")
        assert isinstance(text, str)

    def test_tokenizer_save_load(tmp_path="/tmp/test_tok.json"):
        from train_custom import CharTokenizer
        tok = CharTokenizer()
        tok.train(["مرحباً"])
        tok.save(tmp_path)

        loaded = CharTokenizer.load(tmp_path)
        assert loaded.vocab_size == tok.vocab_size

    def test_dataset_format():
        import torch
        from train_custom import CharTokenizer, CustomDataset
        tok = CharTokenizer()
        texts = ["مرحباً بالعالم"] * 10
        tok.train(texts)
        ds = CustomDataset(texts, tok, seq_len=16)
        assert len(ds) > 0
        item = ds[0]
        assert "input_ids" in item
        assert "labels"    in item

    def test_jsonl_load(tmp_path="/tmp/test_data.jsonl"):
        data = [
            {"instruction": "اكتب دالة جمع", "output": "def add(a,b): return a+b"},
            {"instruction": "اكتب حلقة",     "output": "for i in range(10): print(i)"},
        ]
        with open(tmp_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        loaded = []
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                loaded.append(json.loads(line.strip()))
        assert len(loaded) == 2

    suite.run("CharTokenizer — train + encode + decode", test_char_tokenizer)
    suite.run("CharTokenizer — save + load",             test_tokenizer_save_load)
    suite.run("CustomDataset — format صحيح",             test_dataset_format)
    suite.run("JSONL — قراءة وكتابة",                    test_jsonl_load)


# ════════════════════════════════════════════════════════════════
# 5. اختبارات نظام التقييم
# ════════════════════════════════════════════════════════════════

def test_evaluation(suite: TestSuite):
    print("\n[5] اختبار نظام التقييم")
    print("─" * 65)

    def test_evaluator_scoring():
        from evaluate import ModelEvaluator, QUESTIONS

        class MockEngine:
            def generate(self, prompt: str) -> str:
                return "def add(a, b):\n    return a + b"

        ev     = ModelEvaluator(MockEngine(), "test-model")
        q      = QUESTIONS[0]
        result = ev.evaluate_answer(q, "def sum_list(lst):\n    return sum(lst)")
        assert 0.0 <= result.score <= 1.0
        assert result.question_id == q["id"]

    def test_questions_count():
        from evaluate import QUESTIONS
        assert len(QUESTIONS) == 50, f"عدد الأسئلة {len(QUESTIONS)} وليس 50"

    def test_questions_format():
        from evaluate import QUESTIONS
        for q in QUESTIONS:
            assert "id"           in q
            assert "category"     in q
            assert "question"     in q
            assert "keywords"     in q
            assert "must_contain" in q

    def test_all_categories():
        from evaluate import QUESTIONS
        categories = set(q["category"] for q in QUESTIONS)
        assert len(categories) >= 4, f"الفئات أقل من 4: {categories}"

    suite.run("Evaluator — تقييم إجابة واحدة",  test_evaluator_scoring)
    suite.run("Evaluator — 50 سؤال بالضبط",    test_questions_count)
    suite.run("Evaluator — صيغة الأسئلة صحيحة", test_questions_format)
    suite.run("Evaluator — وجود كل الفئات",     test_all_categories)


# ════════════════════════════════════════════════════════════════
# 6. اختبارات الملفات والمسارات
# ════════════════════════════════════════════════════════════════

def test_files(suite: TestSuite):
    print("\n[6] اختبار وجود الملفات الأساسية")
    print("─" * 65)

    required_files = [
        "main.py",
        "train_all.py",
        "train_custom.py",
        "finetune_lora.py",
        "finetune_arabic.py",
        "evaluate.py",
        "benchmark.py",
        "auto_improve.py",
        "generate_data.py",
        "compare_models.py",
        "rag_code.py",
        "prune_model.py",
        "distill_model.py",
        "sync_to_hf.py",
        "chat.py",
        "requirements.txt",
        ".env.example",
        "stars_ai/__init__.py",
        "stars_ai/key_manager.py",
        "stars_ai/model_registry.py",
        "stars_ai/model_builder.py",
        "stars_ai/gguf_converter.py",
        "stars_ai/agents/__init__.py",
        "stars_ai/agents/manager_agent.py",
        "stars_ai/agents/expert_agents.py",
    ]

    for f in required_files:
        suite.run(f"وجود: {f}", lambda p=f: (_ for _ in ()).throw(
            FileNotFoundError(f"مفقود: {p}")) if not os.path.exists(p) else None)


# ════════════════════════════════════════════════════════════════
# 7. اختبارات متغيرات البيئة
# ════════════════════════════════════════════════════════════════

def test_env(suite: TestSuite):
    print("\n[7] فحص متغيرات البيئة (اختياري)")
    print("─" * 65)

    env_vars = [
        ("OPENAI_API_KEY",   "OpenAI API Key"),
        ("HF_TOKEN",         "HuggingFace Token"),
        ("WANDB_API_KEY",    "Weights & Biases"),
        ("GCP_PROJECT_ID",   "Google Cloud Project"),
    ]

    for var, label in env_vars:
        val = os.getenv(var)
        if val:
            print(f"  ✓  {label:40s} موجود ({var[:4]}...)")
        else:
            print(f"  -  {label:40s} غير موجود ({var})")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — اختبارات تلقائية شاملة"
    )
    parser.add_argument("--fast",   action="store_true",
                        help="اختبارات سريعة فقط (بدون تحميل نماذج ثقيلة)")
    parser.add_argument("--module", choices=["imports","package","model","data","eval","files","env"],
                        help="تشغيل مجموعة اختبارات محددة فقط")
    args = parser.parse_args()

    print("═" * 65)
    print("  Stars AI — Test Suite")
    print("═" * 65)

    suite = TestSuite()

    if args.module == "imports" or not args.module:
        test_imports(suite)

    if args.module == "package" or not args.module:
        test_stars_ai_package(suite)

    if (args.module == "model" or not args.module) and not args.fast:
        test_model(suite)

    if args.module == "data" or not args.module:
        test_data(suite)

    if args.module == "eval" or not args.module:
        test_evaluation(suite)

    if args.module == "files" or not args.module:
        test_files(suite)

    if args.module == "env" or not args.module:
        test_env(suite)

    success = suite.summary()

    if success:
        print("\n  ✅ جميع الاختبارات ناجحة — المشروع جاهز!")
    else:
        print("\n  ❌ بعض الاختبارات فشلت — راجع الأخطاء أعلاه")
        sys.exit(1)


if __name__ == "__main__":
    main()

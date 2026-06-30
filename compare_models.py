"""
═══════════════════════════════════════════════════════════════
  Stars AI — مقارنة نماذج متعددة في نفس الوقت
  اسأل سؤالاً واحداً ويجيب عليه عدة نماذج جنباً إلى جنب
═══════════════════════════════════════════════════════════════

التشغيل:
  python compare_models.py --models phi-2 mistral llama3
  python compare_models.py --gguf model1.gguf model2.gguf
  python compare_models.py --models phi-2 mistral --question "اكتب دالة فرز"
  python compare_models.py --models phi-2 mistral --interactive
"""

import os
import sys
import json
import time
import argparse
import threading
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from stars_ai.model_registry import ModelRegistry


# ── خرائط الأسماء المختصرة ───────────────────────────────────────────────────

MODEL_ALIASES = {
    "phi-2":   "microsoft/phi-2",
    "phi-3":   "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama3":  "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama2":  "meta-llama/Llama-2-7b-chat-hf",
    "deepseek":"deepseek-ai/deepseek-coder-6.7b-instruct",
    "gemma":   "google/gemma-2b-it",
    "qwen":    "Qwen/Qwen2-1.5B-Instruct",
    "dolphin": "cognitivecomputations/dolphin-2.9-llama3-8b",
}

PROMPT_TEMPLATE = "### المهمة:\n{question}\n\n### الكود:\n"


# ── نتيجة نموذج واحد ────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    model_name:  str
    answer:      str
    time_taken:  float
    error:       str = ""

    @property
    def word_count(self) -> int:
        return len(self.answer.split())

    @property
    def has_code(self) -> bool:
        code_keywords = ["def ", "class ", "import ", "for ", "while ", "if "]
        return any(kw in self.answer for kw in code_keywords)


# ── محرك نموذج واحد ─────────────────────────────────────────────────────────

class ModelRunner:
    """يُشغّل نموذجاً واحداً ويجلب إجابته."""

    def __init__(self, model_path: str, is_gguf: bool = False):
        self.model_path = model_path
        self.is_gguf    = is_gguf
        self.name       = os.path.basename(model_path.rstrip("/\\"))
        self._engine    = None

    def load(self):
        """يحمّل النموذج (يُستدعى مرة واحدة)."""
        if self.is_gguf:
            self._load_gguf()
        else:
            self._load_hf()

    def _load_gguf(self):
        try:
            from llama_cpp import Llama
            self._engine = Llama(
                model_path=self.model_path,
                n_ctx=1024,
                n_threads=max(1, os.cpu_count() // 2),
                verbose=False,
            )
        except ImportError:
            raise ImportError("pip install llama-cpp-python")

    def _load_hf(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = MODEL_ALIASES.get(self.model_path, self.model_path)
        self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        if self.is_gguf:
            out = self._engine(
                prompt, max_tokens=max_tokens, temperature=0.3,
                stop=["### المهمة:", "###"], echo=False,
            )
            return out["choices"][0]["text"].strip()
        else:
            import torch
            inputs = self._tok(prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.3, top_p=0.9, do_sample=True,
                    pad_token_id=self._tok.eos_token_id,
                )
            full   = self._tok.decode(out[0], skip_special_tokens=True)
            answer = full[len(self._tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            return answer.strip()


# ── نظام المقارنة ─────────────────────────────────────────────────────────────

class ModelComparator:
    """يُشغّل عدة نماذج بالتوازي ويقارن إجاباتها."""

    def __init__(self, runners: list[ModelRunner]):
        self.runners = runners

    def load_all(self):
        """يحمّل جميع النماذج (بالتوازي)."""
        print("\n[تحميل النماذج...]")
        threads = []
        errors  = []

        def load_one(r):
            try:
                print(f"  ← {r.name}...")
                r.load()
                print(f"  ✓ {r.name}")
            except Exception as e:
                errors.append((r.name, str(e)))
                print(f"  ✗ {r.name}: {e}")

        for r in self.runners:
            t = threading.Thread(target=load_one, args=(r,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            print(f"\n  ⚠ فشل تحميل: {[e[0] for e in errors]}")
        print(f"  تم تحميل {len(self.runners) - len(errors)} نموذج\n")

    def ask(self, question: str, max_tokens: int = 300) -> list[ModelResponse]:
        """يسأل جميع النماذج سؤالاً ويعيد إجاباتهم."""
        prompt    = PROMPT_TEMPLATE.format(question=question)
        responses = [None] * len(self.runners)
        threads   = []

        def run_one(i, runner):
            t0 = time.time()
            try:
                answer = runner.generate(prompt, max_tokens)
            except Exception as e:
                answer = ""
                responses[i] = ModelResponse(runner.name, "", time.time()-t0, str(e))
                return
            responses[i] = ModelResponse(runner.name, answer, round(time.time()-t0, 2))

        for i, r in enumerate(self.runners):
            t = threading.Thread(target=run_one, args=(i, r))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return [r for r in responses if r is not None]

    def print_comparison(self, question: str, responses: list[ModelResponse]):
        """يطبع المقارنة بتنسيق واضح."""
        print(f"\n{'═'*60}")
        print(f"  السؤال: {question}")
        print(f"{'═'*60}")

        for resp in responses:
            icon   = "✓" if resp.has_code else "—"
            status = "يحتوي كود" if resp.has_code else "بدون كود"

            print(f"\n  ┌─ {resp.model_name} ({resp.time_taken}ث | {resp.word_count} كلمة | {status}) {icon}")
            print(f"  │")

            if resp.error:
                print(f"  │  ❌ خطأ: {resp.error}")
            else:
                for line in resp.answer[:600].split("\n"):
                    print(f"  │  {line}")
                if len(resp.answer) > 600:
                    print(f"  │  ... (تم الاقتصار)")
            print(f"  └{'─'*56}")

        # ملخص المقارنة
        print(f"\n  {'─'*60}")
        print(f"  {'النموذج':25s} {'وقت الإجابة':12s} {'الكلمات':10s} {'كود؟'}")
        print(f"  {'─'*60}")
        for r in sorted(responses, key=lambda x: x.time_taken):
            code_tag = "✓" if r.has_code else "✗"
            print(f"  {r.model_name:25s} {r.time_taken:>8.1f}ث   {r.word_count:>6}    {code_tag}")

        fastest = min(responses, key=lambda x: x.time_taken)
        longest = max(responses, key=lambda x: x.word_count)
        print(f"\n  الأسرع : {fastest.model_name} ({fastest.time_taken}ث)")
        print(f"  الأكثر تفصيلاً: {longest.model_name} ({longest.word_count} كلمة)")
        print(f"{'═'*60}\n")

    def save_comparison(self, question: str, responses: list[ModelResponse], path: str):
        """يحفظ نتائج المقارنة في JSON."""
        data = {
            "question":  question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "responses": [
                {
                    "model":     r.model_name,
                    "answer":    r.answer,
                    "time_s":    r.time_taken,
                    "has_code":  r.has_code,
                    "word_count":r.word_count,
                    "error":     r.error,
                }
                for r in responses
            ],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ── وضع التفاعلي ─────────────────────────────────────────────────────────────

WELCOME = """
╔══════════════════════════════════════════════════════════╗
║     Stars AI — مقارنة النماذج التفاعلية                ║
║  اكتب سؤالك وسيجيب جميع النماذج في نفس الوقت          ║
║  أوامر: حفظ | مقارنة | خروج                           ║
╚══════════════════════════════════════════════════════════╝
"""

def run_interactive(comparator: ModelComparator, save_file: str = None):
    print(WELCOME)
    history = []

    while True:
        try:
            question = input("سؤالك: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  إلى اللقاء!")
            break

        if not question:
            continue

        cmd = question.lower()
        if cmd in ("خروج", "exit", "quit"):
            break
        elif cmd in ("حفظ", "save") and save_file and history:
            print(f"  تم الحفظ في: {save_file}")
            continue

        print(f"  ⏳ جميع النماذج تعمل...\n")
        responses = comparator.ask(question)
        comparator.print_comparison(question, responses)
        history.append((question, responses))

        if save_file:
            comparator.save_comparison(question, responses, save_file)


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stars AI — مقارنة نماذج متعددة")
    parser.add_argument("--models",      nargs="+", help="أسماء نماذج HuggingFace أو مختصراتها")
    parser.add_argument("--gguf",        nargs="+", help="مسارات ملفات GGUF")
    parser.add_argument("--question",    help="سؤال واحد (بدل التفاعلي)")
    parser.add_argument("--interactive", action="store_true", help="وضع المحادثة التفاعلية")
    parser.add_argument("--save",        help="حفظ النتائج في ملف JSONL")
    parser.add_argument("--max-tokens",  type=int, default=300)
    args = parser.parse_args()

    # بناء قائمة النماذج
    runners = []
    for m in (args.models or []):
        path = MODEL_ALIASES.get(m, m)
        runners.append(ModelRunner(path, is_gguf=False))
    for g in (args.gguf or []):
        runners.append(ModelRunner(g, is_gguf=True))

    if not runners:
        print("❌ حدّد نموذجاً واحداً على الأقل:")
        print("   python compare_models.py --models phi-2 mistral")
        print("   python compare_models.py --gguf m1.gguf m2.gguf")
        print(f"\n   الأسماء المختصرة المتاحة: {list(MODEL_ALIASES.keys())}")
        sys.exit(1)

    comparator = ModelComparator(runners)
    comparator.load_all()

    if args.question:
        resp = comparator.ask(args.question, args.max_tokens)
        comparator.print_comparison(args.question, resp)
        if args.save:
            comparator.save_comparison(args.question, resp, args.save)
    else:
        run_interactive(comparator, args.save)


if __name__ == "__main__":
    main()

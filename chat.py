"""
═══════════════════════════════════════════════════════════════
  Stars AI — محادثة تفاعلية مع نموذج البرمجة
  اكتب سؤالك ← اضغط Enter ← النموذج يجاوبك بالكود
═══════════════════════════════════════════════════════════════

طريقة التشغيل:

  python chat.py --gguf ./models/stars_expert.gguf
  python chat.py --model ./models/stars_expert_merged
  python chat.py --model microsoft/phi-2

أوامر داخل المحادثة:
  مسح / clear  ← مسح تاريخ المحادثة
  حفظ / save   ← حفظ المحادثة في ملف JSON
  خروج / quit  ← إنهاء البرنامج
  مساعدة       ← عرض الأوامر

بعد المحادثة، يمكنك:
  python evaluate.py    --model ./models/stars_expert_merged
  python benchmark.py   --my-model ./models/stars_expert_merged
  python auto_improve.py --model ./models/stars_expert_merged
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# محرك GGUF
# ════════════════════════════════════════════════════════════════

class GGUFChatEngine:
    """محرك محادثة يستخدم ملف GGUF — يعمل على CPU بدون GPU."""

    def __init__(self, gguf_path: str, n_ctx: int = 2048):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("❌ pip install llama-cpp-python")
            sys.exit(1)

        print(f"  تحميل GGUF: {gguf_path}")
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=os.cpu_count(),
            verbose=False,
        )
        print(f"  ✓ جاهز ({n_ctx} token سياق)")

    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.3) -> str:
        output = self.llm(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=0.9, stop=["### المهمة:", "###"], echo=False,
        )
        return output["choices"][0]["text"].strip()


# ════════════════════════════════════════════════════════════════
# محرك HuggingFace
# ════════════════════════════════════════════════════════════════

class HFChatEngine:
    """محرك محادثة يستخدم نموذج HuggingFace."""

    def __init__(self, model_path: str):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("❌ pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"  تحميل: {model_path}")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()
        print(f"  ✓ جاهز على {self.device.upper()}")

    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.3) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1,
            )
        full   = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full[len(self.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)):]
        return answer.strip()


# ════════════════════════════════════════════════════════════════
# واجهة المحادثة
# ════════════════════════════════════════════════════════════════

PROMPT_TEMPLATE = "### المهمة:\n{question}\n\n### الكود:\n"

WELCOME = """
╔══════════════════════════════════════════════════════════════╗
║          Stars AI — مساعد البرمجة الذكي                    ║
║  اسألني أي سؤال برمجي وسأكتب لك الكود مباشرة              ║
╠══════════════════════════════════════════════════════════════╣
║  أوامر: مسح | حفظ | خروج | مساعدة                         ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLES = [
    "اكتب دالة Python تحسب مجموع قائمة أرقام",
    "اشرح الـ list comprehension مع مثال",
    "اكتب class للتعامل مع ملفات JSON",
    "ما الفرق بين == و is في Python؟",
    "اكتب decorator يقيس وقت تنفيذ الدالة",
    "اكتب خوارزمية Binary Search",
    "اكتب async function تجلب بيانات من API",
    "اكتب generator يولّد أرقام فيبوناتشي",
]

HELP_TEXT = """
  الأوامر المتاحة:
  ─────────────────────────────────────
  مسح     ← مسح تاريخ المحادثة
  حفظ     ← حفظ المحادثة في ملف JSON
  خروج    ← إنهاء البرنامج
  1-8     ← اختيار مثال جاهز بالرقم
  ─────────────────────────────────────
  بعد انتهاء جلستك يمكنك:
  python evaluate.py --model <النموذج>
  python benchmark.py --my-model <النموذج>
  python auto_improve.py --model <النموذج>
"""


class ChatSession:
    def __init__(self, engine):
        self.engine        = engine
        self.history       = []
        self.session_start = datetime.datetime.now()

    def ask(self, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(question=question)
        answer = self.engine.generate(prompt)
        self.history.append({
            "question": question,
            "answer":   answer,
            "time":     datetime.datetime.now().isoformat(),
        })
        return answer

    def save(self, path: str = None):
        if path is None:
            ts   = self.session_start.strftime("%Y%m%d_%H%M%S")
            path = f"./chat_history_{ts}.json"
        data = {
            "session_start":    self.session_start.isoformat(),
            "total_questions":  len(self.history),
            "history":          self.history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ المحادثة محفوظة: {path}")
        return path

    def clear(self):
        self.history.clear()
        print("  ✓ تم مسح تاريخ المحادثة")


def print_answer(answer: str):
    print("\n" + "─" * 56)
    print("  الجواب:")
    print("─" * 56)
    for line in answer.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ",
                                 "return ", "if ", "for ", "while ", "#")):
            print(f"  \033[92m{line}\033[0m")
        elif stripped.startswith(('"""', "'''")):
            print(f"  \033[93m{line}\033[0m")
        else:
            print(f"  {line}")
    print("─" * 56 + "\n")


def run_chat(engine):
    print(WELCOME)
    print("  أمثلة جاهزة:")
    for i, ex in enumerate(EXAMPLES, 1):
        print(f"  {i}. {ex}")
    print()

    session        = ChatSession(engine)
    question_count = 0

    while True:
        try:
            user_input = input("أنت: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  إلى اللقاء!")
            session.save()
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        if cmd in ("خروج", "exit", "quit", "q"):
            session.save()
            print("  إلى اللقاء!")
            break
        elif cmd in ("مسح", "clear", "cls"):
            session.clear()
            os.system("cls" if os.name == "nt" else "clear")
            print(WELCOME)
            continue
        elif cmd in ("حفظ", "save"):
            session.save()
            continue
        elif cmd in ("مساعدة", "help", "?"):
            print(HELP_TEXT)
            continue
        elif cmd.isdigit() and 1 <= int(cmd) <= len(EXAMPLES):
            user_input = EXAMPLES[int(cmd) - 1]
            print(f"  السؤال المختار: {user_input}")

        question_count += 1
        print("  ⏳ جاري الكتابة...", end="", flush=True)

        try:
            answer = session.ask(user_input)
            print("\r" + " " * 30 + "\r", end="")
            print_answer(answer)
        except Exception as e:
            print(f"\n  ❌ خطأ: {e}\n")

        if question_count % 10 == 0:
            session.save()
            print(f"  (حفظ تلقائي — {question_count} سؤال)")


# ════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — مساعد البرمجة التفاعلي",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python chat.py --gguf ./models/stars_expert.gguf
  python chat.py --model ./models/stars_expert_merged
  python chat.py --model microsoft/phi-2
  python chat.py --gguf model.gguf --temp 0.5 --max-tokens 600

بعد الانتهاء:
  python evaluate.py    --model ./models/stars_expert_merged
  python benchmark.py   --my-model ./models/stars_expert_merged
  python auto_improve.py --model ./models/stars_expert_merged --rounds 3
        """,
    )
    parser.add_argument("--gguf",       help="مسار ملف GGUF (أسرع)")
    parser.add_argument("--model",      help="مسار نموذج HuggingFace أو اسمه")
    parser.add_argument("--temp",       type=float, default=0.3,
                        help="درجة الإبداع: 0.1=دقيق, 0.9=مبدع (افتراضي: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--ctx",        type=int, default=2048)

    args = parser.parse_args()

    if not args.gguf and not args.model:
        for path, is_gguf in [
            ("./models/stars_expert.gguf", True),
            ("./models/code_expert.gguf",  True),
            ("./models/stars_expert/stage3_arabic_merged", False),
            ("./models/code_expert",       False),
        ]:
            if os.path.exists(path):
                print(f"  تم اكتشاف: {path}")
                if is_gguf:
                    args.gguf  = path
                else:
                    args.model = path
                break
        else:
            print("❌ لم يُحدَّد نموذج.")
            print("  python chat.py --gguf ./models/stars_expert.gguf")
            print("  python chat.py --model microsoft/phi-2")
            sys.exit(1)

    print("\n[تحميل النموذج...]")
    engine = GGUFChatEngine(args.gguf, n_ctx=args.ctx) if args.gguf else HFChatEngine(args.model)

    orig = engine.generate
    engine.generate = lambda p: orig(p, max_tokens=args.max_tokens, temperature=args.temp)

    run_chat(engine)


if __name__ == "__main__":
    main()

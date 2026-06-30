"""
═══════════════════════════════════════════════════════════════
  Stars AI — نظام RAG للكود
  يقرأ مشروعك البرمجي ثم تسأله عنه وهو يجيب بناءً على كودك
═══════════════════════════════════════════════════════════════

RAG = Retrieval Augmented Generation
المبدأ: بدلاً من تدريب النموذج على كودك، نعطيه الأجزاء ذات الصلة
        من كودك مع كل سؤال فيجيب بناءً عليها.

التشغيل:
  python rag_code.py --project ./my_project --model microsoft/phi-2
  python rag_code.py --project ./my_project --gguf ./models/code_expert.gguf
  python rag_code.py --project . --model microsoft/phi-2 --question "ما الذي تفعله دالة X؟"
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))


# ── أنواع الملفات المدعومة ───────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".r",
    ".sql", ".sh", ".yaml", ".yml", ".json", ".toml", ".md",
}

IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
}


# ── قطعة كود (Chunk) ────────────────────────────────────────────────────────

@dataclass
class CodeChunk:
    """قطعة كود من ملف معين."""
    file_path:  str
    content:    str
    start_line: int
    end_line:   int
    language:   str = ""

    @property
    def header(self) -> str:
        return f"# ملف: {self.file_path} (سطر {self.start_line}-{self.end_line})"

    def to_context(self) -> str:
        return f"{self.header}\n```{self.language}\n{self.content}\n```"


# ── قارئ وفهرس الكود ─────────────────────────────────────────────────────────

class CodeIndexer:
    """
    يقرأ جميع ملفات المشروع ويبنيها في فهرس للبحث.
    يستخدم TF-IDF بسيطاً للبحث (بدون embeddings — سريع ولا يحتاج GPU).
    """

    EXT_LANG = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
        ".rs": "rust", ".rb": "ruby", ".php": "php", ".sql": "sql",
        ".sh": "bash", ".md": "markdown",
    }

    def __init__(self, project_dir: str, chunk_size: int = 50):
        self.project_dir = project_dir
        self.chunk_size  = chunk_size
        self.chunks:     list[CodeChunk] = []
        self._tfidf:     list[dict]      = []
        self._vocab:     dict[str, int]  = {}

    def index(self) -> int:
        """يقرأ جميع الملفات ويبني الفهرس."""
        print(f"\n[RAG] فهرسة المشروع: {self.project_dir}")
        files_indexed = 0

        for root, dirs, files in os.walk(self.project_dir):
            # تجاهل المجلدات غير المهمة
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                fpath = os.path.join(root, fname)
                rel   = os.path.relpath(fpath, self.project_dir)

                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except Exception:
                    continue

                lang = self.EXT_LANG.get(ext, "")
                # تقسيم الملف إلى قطع
                for start in range(0, len(lines), self.chunk_size):
                    end     = min(start + self.chunk_size, len(lines))
                    content = "".join(lines[start:end]).strip()
                    if len(content) < 30:
                        continue
                    self.chunks.append(CodeChunk(rel, content, start+1, end, lang))

                files_indexed += 1

        print(f"  ✓ ملفات: {files_indexed} | قطع: {len(self.chunks)}")
        self._build_tfidf()
        return len(self.chunks)

    def _tokenize(self, text: str) -> list[str]:
        """تقطيع النص إلى كلمات."""
        import re
        tokens = re.findall(r"[a-zA-Z_\u0600-\u06FF][a-zA-Z0-9_\u0600-\u06FF]*", text.lower())
        return tokens

    def _build_tfidf(self):
        """بناء مصفوفة TF-IDF بسيطة."""
        # حساب Document Frequency
        df: dict[str, int] = {}
        doc_tokens = []
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk.content + " " + chunk.file_path))
            doc_tokens.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        N = len(self.chunks)
        # حساب TF-IDF لكل قطعة
        self._tfidf = []
        for tokens in doc_tokens:
            scores = {}
            for t in tokens:
                tf  = 1
                idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
                scores[t] = tf * idf
            self._tfidf.append(scores)

    def search(self, query: str, top_k: int = 5) -> list[CodeChunk]:
        """يبحث عن أقرب القطع للسؤال."""
        if not self.chunks:
            return []

        q_tokens = self._tokenize(query)
        scores   = []

        for i, tfidf in enumerate(self._tfidf):
            score = sum(tfidf.get(t, 0) for t in q_tokens)
            if score > 0:
                scores.append((score, i))

        scores.sort(reverse=True)
        top = [self.chunks[i] for _, i in scores[:top_k]]
        return top

    def get_stats(self) -> dict:
        ext_count: dict[str, int] = {}
        for c in self.chunks:
            lang = c.language or "other"
            ext_count[lang] = ext_count.get(lang, 0) + 1
        return {
            "total_chunks": len(self.chunks),
            "by_language":  ext_count,
        }


# ── محرك RAG ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    يجمع بين الفهرس والنموذج للإجابة على أسئلة عن الكود.
    """

    SYSTEM_PROMPT = """أنت مساعد برمجي متخصص في تحليل الكود.
لديك أجزاء من مشروع برمجي. أجب على الأسئلة بناءً على هذا الكود فقط.
إذا لم تجد المعلومة في الكود المعطى، قل ذلك بوضوح."""

    def __init__(self, indexer: CodeIndexer, model_path: str, is_gguf: bool = False):
        self.indexer  = indexer
        self.is_gguf  = is_gguf
        self.model_path = model_path
        self._engine = None

    def load(self):
        if self.is_gguf:
            try:
                from llama_cpp import Llama
                self._engine = Llama(
                    model_path=self.model_path, n_ctx=4096,
                    n_threads=os.cpu_count(), verbose=False,
                )
            except ImportError:
                print("❌ pip install llama-cpp-python")
                sys.exit(1)
        else:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            path = self.model_path
            self._tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                path, device_map="auto", trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def ask(self, question: str, top_k: int = 4, max_tokens: int = 500) -> dict:
        """يجيب على سؤال باستخدام الكود ذي الصلة."""
        # 1. البحث عن القطع ذات الصلة
        relevant = self.indexer.search(question, top_k)

        if not relevant:
            return {"answer": "لم أجد كوداً ذا صلة في المشروع.", "sources": []}

        # 2. بناء السياق
        context = "\n\n".join(c.to_context() for c in relevant)
        sources = [{"file": c.file_path, "lines": f"{c.start_line}-{c.end_line}"}
                   for c in relevant]

        # 3. بناء الـ prompt
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"=== الكود ذو الصلة ===\n{context}\n\n"
            f"=== السؤال ===\n{question}\n\n"
            f"=== الإجابة ===\n"
        )

        # 4. توليد الإجابة
        answer = self._generate(prompt, max_tokens)

        return {"answer": answer, "sources": sources, "context_chunks": len(relevant)}

    def _generate(self, prompt: str, max_tokens: int) -> str:
        if self.is_gguf:
            out = self._engine(
                prompt, max_tokens=max_tokens, temperature=0.2,
                stop=["=== السؤال ===", "==="], echo=False,
            )
            return out["choices"][0]["text"].strip()
        else:
            import torch
            inputs = self._tok(prompt, return_tensors="pt",
                               truncation=True, max_length=3500).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.2, top_p=0.9, do_sample=True,
                    pad_token_id=self._tok.eos_token_id,
                )
            full = self._tok.decode(out[0], skip_special_tokens=True)
            return full[len(self._tok.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


# ── واجهة المحادثة ────────────────────────────────────────────────────────────

WELCOME_RAG = """
╔══════════════════════════════════════════════════════════╗
║         Stars AI — مساعد كودك الشخصي (RAG)            ║
║  اسألني أي شيء عن مشروعك وسأجيب بناءً على كودك        ║
╚══════════════════════════════════════════════════════════╝
"""

def run_rag_chat(rag: RAGEngine):
    stats = rag.indexer.get_stats()
    print(WELCOME_RAG)
    print(f"  الفهرس: {stats['total_chunks']} قطعة | {stats['by_language']}")
    print()

    while True:
        try:
            question = input("سؤالك عن الكود: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  إلى اللقاء!")
            break

        if not question:
            continue
        if question.lower() in ("خروج", "exit", "quit"):
            break

        print("  ⏳ البحث في الكود والإجابة...\n")
        result = rag.ask(question)

        print("  ─── الإجابة ───────────────────────────────")
        print(f"  {result['answer']}")
        print(f"\n  ─── المصادر ({result['context_chunks']} قطعة) ─────────────")
        for s in result["sources"]:
            print(f"  📄 {s['file']} (سطر {s['lines']})")
        print()


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stars AI — RAG للكود")
    parser.add_argument("--project",  required=True, help="مسار مجلد مشروعك")
    parser.add_argument("--model",    help="نموذج HuggingFace")
    parser.add_argument("--gguf",     help="ملف GGUF")
    parser.add_argument("--question", help="سؤال واحد (بدون وضع تفاعلي)")
    parser.add_argument("--top-k",    type=int, default=4, help="عدد القطع للبحث")
    parser.add_argument("--chunk",    type=int, default=50, help="حجم القطعة (سطر)")
    args = parser.parse_args()

    if not args.model and not args.gguf:
        print("❌ حدّد نموذجاً: --model microsoft/phi-2 أو --gguf model.gguf")
        sys.exit(1)

    # بناء الفهرس
    indexer = CodeIndexer(args.project, chunk_size=args.chunk)
    indexer.index()

    # تحميل النموذج
    is_gguf = bool(args.gguf)
    model_p = args.gguf or args.model
    print(f"\n[RAG] تحميل النموذج: {model_p}")
    rag = RAGEngine(indexer, model_p, is_gguf)
    rag.load()
    print("  ✓ جاهز\n")

    if args.question:
        result = rag.ask(args.question, top_k=args.top_k)
        print(f"\n  الإجابة:\n  {result['answer']}")
        print(f"\n  المصادر:")
        for s in result["sources"]:
            print(f"    📄 {s['file']} (سطر {s['lines']})")
    else:
        run_rag_chat(rag)


if __name__ == "__main__":
    main()

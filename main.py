"""
═══════════════════════════════════════════════════════════════════
  Stars AI — نقطة الدخول الرئيسية للنظام
═══════════════════════════════════════════════════════════════════

الملفات الرئيسية في المشروع:
  train_all.py       ← التدريب الشامل (38,000+ مثال، 6 مراحل تلقائية)
  benchmark.py       ← Benchmark عالمي مع 9 نماذج مرجعية
  auto_improve.py    ← تحسين تلقائي — يكتشف الضعف ويعالجه
  generate_data.py   ← توليد بيانات تدريب بـ GPT-4
  compare_models.py  ← مقارنة نماذج متعددة على نفس السؤال
  rag_code.py        ← مساعد يقرأ كودك ويجيب عنه
  finetune_arabic.py ← Fine-tuning للغة العربية
  prune_model.py     ← ضغط النموذج وتصغيره
  finetune_lora.py   ← Fine-tuning بـ LoRA (برمجة)
  evaluate.py        ← تقييم تلقائي على 50 سؤال برمجي
  chat.py            ← محادثة تفاعلية من Terminal
  train_custom.py    ← تدريب StarsLM من الصفر

أوضاع هذا الملف (main.py):
  registry  ← استعراض 17 مزود و70+ نموذج
  train     ← بناء وتدريب نموذج StarsLM من الصفر
  convert   ← تحويل نموذج إلى GGUF
  swarm     ← تشغيل نظام Multi-Agent Swarm بـ CrewAI
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stars_ai.key_manager    import KeyManager
from stars_ai.model_registry import ModelRegistry


# ════════════════════════════════════════════════════════════════════
# وضع 1: بناء وتدريب نموذج StarsLM من الصفر
# ════════════════════════════════════════════════════════════════════

def run_train(args):
    """يبني ويدرّب نموذج StarsLM صغيراً من الصفر."""
    try:
        import torch
        from stars_ai.model_builder import StarsLM, StarsConfig, Trainer, TextDataset
    except ImportError:
        print("[خطأ] pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  Stars AI — تدريب نموذج StarsLM")
    print("=" * 60)
    print("\n  للتدريب الكامل على 38,000+ مثال استخدم:")
    print("  python train_all.py\n")

    cfg = StarsConfig(
        vocab_size  = args.vocab_size,
        hidden_size = args.hidden_size,
        num_layers  = args.num_layers,
        num_heads   = args.num_heads,
        max_seq_len = args.seq_len,
    )
    model = StarsLM(cfg)
    print(f"\n[النموذج] المعاملات: {model.count_parameters():,}")
    print(f"[النموذج] الإعدادات: {cfg.to_dict()}\n")

    # بيانات تجريبية سريعة
    sample_texts = [
        "الذكاء الاصطناعي هو محاكاة العمليات الذهنية البشرية بواسطة الآلات.",
        "نماذج اللغة الكبيرة تعتمد على هندسة Transformer.",
        "Stars AI يدعم أكثر من 17 مزوداً و70 نموذجاً.",
        "GGUF هو تنسيق ثنائي مُحسَّن لتشغيل نماذج اللغة محلياً.",
        "يمكن تكميم النماذج إلى Q4_0 لتقليل حجمها مع الحفاظ على الجودة.",
    ] * 50

    class CharTokenizer:
        def __init__(self, vocab_size=32000):
            self.vocab_size = vocab_size
        def encode(self, text: str) -> list[int]:
            return [ord(c) % self.vocab_size for c in text]

    tokenizer = CharTokenizer(cfg.vocab_size)
    dataset   = TextDataset(sample_texts, tokenizer, seq_len=args.seq_len)
    print(f"[البيانات] عدد العينات: {len(dataset)}")

    trainer = Trainer(
        model,
        dataset,
        batch_size = args.batch_size,
        lr         = args.lr,
        epochs     = args.epochs,
    )
    history = trainer.train()
    print(f"\n[التدريب] اكتمل. الخسائر: {[f'{l:.4f}' for l in history]}")

    save_dir = args.output or "./models/starslm"
    model.save_pretrained(save_dir)

    print("\n[توليد] مثال:")
    import torch
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    sample_ids = torch.tensor(
        [tokenizer.encode("الذكاء")], dtype=torch.long
    ).to(device)
    model.to(device)
    out   = model.generate(sample_ids, max_new_tokens=50)
    chars = [chr(t % 128) for t in out[0].tolist()]
    print("".join(chars))


# ════════════════════════════════════════════════════════════════════
# وضع 2: تحويل نموذج إلى GGUF
# ════════════════════════════════════════════════════════════════════

def run_convert(args):
    """يحوّل نموذجاً إلى صيغة GGUF."""
    from stars_ai.gguf_converter import StarsLMToGGUF, HuggingFaceToGGUF

    print("=" * 60)
    print("  Stars AI — تحويل إلى GGUF")
    print("=" * 60)

    output = args.output or "./models/output.gguf"
    quant  = args.quant  or "q8_0"

    if args.hf:
        print(f"\n[HF→GGUF] نموذج: {args.hf}")
        converter = HuggingFaceToGGUF(
            hf_model_id   = args.hf,
            output_dir    = str(Path(output).parent),
            llama_cpp_dir = args.llama_cpp or "./llama.cpp",
            quant         = quant,
        )
        converter.convert(hf_cache_dir=args.cache_dir)
    else:
        model_dir = args.model_dir or "./models/starslm"
        print(f"\n[StarsLM→GGUF] من: {model_dir}")
        converter = StarsLMToGGUF(
            model_dir   = model_dir,
            output_path = output,
            quant       = quant,
        )
        converter.convert()

    print(f"\n[تم] ملف GGUF محفوظ: {output}")
    print(f"للتشغيل: llama-cli -m {output} -p 'مرحباً' -n 200 --temp 0.8")
    print(f"للمحادثة: python chat.py --gguf {output}")


# ════════════════════════════════════════════════════════════════════
# وضع 3: نظام Multi-Agent Swarm
# ════════════════════════════════════════════════════════════════════

def run_swarm(args):
    """يُشغّل نظام Multi-Agent Swarm بـ CrewAI."""
    try:
        from crewai import Crew, Task, Process
    except ImportError:
        print("[خطأ] pip install crewai langchain-openai")
        sys.exit(1)

    from stars_ai.agents.manager_agent import create_manager_agent
    from stars_ai.agents.expert_agents import get_all_experts

    print("=" * 60)
    print("  Stars AI — Multi-Agent Swarm System")
    print("=" * 60)

    key_manager = KeyManager(use_gcp=args.use_gcp, project_id=args.gcp_project)
    try:
        openai_key = key_manager.get_key("openai")
    except ValueError as e:
        print(f"[خطأ] {e}")
        print("ضبط: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    registry = ModelRegistry()
    summary  = registry.summary()
    print(f"\n[السجل] {summary}")
    print(f"[GGUF] النماذج القابلة للتحويل: {len(registry.get_gguf_compatible())}")

    task_description = args.task or (
        "قارن بين GPT-4o وClaude 3 Sonnet وGemini 1.5 Pro وLlama 4 Scout "
        "لتطبيق chatbot عربي يعمل في بيئة مقيّدة الموارد. "
        "قدّم جدولاً مقارناً وتوصية نهائية مع مبرراتها."
    )
    print(f"\n[المهمة]\n{task_description}\n")

    manager = create_manager_agent(openai_key)
    experts = get_all_experts(openai_key)

    tasks = []
    for expert in experts:
        tasks.append(Task(
            description=(
                f"من منظور خبرتك كـ {expert.role}:\n"
                f"{task_description}\n\n"
                "قدّم:\n"
                "1. نقاط قوة نماذجك لهذه الحالة\n"
                "2. نقاط الضعف أو القيود\n"
                "3. توصيتك المحددة"
            ),
            expected_output=f"تحليل {expert.role} في 200-300 كلمة مع توصية محددة",
            agent=expert,
        ))

    tasks.append(Task(
        description=(
            "بناءً على تحليلات جميع الخبراء، قدّم:\n"
            "1. جدول مقارنة موجز\n"
            "2. توصية نهائية مع تبريرها\n"
            "3. خطة النشر المقترحة (3 خطوات)"
        ),
        expected_output="تقرير تنفيذي واضح",
        agent=manager,
    ))

    crew = Crew(
        agents        = experts + [manager],
        tasks         = tasks,
        process       = Process.hierarchical,
        manager_agent = manager,
        verbose       = True,
    )

    print("[تشغيل] بدء نظام Swarm...\n" + "─" * 60)
    result = crew.kickoff()
    print("\n" + "═" * 60)
    print("  النتيجة النهائية")
    print("═" * 60)
    print(result)


# ════════════════════════════════════════════════════════════════════
# وضع 4: عرض سجل النماذج
# ════════════════════════════════════════════════════════════════════

def run_registry(args):
    """يعرض سجل النماذج الكامل."""
    registry = ModelRegistry()
    summary  = registry.summary()

    print("=" * 60)
    print("  Stars AI — Model Registry")
    print("=" * 60)
    print(f"\nإجمالي المزودين  : {summary['total_providers']}")
    print(f"إجمالي النماذج   : {summary['total_models']}")
    print(f"قابلة للـ GGUF   : {summary['gguf_compatible']}")
    print(f"\nالتوزيع حسب الفئة:")
    for cat, count in sorted(summary["by_category"].items()):
        bar = "█" * count
        print(f"  {cat:12s}: {bar} ({count})")

    if args.provider:
        provider = registry.get_provider(args.provider)
        if not provider:
            print(f"\n[خطأ] المزود غير موجود: {args.provider}")
            available = [p.id for p in registry.get_all_providers()]
            print(f"المزودون المتاحون: {', '.join(available)}")
        else:
            print(f"\n[{provider.name}] النماذج:")
            for m in provider.models:
                gguf_tag = " ✓GGUF" if m.supports_gguf else ""
                ctx = f" [{m.context_window:,} tokens]" if m.context_window else ""
                print(f"  - {m.name:30s} [{m.category}]{ctx}{gguf_tag}")

    if args.gguf_only:
        print("\n[GGUF] النماذج القابلة للتحويل:")
        for m in registry.get_gguf_compatible():
            hf = f" → {m.hf_repo}" if m.hf_repo else ""
            print(f"  - {m.provider_name:15s} | {m.name:30s}{hf}")

    if args.search:
        results = registry.search(args.search)
        print(f"\n[بحث: '{args.search}'] النتائج ({len(results)}):")
        for m in results:
            print(f"  - {m.provider_name:15s} | {m.name:30s} [{m.category}]")

    # ── دليل سريع للملفات الجديدة ───────────────────────────────
    print(f"\n{'═'*60}")
    print("  دليل سريع — الملفات الرئيسية")
    print(f"{'═'*60}")
    files = [
        ("train_all.py",        "التدريب الكامل على 38,000+ مثال (6 مراحل)"),
        ("benchmark.py",        "Benchmark مع 9 نماذج مرجعية عالمية"),
        ("auto_improve.py",     "تحسين تلقائي — يكتشف الضعف ويعالجه"),
        ("generate_data.py",    "توليد بيانات تدريب بـ GPT-4"),
        ("compare_models.py",   "مقارنة عدة نماذج على نفس السؤال"),
        ("rag_code.py",         "مساعد يقرأ كودك ويجيب عنه"),
        ("finetune_arabic.py",  "Fine-tuning للغة العربية"),
        ("prune_model.py",      "ضغط النموذج وتصغيره"),
        ("finetune_lora.py",    "Fine-tuning بـ LoRA للبرمجة"),
        ("evaluate.py",         "تقييم تلقائي على 50 سؤال برمجي"),
        ("chat.py",             "محادثة تفاعلية من Terminal"),
    ]
    for name, desc in files:
        print(f"  python {name:25s} ← {desc}")


# ════════════════════════════════════════════════════════════════════
# نقطة الدخول
# ════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stars-ai",
        description="Stars AI — نظام الوكيل الذكي المتعدد النماذج",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════
  الأوامر الرئيسية الموصى بها:
═══════════════════════════════════════════════════════

  التدريب الكامل (الأفضل):
    python train_all.py

  Benchmark عالمي:
    python benchmark.py --my-model ./models/stars_expert_merged

  تحسين تلقائي:
    python auto_improve.py --model ./models/stars_expert_merged --rounds 3

  مقارنة نماذج:
    python compare_models.py --models phi-2 mistral llama3

  محادثة:
    python chat.py --gguf ./models/stars_expert.gguf

═══════════════════════════════════════════════════════
  أوامر هذا الملف (main.py):
═══════════════════════════════════════════════════════

  python main.py registry --gguf-only
  python main.py registry --provider meta --search llama
  python main.py train --epochs 5 --hidden-size 768 --num-layers 8
  python main.py convert --model-dir ./models/starslm --output model.gguf --quant q4_0
  python main.py convert --hf meta-llama/Llama-3-8B --quant q4_0 --llama-cpp ./llama.cpp
  python main.py swarm --task "قارن أفضل 3 نماذج للترجمة العربية"
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── registry ──────────────────────────────────────────────────
    p_reg = sub.add_parser("registry", help="استعراض سجل النماذج")
    p_reg.add_argument("--provider",  help="عرض نماذج مزود محدد (meta, openai, ...)")
    p_reg.add_argument("--gguf-only", action="store_true", help="النماذج القابلة للـ GGUF فقط")
    p_reg.add_argument("--search",    help="البحث في أسماء النماذج")

    # ── train ──────────────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="بناء وتدريب نموذج StarsLM")
    p_tr.add_argument("--vocab-size",  type=int, default=32000)
    p_tr.add_argument("--hidden-size", type=int, default=512)
    p_tr.add_argument("--num-layers",  type=int, default=6)
    p_tr.add_argument("--num-heads",   type=int, default=8)
    p_tr.add_argument("--seq-len",     type=int, default=256)
    p_tr.add_argument("--batch-size",  type=int, default=8)
    p_tr.add_argument("--epochs",      type=int, default=3)
    p_tr.add_argument("--lr",          type=float, default=3e-4)
    p_tr.add_argument("--output",      help="مسار حفظ النموذج")

    # ── convert ────────────────────────────────────────────────────
    p_cv = sub.add_parser("convert", help="تحويل نموذج إلى GGUF")
    p_cv.add_argument("--model-dir",  help="مسار نموذج StarsLM المحلي")
    p_cv.add_argument("--hf",         help="معرّف نموذج HuggingFace")
    p_cv.add_argument("--output",     help="مسار ملف GGUF الناتج")
    p_cv.add_argument("--quant",      choices=["f32","f16","q8_0","q4_0","q4_1"], default="q8_0")
    p_cv.add_argument("--llama-cpp",  help="مسار مجلد llama.cpp", default="./llama.cpp")
    p_cv.add_argument("--cache-dir",  help="مجلد تخزين مؤقت لـ HuggingFace")

    # ── swarm ──────────────────────────────────────────────────────
    p_sw = sub.add_parser("swarm", help="تشغيل نظام Multi-Agent Swarm")
    p_sw.add_argument("--task",       help="المهمة المطلوب تحليلها")
    p_sw.add_argument("--use-gcp",    action="store_true")
    p_sw.add_argument("--gcp-project",help="معرّف مشروع GCP")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "registry": run_registry,
        "train":    run_train,
        "convert":  run_convert,
        "swarm":    run_swarm,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

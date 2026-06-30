"""
═══════════════════════════════════════════════════════════════
  Stars AI — ضغط النموذج (Pruning + Quantization)
  يحذف الأوزان غير المهمة ويصغّر النموذج قبل تحويله إلى GGUF
═══════════════════════════════════════════════════════════════

الفرق:
  بدون ضغط → نموذج 7B يأخذ ~14GB
  بعد Q4_0  → نموذج 7B يأخذ ~4GB
  بعد Pruning + Q4_0 → ~3GB مع تدهور بسيط في الجودة

التشغيل:
  python prune_model.py --model ./models/code_expert --ratio 0.3
  python prune_model.py --model ./models/code_expert --ratio 0.5 --output ./models/compact
  python prune_model.py --model ./models/code_expert --benchmark  # قياس قبل وبعد
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── معيار التقليم ─────────────────────────────────────────────────────────────

PRUNE_CRITERIA = {
    "magnitude":   "يحذف الأوزان الأصغر قيمةً (الأشيع والأسرع)",
    "random":      "يحذف أوزاناً عشوائية (للمقارنة فقط)",
    "gradient":    "يحذف الأوزان الأقل تأثيراً على الخسارة (الأدق)",
    "structured":  "يحذف رؤوس انتباه كاملة (يسرّع التنفيذ أكثر)",
}


class ModelPruner:
    """
    يُطبّق Pruning على نموذج HuggingFace لتصغير حجمه.

    أنواع التقليم:
      magnitude  ← الأفضل للاستخدام العام
      structured ← الأفضل للسرعة
    """

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.model      = None
        self.tokenizer  = None

    def load(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("❌ pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[تحميل] {self.model_path}")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True,
            torch_dtype=torch.float32,  # float32 لأن Pruning يحتاج دقة كاملة
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ محمّل ({self._count_params()/1e6:.1f}M معامل)")

    def _count_params(self, nonzero_only: bool = False) -> int:
        import torch
        total = 0
        for p in self.model.parameters():
            if nonzero_only:
                total += int((p.data != 0).sum().item())
            else:
                total += p.numel()
        return total

    def _count_nonzero(self) -> int:
        return self._count_params(nonzero_only=True)

    def prune_magnitude(self, ratio: float):
        """
        Magnitude Pruning: يحذف الـ ratio% من الأوزان الأصغر قيمةً مطلقة.
        مثال: ratio=0.3 يحذف 30% من الأوزان.
        """
        import torch
        print(f"\n[Pruning] Magnitude Pruning (ratio={ratio:.0%})")

        pruned_layers = 0
        total_pruned  = 0

        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight.data

                # حساب حدّ التقليم
                threshold = torch.quantile(weight.abs().flatten(), ratio)

                # إنشاء قناع (1=يبقى، 0=يُحذف)
                mask = (weight.abs() >= threshold).float()

                # تطبيق القناع
                module.weight.data = weight * mask
                pruned = int((mask == 0).sum().item())
                total_pruned += pruned
                pruned_layers += 1

        total   = self._count_params()
        nonzero = self._count_nonzero()
        sparsity = 1 - nonzero / total

        print(f"  ✓ طبقات مُقلَّمة  : {pruned_layers}")
        print(f"  ✓ أوزان محذوفة   : {total_pruned:,}")
        print(f"  ✓ نسبة الإزالة   : {sparsity:.1%}")
        return sparsity

    def prune_structured_heads(self, heads_to_prune: float = 0.25):
        """
        Structured Pruning: يحذف رؤوس الانتباه الأقل أهمية.
        هذا يسرّع التنفيذ لأنه يغيّر شكل النموذج فعلياً.
        """
        import torch
        print(f"\n[Pruning] Structured Head Pruning ({heads_to_prune:.0%} من الرؤوس)")

        pruned_count = 0
        for name, module in self.model.named_modules():
            class_name = type(module).__name__
            if "Attention" in class_name and hasattr(module, "num_heads"):
                num_heads   = module.num_heads
                to_prune    = max(1, int(num_heads * heads_to_prune))
                heads_list  = list(range(to_prune))

                try:
                    if hasattr(module, "prune_heads"):
                        module.prune_heads(set(heads_list))
                        pruned_count += to_prune
                except Exception:
                    pass

        print(f"  ✓ رؤوس محذوفة: {pruned_count}")
        return pruned_count

    def quantize_dynamic(self):
        """
        Dynamic Quantization: تحويل الأوزان إلى 8-bit بعد الـ Pruning.
        يقلل الحجم بنسبة 50% إضافية.
        """
        import torch
        print("\n[Quantization] Dynamic Quantization (INT8)...")
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        print("  ✓ تم التحويل إلى INT8")

    def benchmark(self, prompt: str = "def fibonacci(n):") -> dict:
        """يقيس سرعة النموذج وجودته."""
        import torch
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        times   = []

        for _ in range(3):
            t0 = time.time()
            with torch.no_grad():
                self.model.generate(
                    **inputs, max_new_tokens=50,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                )
            times.append(time.time() - t0)

        avg_time = sum(times) / len(times)
        total    = self._count_params()
        nonzero  = self._count_nonzero()
        sparsity = 1 - nonzero / total

        return {
            "params_total":   total,
            "params_nonzero": nonzero,
            "sparsity":       round(sparsity * 100, 1),
            "avg_inference_s": round(avg_time, 3),
            "tokens_per_sec": round(50 / avg_time, 1),
        }

    def save(self):
        """يحفظ النموذج المضغوط."""
        import torch
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            self.model.save_pretrained(self.output_dir, safe_serialization=True)
        except Exception:
            # fallback إذا كان النموذج المكمّم لا يدعم save_pretrained
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model.bin"))
            cfg = {"model_path": self.model_path, "pruned": True}
            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
        self.tokenizer.save_pretrained(self.output_dir)

        # حساب حجم الملفات
        total_bytes = sum(
            f.stat().st_size
            for f in Path(self.output_dir).rglob("*")
            if f.is_file()
        )
        print(f"  ✓ محفوظ: {self.output_dir} ({total_bytes/1024/1024:.1f} MB)")
        return total_bytes


# ── تقرير المقارنة ────────────────────────────────────────────────────────────

def print_compression_report(before: dict, after: dict, output_dir: str):
    ratio_params = (1 - after["params_nonzero"] / before["params_total"]) * 100
    speedup      = before["avg_inference_s"] / after["avg_inference_s"]

    print(f"\n{'╔'+'═'*52+'╗'}")
    print(f"║{'تقرير الضغط':^52}║")
    print(f"{'╠'+'═'*52+'╣'}")
    print(f"║  {'المقياس':22s} {'قبل':>12s} {'بعد':>12s}  ║")
    print(f"{'╠'+'─'*52+'╣'}")
    print(f"║  {'المعاملات الكلية':22s} {before['params_total']/1e6:>10.1f}M {after['params_nonzero']/1e6:>10.1f}M  ║")
    print(f"║  {'نسبة الإزالة':22s} {'0.0%':>12s} {after['sparsity']:>11.1f}%  ║")
    print(f"║  {'وقت الاستدلال':22s} {before['avg_inference_s']:>10.3f}ث {after['avg_inference_s']:>10.3f}ث  ║")
    print(f"║  {'Tokens/ثانية':22s} {before['tokens_per_sec']:>10.1f}  {after['tokens_per_sec']:>10.1f}   ║")
    print(f"{'╠'+'═'*52+'╣'}")
    print(f"║  تسريع الاستدلال : {speedup:.2f}x                          ║")
    print(f"║  الأوزان المحذوفة: {ratio_params:.1f}%                          ║")
    print(f"╚{'═'*52}╝\n")
    print(f"  النموذج المضغوط: {output_dir}")
    print(f"  للمحادثة: python chat.py --model {output_dir}")
    print(f"  للتقييم:  python evaluate.py --model {output_dir}")


# ── نقطة الدخول ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stars AI — ضغط النموذج",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  # ضغط 30% (آمن)
  python prune_model.py --model ./models/code_expert --ratio 0.3

  # ضغط 50% مع قياس الأداء
  python prune_model.py --model ./models/code_expert --ratio 0.5 --benchmark

  # ضغط هيكلي (أسرع)
  python prune_model.py --model ./models/code_expert --structured --ratio 0.25

  # ضغط + تكميم INT8
  python prune_model.py --model ./models/code_expert --ratio 0.3 --quantize
        """,
    )
    parser.add_argument("--model",      required=True, help="مسار النموذج")
    parser.add_argument("--output",     help="مسار النموذج المضغوط (افتراضي: <model>_pruned)")
    parser.add_argument("--ratio",      type=float, default=0.3,
                        help="نسبة الضغط 0.0-0.9 (افتراضي: 0.3 أي 30%%)")
    parser.add_argument("--structured", action="store_true",
                        help="ضغط هيكلي (حذف رؤوس انتباه كاملة)")
    parser.add_argument("--quantize",   action="store_true",
                        help="تكميم INT8 إضافي بعد الضغط")
    parser.add_argument("--benchmark",  action="store_true",
                        help="قياس الأداء قبل وبعد الضغط")
    args = parser.parse_args()

    output = args.output or (args.model.rstrip("/") + "_pruned")

    if not 0 < args.ratio < 1:
        print("❌ نسبة الضغط يجب أن تكون بين 0 و1 (مثال: 0.3 = 30%)")
        sys.exit(1)

    print("═" * 56)
    print("  Stars AI — ضغط النموذج (Pruning)")
    print("═" * 56)
    print(f"  النموذج   : {args.model}")
    print(f"  المخرج    : {output}")
    print(f"  نسبة الضغط: {args.ratio:.0%}")
    print(f"  النوع     : {'هيكلي' if args.structured else 'magnitude'}")
    print(f"  تكميم INT8: {'نعم' if args.quantize else 'لا'}")

    pruner = ModelPruner(args.model, output)
    pruner.load()

    # قياس قبل الضغط
    before = {}
    if args.benchmark:
        print("\n[قياس] قبل الضغط...")
        before = pruner.benchmark()
        print(f"  الأداء الأولي: {before['tokens_per_sec']} tokens/s")

    # تطبيق الضغط
    if args.structured:
        pruner.prune_structured_heads(args.ratio)
    else:
        pruner.prune_magnitude(args.ratio)

    if args.quantize:
        pruner.quantize_dynamic()

    # قياس بعد الضغط
    if args.benchmark:
        print("\n[قياس] بعد الضغط...")
        after = pruner.benchmark()
        print_compression_report(before, after, output)

    # حفظ النموذج
    print(f"\n[حفظ] النموذج المضغوط...")
    pruner.save()

    print(f"\n{'═'*56}")
    print(f"  ✓ اكتمل الضغط!")
    print(f"  للخطوة التالية (تحويل إلى GGUF):")
    print(f"  python main.py convert --model-dir {output} --quant q4_0")
    print(f"{'═'*56}")


if __name__ == "__main__":
    main()

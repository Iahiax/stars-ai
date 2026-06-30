"""
Stars AI — نظام الوكيل الذكي المتعدد النماذج

الإصدار 2.0 — يشمل:
  - Fine-tuning بـ LoRA (برمجة + عربي)
  - توليد بيانات تلقائي
  - Benchmark مع 9 نماذج عالمية
  - تحسين تلقائي (auto_improve)
  - ضغط النموذج (Pruning)
  - RAG على ملفات الكود
  - مقارنة نماذج متعددة
"""

__version__ = "2.0.0"
__author__  = "Stars AI Team"

from stars_ai.key_manager    import KeyManager
from stars_ai.model_registry  import ModelRegistry, ModelInfo, ProviderInfo
from stars_ai.model_builder   import StarsLM, StarsConfig, Trainer, TextDataset
from stars_ai.gguf_converter  import StarsLMToGGUF, HuggingFaceToGGUF, GGUFWriter

__all__ = [
    "KeyManager",
    "ModelRegistry",
    "ModelInfo",
    "ProviderInfo",
    "StarsLM",
    "StarsConfig",
    "Trainer",
    "TextDataset",
    "StarsLMToGGUF",
    "HuggingFaceToGGUF",
    "GGUFWriter",
]

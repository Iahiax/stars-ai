"""
Model Registry: سجل شامل لجميع نماذج الذكاء الاصطناعي ومزوديها.
"""

from dataclasses import dataclass, field
from typing import Literal

ModelCategory = Literal["llm", "image", "video", "audio", "agent", "multimodal"]


@dataclass
class ModelInfo:
    """معلومات نموذج واحد."""
    id: str
    name: str
    provider_id: str
    provider_name: str
    category: ModelCategory
    description: str = ""
    context_window: int | None = None
    supports_gguf: bool = False
    hf_repo: str | None = None


@dataclass
class ProviderInfo:
    """معلومات مزود واحد."""
    id: str
    name: str
    website: str
    models: list[ModelInfo] = field(default_factory=list)


class ModelRegistry:
    """
    سجل مركزي لجميع نماذج الذكاء الاصطناعي.
    يدعم البحث والتصفية حسب المزود أو الفئة أو معرّف النموذج.
    """

    def __init__(self):
        self._providers: dict[str, ProviderInfo] = {}
        self._models: dict[str, ModelInfo] = {}
        self._build_registry()

    def _register(self, provider: ProviderInfo):
        self._providers[provider.id] = provider
        for model in provider.models:
            self._models[model.id] = model

    def _build_registry(self):
        """يبني السجل الكامل لجميع المزودين والنماذج."""

        # ── 1. OpenAI ──────────────────────────────────────────────────
        self._register(ProviderInfo(
            id="openai", name="OpenAI", website="https://openai.com",
            models=[
                ModelInfo("gpt-5",       "GPT-5",          "openai", "OpenAI", "llm",   context_window=200_000),
                ModelInfo("gpt-4o",      "GPT-4o",         "openai", "OpenAI", "llm",   context_window=128_000),
                ModelInfo("gpt-4-1",     "GPT-4.1",        "openai", "OpenAI", "llm",   context_window=128_000),
                ModelInfo("gpt-4",       "GPT-4",          "openai", "OpenAI", "llm",   context_window=8_192),
                ModelInfo("gpt-3.5",     "GPT-3.5",        "openai", "OpenAI", "llm",   context_window=16_385),
                ModelInfo("o3-mini",     "o3-mini",        "openai", "OpenAI", "llm"),
                ModelInfo("operator",    "Operator",       "openai", "OpenAI", "agent"),
                ModelInfo("dall-e-3",    "DALL·E 3",       "openai", "OpenAI", "image"),
                ModelInfo("gpt-image-2", "GPT-Image-2",    "openai", "OpenAI", "image"),
                ModelInfo("sora",        "Sora",           "openai", "OpenAI", "video"),
                ModelInfo("sora-turbo",  "Sora Turbo",     "openai", "OpenAI", "video"),
                ModelInfo("whisper",     "Whisper",        "openai", "OpenAI", "audio"),
            ]
        ))

        # ── 2. Anthropic ───────────────────────────────────────────────
        self._register(ProviderInfo(
            id="anthropic", name="Anthropic", website="https://anthropic.com",
            models=[
                ModelInfo("claude-4",            "Claude 4",          "anthropic", "Anthropic", "llm", context_window=200_000),
                ModelInfo("claude-3-opus",        "Claude 3 Opus",     "anthropic", "Anthropic", "llm", context_window=200_000),
                ModelInfo("claude-3-sonnet",      "Claude 3 Sonnet",   "anthropic", "Anthropic", "llm", context_window=200_000),
                ModelInfo("claude-3-haiku",       "Claude 3 Haiku",    "anthropic", "Anthropic", "llm", context_window=200_000),
                ModelInfo("claude-2",             "Claude 2",          "anthropic", "Anthropic", "llm"),
                ModelInfo("claude-1",             "Claude 1",          "anthropic", "Anthropic", "llm"),
            ]
        ))

        # ── 3. Google DeepMind ─────────────────────────────────────────
        self._register(ProviderInfo(
            id="google", name="Google DeepMind", website="https://deepmind.google",
            models=[
                ModelInfo("gemini-2.0-pro-exp",  "Gemini 2.0 Pro Experimental", "google", "Google", "llm", context_window=1_000_000),
                ModelInfo("gemini-1.5-pro",       "Gemini 1.5 Pro",              "google", "Google", "llm", context_window=1_000_000),
                ModelInfo("gemini-1.5-ultra",     "Gemini 1.5 Ultra",            "google", "Google", "llm"),
                ModelInfo("gemini-1.5-flash",     "Gemini 1.5 Flash",            "google", "Google", "llm"),
                ModelInfo("gemini-1.5-nano",      "Gemini 1.5 Nano",             "google", "Google", "llm"),
                ModelInfo("gemma-3",              "Gemma 3",                     "google", "Google", "llm", supports_gguf=True, hf_repo="google/gemma-3"),
                ModelInfo("gemma-4-31b",          "Gemma 4 31B",                 "google", "Google", "llm", supports_gguf=True, hf_repo="google/gemma-4-31b"),
                ModelInfo("veo-2",                "Google Veo 2",                "google", "Google", "video"),
            ]
        ))

        # ── 4. Meta ────────────────────────────────────────────────────
        self._register(ProviderInfo(
            id="meta", name="Meta", website="https://ai.meta.com",
            models=[
                ModelInfo("llama-4-scout",    "Llama 4 Scout",    "meta", "Meta", "llm", supports_gguf=True, hf_repo="meta-llama/Llama-4-Scout"),
                ModelInfo("llama-4-maverick", "Llama 4 Maverick", "meta", "Meta", "llm", supports_gguf=True, hf_repo="meta-llama/Llama-4-Maverick"),
                ModelInfo("llama-3",          "Llama 3",          "meta", "Meta", "llm", supports_gguf=True, hf_repo="meta-llama/Meta-Llama-3-8B"),
                ModelInfo("llama-2",          "Llama 2",          "meta", "Meta", "llm", supports_gguf=True, hf_repo="meta-llama/Llama-2-7b-hf"),
                ModelInfo("llama-1",          "Llama 1",          "meta", "Meta", "llm", supports_gguf=True),
                ModelInfo("voicebox",         "Voicebox",         "meta", "Meta", "audio"),
            ]
        ))

        # ── 5. Microsoft ───────────────────────────────────────────────
        self._register(ProviderInfo(
            id="microsoft", name="Microsoft", website="https://microsoft.com/ai",
            models=[
                ModelInfo("phi-3",               "Phi-3",                "microsoft", "Microsoft", "llm",         supports_gguf=True, hf_repo="microsoft/Phi-3-mini-4k-instruct"),
                ModelInfo("phi-2",               "Phi-2",                "microsoft", "Microsoft", "llm",         supports_gguf=True, hf_repo="microsoft/phi-2"),
                ModelInfo("orca-2",              "Orca 2",               "microsoft", "Microsoft", "llm"),
                ModelInfo("orca-mini",           "Orca Mini",            "microsoft", "Microsoft", "llm",         supports_gguf=True),
                ModelInfo("kosmos-2",            "Kosmos-2",             "microsoft", "Microsoft", "multimodal"),
                ModelInfo("kosmos-1",            "Kosmos-1",             "microsoft", "Microsoft", "multimodal"),
                ModelInfo("florence",            "Florence",             "microsoft", "Microsoft", "multimodal"),
                ModelInfo("vall-e",              "VALL-E",               "microsoft", "Microsoft", "audio"),
                ModelInfo("azure-openai",        "Azure OpenAI Models",  "microsoft", "Microsoft", "llm"),
                ModelInfo("azure-vision",        "Azure Vision Models",  "microsoft", "Microsoft", "multimodal"),
                ModelInfo("azure-speech",        "Azure Speech Models",  "microsoft", "Microsoft", "audio"),
            ]
        ))

        # ── 6. xAI ────────────────────────────────────────────────────
        self._register(ProviderInfo(
            id="xai", name="xAI", website="https://x.ai",
            models=[
                ModelInfo("grok-3",  "Grok 3",  "xai", "xAI", "llm"),
                ModelInfo("grok-2",  "Grok 2",  "xai", "xAI", "llm"),
                ModelInfo("grok-1",  "Grok 1",  "xai", "xAI", "llm", supports_gguf=True, hf_repo="xai-org/grok-1"),
            ]
        ))

        # ── 7. Mistral AI ──────────────────────────────────────────────
        self._register(ProviderInfo(
            id="mistral", name="Mistral AI", website="https://mistral.ai",
            models=[
                ModelInfo("mistral-small-4",   "Mistral Small 4",  "mistral", "Mistral", "llm"),
                ModelInfo("mistral-large-3",   "Mistral Large 3",  "mistral", "Mistral", "llm"),
                ModelInfo("mixtral-8x7b",      "Mixtral 8x7B",     "mistral", "Mistral", "llm", supports_gguf=True, hf_repo="mistralai/Mixtral-8x7B-v0.1"),
                ModelInfo("mixtral-8x22b",     "Mixtral 8x22B",    "mistral", "Mistral", "llm", supports_gguf=True, hf_repo="mistralai/Mixtral-8x22B-v0.1"),
                ModelInfo("le-chat",           "Le Chat",          "mistral", "Mistral", "multimodal"),
            ]
        ))

        # ── 8. DeepSeek ────────────────────────────────────────────────
        self._register(ProviderInfo(
            id="deepseek", name="DeepSeek", website="https://deepseek.com",
            models=[
                ModelInfo("deepseek-v4-pro",  "DeepSeek V4 Pro",  "deepseek", "DeepSeek", "llm", supports_gguf=True, hf_repo="deepseek-ai/DeepSeek-V4-Pro"),
                ModelInfo("deepseek-v3",      "DeepSeek V3",      "deepseek", "DeepSeek", "llm", supports_gguf=True, hf_repo="deepseek-ai/DeepSeek-V3"),
                ModelInfo("deepseek-r1",      "DeepSeek R1",      "deepseek", "DeepSeek", "llm", supports_gguf=True, hf_repo="deepseek-ai/DeepSeek-R1"),
            ]
        ))

        # ── 9. Moonshot AI ─────────────────────────────────────────────
        self._register(ProviderInfo(
            id="moonshot", name="Moonshot AI", website="https://moonshot.ai",
            models=[
                ModelInfo("kimi-k2.6",  "Kimi K2.6",  "moonshot", "Moonshot", "llm"),
            ]
        ))

        # ── 10. Alibaba Cloud ──────────────────────────────────────────
        self._register(ProviderInfo(
            id="alibaba", name="Alibaba Cloud", website="https://qwen.aliyun.com",
            models=[
                ModelInfo("qwen-3.6-35b",  "Qwen 3.6-35B-A3B",  "alibaba", "Alibaba", "llm", supports_gguf=True, hf_repo="Qwen/Qwen2-72B-Instruct"),
                ModelInfo("qwen-2",        "Qwen 2",             "alibaba", "Alibaba", "llm", supports_gguf=True),
                ModelInfo("qwen-1.5",      "Qwen 1.5",           "alibaba", "Alibaba", "llm", supports_gguf=True),
            ]
        ))

        # ── 11. Zhipu AI ───────────────────────────────────────────────
        self._register(ProviderInfo(
            id="zhipu", name="Zhipu AI", website="https://zhipuai.cn",
            models=[
                ModelInfo("glm-5",  "GLM-5",  "zhipu", "Zhipu", "llm"),
                ModelInfo("glm-4",  "GLM-4",  "zhipu", "Zhipu", "llm", supports_gguf=True, hf_repo="THUDM/glm-4-9b-chat"),
                ModelInfo("glm-3",  "GLM-3",  "zhipu", "Zhipu", "llm"),
            ]
        ))

        # ── 12. IBM ────────────────────────────────────────────────────
        self._register(ProviderInfo(
            id="ibm", name="IBM", website="https://ibm.com/granite",
            models=[
                ModelInfo("granite-3b",      "Granite 3B",       "ibm", "IBM", "llm",         supports_gguf=True, hf_repo="ibm-granite/granite-3b-code-base"),
                ModelInfo("granite-7b",      "Granite 7B",       "ibm", "IBM", "llm",         supports_gguf=True),
                ModelInfo("granite-20b",     "Granite 20B",      "ibm", "IBM", "llm"),
                ModelInfo("granite-34b",     "Granite 34B",      "ibm", "IBM", "llm"),
                ModelInfo("granite-vision",  "Granite Vision",   "ibm", "IBM", "multimodal"),
                ModelInfo("granite-code",    "Granite Code",     "ibm", "IBM", "llm",         supports_gguf=True),
            ]
        ))

        # ── 13. Dolphin (Open Source) ──────────────────────────────────
        self._register(ProviderInfo(
            id="dolphin", name="Dolphin (Open Source)", website="https://erichartford.com",
            models=[
                ModelInfo("dolphin-2.9",           "Dolphin-2.9",           "dolphin", "Dolphin", "llm", supports_gguf=True, hf_repo="cognitivecomputations/dolphin-2.9-llama3-8b"),
                ModelInfo("dolphin-3.0",           "Dolphin-3.0",           "dolphin", "Dolphin", "llm", supports_gguf=True, hf_repo="cognitivecomputations/dolphin-3.0-llama3.2-1b"),
                ModelInfo("dolphin-mixtral-8x7b",  "Dolphin-Mixtral-8x7B",  "dolphin", "Dolphin", "llm", supports_gguf=True, hf_repo="cognitivecomputations/dolphin-2.7-mixtral-8x7b"),
                ModelInfo("dolphin-llama-3-70b",   "Dolphin-Llama-3-70B",   "dolphin", "Dolphin", "llm", supports_gguf=True),
                ModelInfo("dolphin-coder",         "Dolphin-Coder",         "dolphin", "Dolphin", "llm", supports_gguf=True),
            ]
        ))

        # ── 14. Image Models ───────────────────────────────────────────
        self._register(ProviderInfo(
            id="image-providers", name="Image Providers", website="",
            models=[
                ModelInfo("midjourney-v7",  "Midjourney V7",        "image-providers", "Image", "image"),
                ModelInfo("sdxl",           "Stable Diffusion XL",  "image-providers", "Image", "image"),
                ModelInfo("runway-gen2",    "Runway Gen-2",         "image-providers", "Image", "image"),
                ModelInfo("pika-labs",      "Pika Labs",            "image-providers", "Image", "image"),
            ]
        ))

        # ── 15. Audio Models ───────────────────────────────────────────
        self._register(ProviderInfo(
            id="audio-providers", name="Audio Providers", website="",
            models=[
                ModelInfo("elevenlabs",  "ElevenLabs",  "audio-providers", "Audio", "audio"),
                ModelInfo("audiolm",     "AudioLM",     "audio-providers", "Audio", "audio"),
            ]
        ))

        # ── 16. Video Models ───────────────────────────────────────────
        self._register(ProviderInfo(
            id="video-providers", name="Video Providers", website="",
            models=[
                ModelInfo("stability-video-4d",  "Stability Video 4D",  "video-providers", "Video", "video"),
            ]
        ))

        # ── 17. Agent Models ───────────────────────────────────────────
        self._register(ProviderInfo(
            id="agents-providers", name="Agent Providers", website="",
            models=[
                ModelInfo("crewai",                     "CrewAI",                       "agents-providers", "Agents", "agent"),
                ModelInfo("perplexity-deep-search",     "Perplexity Deep Search",       "agents-providers", "Agents", "agent"),
            ]
        ))

    # ── Public API ────────────────────────────────────────────────────

    def get_all_providers(self) -> list[ProviderInfo]:
        return list(self._providers.values())

    def get_provider(self, provider_id: str) -> ProviderInfo | None:
        return self._providers.get(provider_id)

    def get_all_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def filter_by_category(self, category: ModelCategory) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.category == category]

    def filter_by_provider(self, provider_id: str) -> list[ModelInfo]:
        return [m for m in self._models.values() if m.provider_id == provider_id]

    def get_gguf_compatible(self) -> list[ModelInfo]:
        """يُعيد فقط النماذج التي يمكن تحويلها إلى GGUF."""
        return [m for m in self._models.values() if m.supports_gguf]

    def search(self, query: str) -> list[ModelInfo]:
        """يبحث في أسماء النماذج والمزودين."""
        q = query.lower()
        return [m for m in self._models.values()
                if q in m.name.lower() or q in m.provider_name.lower()]

    def summary(self) -> dict:
        """ملخص إحصائي للسجل."""
        from collections import Counter
        cats = Counter(m.category for m in self._models.values())
        return {
            "total_providers": len(self._providers),
            "total_models": len(self._models),
            "gguf_compatible": len(self.get_gguf_compatible()),
            "by_category": dict(cats),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ModelRegistry("
            f"providers={s['total_providers']}, "
            f"models={s['total_models']}, "
            f"gguf_ready={s['gguf_compatible']})"
        )

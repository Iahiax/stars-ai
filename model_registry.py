## “””
model_registry.py

سجل مركزي (Model Registry) يربط كل نموذج ذكاء اصطناعي بمزوده الفعلي
(لتحديد مفتاح API الصحيح عبر KeyManager) وبتصنيفه (llm / image / video /
audio / multimodal / vision / agent / code).

ملاحظة هامة: بعض الأسماء أدناه (مثل GPT-5، Claude 4، Gemini 2.0 Pro
Experimental، DeepSeek V4 Pro…) تمثل تسميات استشرافية/قابلة للتغيير ذكرها
المستخدم في الطلب. السجل مبني بحيث يمكن تعديل أي اسم نموذج أو إضافة نماذج
جديدة دون تغيير منطق الكود — فقط أضف/عدّل المفاتيح في القاموس أدناه.
“””

MODEL_REGISTRY = {
# –––––––– OpenAI ––––––––
“gpt-5”:            {“provider”: “openai”, “category”: “llm”},
“gpt-4.1”:           {“provider”: “openai”, “category”: “llm”},
“gpt-4o”:            {“provider”: “openai”, “category”: “llm”},
“gpt-3.5”:           {“provider”: “openai”, “category”: “llm”},
“o3-mini”:           {“provider”: “openai”, “category”: “llm”},
“deep-research”:     {“provider”: “openai”, “category”: “agent”},
“operator”:          {“provider”: “openai”, “category”: “agent”},
“dalle-3”:           {“provider”: “openai”, “category”: “image”},
“gpt-image-2”:       {“provider”: “openai”, “category”: “image”},
“sora”:              {“provider”: “openai”, “category”: “video”},
“sora-turbo”:        {“provider”: “openai”, “category”: “video”},
“whisper”:           {“provider”: “openai”, “category”: “audio”},

```
# ---------------- Anthropic ----------------
"claude-4":          {"provider": "anthropic", "category": "llm"},
"claude-3-opus":     {"provider": "anthropic", "category": "llm"},
"claude-3-sonnet":   {"provider": "anthropic", "category": "llm"},
"claude-3-haiku":    {"provider": "anthropic", "category": "llm"},
"claude-2":          {"provider": "anthropic", "category": "llm"},
"claude-1":          {"provider": "anthropic", "category": "llm"},

# ---------------- Google DeepMind ----------------
"gemini-2.0-pro-exp": {"provider": "google", "category": "llm"},
"gemini-1.5-pro":     {"provider": "google", "category": "llm"},
"gemini-1.5-ultra":   {"provider": "google", "category": "llm"},
"gemini-1.5-flash":   {"provider": "google", "category": "llm"},
"gemini-1.5-nano":    {"provider": "google", "category": "llm"},
"gemma-3":            {"provider": "google", "category": "llm"},
"veo-2":              {"provider": "google", "category": "video"},

# ---------------- Meta ----------------
"llama-4-scout":      {"provider": "meta", "category": "llm"},
"llama-4-maverick":   {"provider": "meta", "category": "llm"},
"llama-3":            {"provider": "meta", "category": "llm"},
"llama-2":            {"provider": "meta", "category": "llm"},
"llama-1":            {"provider": "meta", "category": "llm"},
"voicebox":           {"provider": "meta", "category": "audio"},

# ---------------- Microsoft / Azure ----------------
"phi-3":              {"provider": "microsoft", "category": "llm"},
"phi-2":               {"provider": "microsoft", "category": "llm"},
"orca-2":              {"provider": "microsoft", "category": "llm"},
"orca-mini":           {"provider": "microsoft", "category": "llm"},
"kosmos-2":            {"provider": "microsoft", "category": "multimodal"},
"kosmos-1":            {"provider": "microsoft", "category": "multimodal"},
"florence":            {"provider": "microsoft", "category": "vision"},
"valle":               {"provider": "microsoft", "category": "audio"},
"azure-openai":        {"provider": "microsoft", "category": "llm"},
"azure-vision":        {"provider": "microsoft", "category": "vision"},
"azure-speech":        {"provider": "microsoft", "category": "audio"},

# ---------------- xAI ----------------
"grok-3":              {"provider": "xai", "category": "llm"},
"grok-2":              {"provider": "xai", "category": "llm"},
"grok-1":              {"provider": "xai", "category": "llm"},

# ---------------- Mistral AI ----------------
"mistral-small":       {"provider": "mistral", "category": "llm"},
"mistral-large":       {"provider": "mistral", "category": "llm"},
"mixtral-8x7b":        {"provider": "mistral", "category": "llm"},
"mixtral-8x22b":       {"provider": "mistral", "category": "llm"},
"le-chat":             {"provider": "mistral", "category": "agent"},

# ---------------- DeepSeek ----------------
"deepseek-v4-pro":     {"provider": "deepseek", "category": "llm"},
"deepseek-v3":         {"provider": "deepseek", "category": "llm"},
"deepseek-r1":         {"provider": "deepseek", "category": "llm"},

# ---------------- Moonshot AI ----------------
"kimi-k2":             {"provider": "moonshot", "category": "llm"},

# ---------------- Alibaba Cloud ----------------
"qwen-2":              {"provider": "alibaba", "category": "llm"},
"qwen-1.5":            {"provider": "alibaba", "category": "llm"},

# ---------------- Zhipu AI ----------------
"glm-4":               {"provider": "zhipu", "category": "llm"},
"glm-3":               {"provider": "zhipu", "category": "llm"},

# ---------------- IBM ----------------
"granite-3b":          {"provider": "ibm", "category": "llm"},
"granite-7b":          {"provider": "ibm", "category": "llm"},
"granite-20b":         {"provider": "ibm", "category": "llm"},
"granite-34b":         {"provider": "ibm", "category": "llm"},
"granite-vision":      {"provider": "ibm", "category": "vision"},
"granite-code":        {"provider": "ibm", "category": "code"},

# ---------------- نماذج صور / فيديو / صوت من مزودين متخصصين ----------------
"midjourney-v7":       {"provider": "midjourney", "category": "image"},
"stable-diffusion-xl": {"provider": "stability", "category": "image"},
"stable-video-4d":     {"provider": "stability", "category": "video"},
"runway-gen-2":        {"provider": "runway", "category": "video"},
"pika-labs":           {"provider": "pika", "category": "video"},
"elevenlabs":          {"provider": "elevenlabs", "category": "audio"},
"audiolm":             {"provider": "google", "category": "audio"},

# ---------------- نماذج/أطر الوكلاء الذكية ----------------
"crewai-agent":        {"provider": "internal", "category": "agent"},
"perplexity-deep-search": {"provider": "perplexity", "category": "agent"},
```

}

def get_models_by_category(category: str) -> list:
“”“إرجاع أسماء جميع النماذج التي تنتمي إلى تصنيف معيّن.”””
return [name for name, meta in MODEL_REGISTRY.items() if meta[“category”] == category]

def get_models_by_provider(provider: str) -> list:
“”“إرجاع أسماء جميع النماذج التابعة لمزود معيّن.”””
return [name for name, meta in MODEL_REGISTRY.items() if meta[“provider”] == provider]

def get_model_provider(model_name: str) -> str:
“”“إرجاع اسم المزود الفعلي لنموذج معيّن (لاستخدامه مباشرة مع KeyManager).”””
meta = MODEL_REGISTRY.get(model_name)
if not meta:
raise ValueError(f”النموذج ‘{model_name}’ غير مسجل في MODEL_REGISTRY”)
return meta[“provider”]

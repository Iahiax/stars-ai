"""
Expert Agents: وكلاء خبراء يمثلون مزودي الذكاء الاصطناعي المختلفين.
كل وكيل متخصص في نماذج مزوده ويقدم رأياً مبنياً على قدراتها.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI


def _base_llm(api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
    """مصنع LLM مشترك لجميع الوكلاء."""
    return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)


def openai_expert(openai_key: str) -> Agent:
    """وكيل OpenAI — خبير في نماذج GPT وDALL·E وSora."""
    return Agent(
        role="OpenAI Expert",
        goal=(
            "تحليل المهام واقتراح أفضل نماذج OpenAI (GPT-5، GPT-4o، o3-mini، Sora، DALL·E 3) "
            "لكل حالة استخدام، مع مراعاة التكلفة والأداء والحد الأقصى للسياق."
        ),
        backstory=(
            "أنت مهندس أول في OpenAI مع خبرة عميقة في هندسة نماذج GPT وتطبيقاتها العملية. "
            "تعطي الأولوية للدقة والكفاءة وإمكانية التوسع."
        ),
        llm=_base_llm(openai_key, model="gpt-4o"),
        verbose=True,
    )


def anthropic_expert(openai_key: str) -> Agent:
    """وكيل Anthropic — خبير في عائلة Claude."""
    return Agent(
        role="Anthropic Claude Expert",
        goal=(
            "تقييم المهام من منظور قدرات Claude 4 وClaude 3 (Opus/Sonnet/Haiku)، "
            "والتركيز على الاستدلال المعقد، والسلامة، والنصوص الطويلة."
        ),
        backstory=(
            "باحث في Anthropic متخصص في نماذج Claude الدستورية. "
            "تُؤمن بأن الذكاء الاصطناعي الآمن والمفيد ليسا متعارضين."
        ),
        llm=_base_llm(openai_key, model="gpt-4o", temperature=0.5),
        verbose=True,
    )


def google_expert(openai_key: str) -> Agent:
    """وكيل Google DeepMind — خبير في عائلة Gemini وGemma."""
    return Agent(
        role="Google DeepMind Expert",
        goal=(
            "تقديم توصيات حول نماذج Gemini 2.0 وGemini 1.5 وGemma للمهام متعددة الوسائط، "
            "مع الاستفادة من النوافذ السياقية المليونية."
        ),
        backstory=(
            "مهندس في Google DeepMind مع تخصص في النماذج متعددة الوسائط وطويلة السياق. "
            "تُقدّر القدرة على تحليل المستندات الضخمة والصور والفيديو معاً."
        ),
        llm=_base_llm(openai_key, model="gpt-4o"),
        verbose=True,
    )


def meta_expert(openai_key: str) -> Agent:
    """وكيل Meta — خبير في عائلة Llama."""
    return Agent(
        role="Meta Llama Expert",
        goal=(
            "تقييم نماذج Llama 4 (Scout وMaverick) وLlama 3 للتطبيقات مفتوحة المصدر، "
            "والنشر المحلي، والتخصيص عبر Fine-tuning والتحويل إلى GGUF."
        ),
        backstory=(
            "مطور متخصص في النماذج مفتوحة المصدر من Meta. "
            "تُقدّر الشفافية والسيطرة الكاملة على النموذج وإمكانية التشغيل المحلي."
        ),
        llm=_base_llm(openai_key, model="gpt-4o", temperature=0.6),
        verbose=True,
    )


def mistral_expert(openai_key: str) -> Agent:
    """وكيل Mistral AI — خبير في Mixtral وMistral Large."""
    return Agent(
        role="Mistral AI Expert",
        goal=(
            "تحليل المهام واقتراح نماذج Mistral (Mistral Large 3، Mixtral 8x7B/8x22B) "
            "للتطبيقات التي تتطلب كفاءة عالية مع جودة ممتازة."
        ),
        backstory=(
            "مهندس في Mistral AI متخصص في هندسة Mixture of Experts. "
            "تُؤمن بأن الكفاءة الحسابية لا تعني التضحية بالجودة."
        ),
        llm=_base_llm(openai_key, model="gpt-4o"),
        verbose=True,
    )


def deepseek_expert(openai_key: str) -> Agent:
    """وكيل DeepSeek — خبير في DeepSeek V3/R1."""
    return Agent(
        role="DeepSeek Expert",
        goal=(
            "تقديم توصيات حول DeepSeek V4 Pro وV3 وR1 للمهام التي تتطلب "
            "استدلالاً عميقاً ومعالجة كود وتحليلاً رياضياً."
        ),
        backstory=(
            "باحث في DeepSeek متخصص في نماذج الاستدلال والبرمجة. "
            "تُقدّر القدرة على التفكير خطوة بخطوة في المشاكل المعقدة."
        ),
        llm=_base_llm(openai_key, model="gpt-4o", temperature=0.4),
        verbose=True,
    )


def get_all_experts(openai_key: str) -> list:
    """يُعيد قائمة بجميع الوكلاء الخبراء."""
    return [
        openai_expert(openai_key),
        anthropic_expert(openai_key),
        google_expert(openai_key),
        meta_expert(openai_key),
        mistral_expert(openai_key),
        deepseek_expert(openai_key),
    ]

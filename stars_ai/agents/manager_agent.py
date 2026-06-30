"""
Manager Agent: وكيل المدير الذي ينسق بين الخبراء في النظام الهرمي.
يتلقى المهمة، يوزعها على الخبراء، ثم يجمع النتائج ويُقدم توصية نهائية.
"""

from crewai import Agent
from langchain_openai import ChatOpenAI


def create_manager_agent(openai_key: str) -> Agent:
    """
    وكيل المدير (Orchestrator) — ينسق جميع الوكلاء الخبراء.

    :param openai_key: مفتاح OpenAI API
    :return: وكيل المدير المُهيأ
    """
    llm = ChatOpenAI(
        api_key=openai_key,
        model="gpt-4o",
        temperature=0.3,
    )

    return Agent(
        role="AI Systems Architect & Manager",
        goal=(
            "تنسيق فريق من خبراء الذكاء الاصطناعي لتحليل المهام واختيار أفضل نموذج أو مجموعة نماذج. "
            "تجميع آراء الخبراء، تقييم المقايضات (التكلفة/الأداء/الكمون)، "
            "وتقديم توصية نهائية واضحة وقابلة للتنفيذ."
        ),
        backstory=(
            "أنت مهندس أنظمة ذكاء اصطناعي من الفئة الأولى مع خبرة 10+ سنوات في نشر "
            "نماذج اللغة الكبيرة على نطاق واسع. عملت مع جميع المزودين الكبار وتفهم "
            "نقاط قوة وضعف كل نموذج. قرارتك مبنية على البيانات والمنطق، لا العواطف. "
            "تُعطي الأولوية للحلول العملية القابلة للنشر الفوري."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

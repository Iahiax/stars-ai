## “””
main.py

نظام الوكيل الذكي المتعدد النماذج (Multi-Agent Swarm System).

يعتمد على CrewAI مع Process.hierarchical:

- Manager Agent  : يستقبل المهمة، يوزّعها على الخبراء، ويدمج آراءهم في
  قرار/تقرير نهائي موحد.
- Expert Agents  : كل وكيل يمثل شركة/مزوداً مختلفاً، ويُستدعى بمفتاحه
  الخاص الذي يُجلب بأمان عبر KeyManager من Secret Manager.
  “””

import os
import logging

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from key_manager import KeyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(“MultiAgentSwarm”)

# ——————————————————————

# 1. تهيئة مدير المفاتيح (يجلب المفاتيح من Google Secret Manager)

# ——————————————————————

key_manager = KeyManager(project_id=os.environ.get(“GCP_PROJECT_ID”))

def build_llm(provider: str, model_name: str, base_url: str = None) -> ChatOpenAI:
“””
دالة مساعدة لبناء كائن LLM لأي مزود، باستخدام مفتاحه الخاص الذي يُجلب
عبر KeyManager. تستخدم ChatOpenAI كواجهة موحّدة لأن العديد من المزودين
(Mistral, DeepSeek, …) يوفرون endpoint متوافقاً مع OpenAI API.
لاستخدام عميل مخصص فعلياً (مثل Anthropic أو Google) استبدل الاستدعاء
بالكلاس المناسب من مكتبة langchain الخاصة بذلك المزود
(مثال: from langchain_anthropic import ChatAnthropic).
“””
api_key = key_manager.get_key(provider)
kwargs = {“model”: model_name, “api_key”: api_key, “temperature”: 0.4}
if base_url:
kwargs[“base_url”] = base_url
return ChatOpenAI(**kwargs)

# ——————————————————————

# 2. تعريف الوكلاء (Agents)

# ——————————————————————

manager_agent = Agent(
role=“مدير تنسيق الوكلاء (Manager Agent)”,
goal=(
“تنسيق النقاش بين الخبراء المتخصصين، توزيع المهام الفرعية عليهم، “
“وتجميع إجاباتهم في قرار/تقرير نهائي موحد وعالي الجودة.”
),
backstory=(
“أنت مدير مشروع ذكاء اصطناعي خبير، تدير فريقاً من الخبراء يمثل كل “
“منهم شركة تقنية مختلفة، وتوجّه النقاش بينهم بحيادية لإنتاج أفضل “
“إجابة ممكنة لصاحب الطلب.”
),
llm=build_llm(“openai”, “gpt-4o”),
allow_delegation=True,
verbose=True,
)

openai_expert = Agent(
role=“خبير OpenAI”,
goal=“تقديم تحليل دقيق بالاستناد إلى نقاط قوة نماذج GPT في الاستدلال والتوليد البرمجي.”,
backstory=“خبير متخصص في نماذج GPT، يعرف جيداً متى يُفضَّل استخدامها عن غيرها.”,
llm=build_llm(“openai”, “gpt-4o”),
allow_delegation=False,
verbose=True,
)

anthropic_expert = Agent(
role=“خبير Anthropic”,
goal=“تقديم تحليل متوازن يعتمد على نقاط قوة عائلة Claude في الأمان والاستدلال على سياقات طويلة.”,
backstory=“خبير متخصص في نماذج Claude وصياغة الـ Prompt الآمن والمنظم.”,
llm=build_llm(“anthropic”, “claude-3-opus”),
allow_delegation=False,
verbose=True,
)

google_expert = Agent(
role=“خبير Google DeepMind”,
goal=“تقديم رؤية تقنية تستفيد من قدرات Gemini في تعدد الوسائط والسياق الطويل.”,
backstory=“خبير في نماذج Gemini وGemma، يفهم تكامل النص والصورة والفيديو.”,
llm=build_llm(“google”, “gemini-1.5-pro”),
allow_delegation=False,
verbose=True,
)

mistral_expert = Agent(
role=“خبير Mistral AI”,
goal=“تقديم حلول فعّالة من حيث التكلفة بالاستناد إلى نماذج Mistral/Mixtral.”,
backstory=“خبير في نماذج Mistral المفتوحة، يفضّل الحلول الخفيفة والسريعة.”,
llm=build_llm(“mistral”, “mistral-large-latest”, base_url=“https://api.mistral.ai/v1”),
allow_delegation=False,
verbose=True,
)

deepseek_expert = Agent(
role=“خبير DeepSeek”,
goal=“تقديم تحليل عميق يركّز على الاستدلال الرياضي والبرمجي بالاستناد إلى نماذج DeepSeek.”,
backstory=“خبير في نماذج DeepSeek R1/V3، يتميز بالتركيز على الاستدلال المنطقي الدقيق.”,
llm=build_llm(“deepseek”, “deepseek-reasoner”, base_url=“https://api.deepseek.com/v1”),
allow_delegation=False,
verbose=True,
)

# قائمة الخبراء (Expert Agents) — يمكن إضافة مزودين آخرين بنفس النمط

expert_agents = [
openai_expert,
anthropic_expert,
google_expert,
mistral_expert,
deepseek_expert,
]

# ——————————————————————

# 3. تعريف المهمة (Task)

# ——————————————————————

def build_task(topic: str) -> Task:
“”“بناء مهمة واحدة يوجّهها المدير لفريق الخبراء حول موضوع معيّن.”””
return Task(
description=(
f”حلّل الموضوع التالي من زوايا متعددة: ‘{topic}’.\n”
“كل خبير يقدّم رأيه المتخصص استناداً إلى نقاط قوة الشركة التي “
“يمثلها، وعلى المدير دمج كل الآراء في تقرير نهائي موحد وواضح “
“يتضمن توصية عملية.”
),
expected_output=“تقرير نهائي شامل يدمج آراء جميع الخبراء مع توصية واضحة ومبررة.”,
agent=manager_agent,
)

# ——————————————————————

# 4. تجميع الفريق (Crew) — باستخدام Process.hierarchical

# ——————————————————————

def run_swarm(topic: str) -> str:
“””
تشغيل النظام: المدير (manager_agent) يوزّع المهمة على الخبراء
(expert_agents) عبر Process.hierarchical، ثم يدمج نتائجهم.
“””
task = build_task(topic)
crew = Crew(
agents=expert_agents,
tasks=[task],
manager_agent=manager_agent,
process=Process.hierarchical,
verbose=True,
)
return crew.kickoff()

if **name** == “**main**”:
topic = os.environ.get(
“SWARM_TOPIC”,
“أفضل استراتيجية لإطلاق منتج ذكاء اصطناعي جديد يستهدف مجتمعات Wolf Live العربية”,
)
logger.info(f”بدء تشغيل النظام حول الموضوع: {topic}”)

```
result = run_swarm(topic)

print("\n" + "=" * 60)
print("النتيجة النهائية:")
print("=" * 60)
print(result)
```

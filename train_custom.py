# Stars AI — نظام الذكاء الاصطناعي المتكامل للبرمجة

مشروع Python متكامل يشمل: تدريب نماذج LLM، Fine-tuning بـ LoRA، تقطير المعرفة،
تحويل GGUF، محادثة تفاعلية، تقييم تلقائي، ورفع لـ HuggingFace Hub.

---

## هيكل المشروع الكامل

```
/  (جذر المستودع)
│
├── train_all.py           ★★  التدريب الشامل — 6 مراحل كاملة (W&B + --resume)
├── distill_model.py       ★★  تقطير المعرفة Teacher → Student
├── sync_to_hf.py          ★★  رفع تلقائي إلى HuggingFace Hub
├── benchmark.py           ★★  Benchmark احترافي — تصنيف نموذجك عالمياً
├── auto_improve.py        ★★  تحسين تلقائي — يكتشف الضعف ويعالجه وحده
├── test_suite.py          ★   اختبارات تلقائية شاملة لكل مكونات المشروع
├── chat.py                ★   محادثة تفاعلية (GGUF + HuggingFace + Ollama)
├── evaluate.py                تقييم تلقائي على 50 سؤال برمجي
├── generate_data.py           توليد بيانات التدريب تلقائياً بـ GPT-4
├── compare_models.py          مقارنة عدة نماذج على نفس السؤال
├── rag_code.py                مساعد يقرأ كودك ويجيب عنه (RAG)
├── finetune_lora.py           Fine-tuning بـ LoRA (بيانات البرمجة)
├── finetune_arabic.py         Fine-tuning للغة العربية
├── prune_model.py             ضغط النموذج وتصغيره
├── train_custom.py            تدريب StarsLM من الصفر على بياناتك
├── main.py                    استعراض النماذج + تدريب + تحويل + Swarm
├── requirements.txt
├── .env.example               قالب الإعدادات ومفاتيح API
│
└── stars_ai/
    ├── __init__.py
    ├── key_manager.py         إدارة مفاتيح API (GCP + .env)
    ├── model_registry.py      17 مزود، 70+ نموذج
    ├── model_builder.py       بناء StarsLM من الصفر
    ├── gguf_converter.py      تحويل إلى GGUF
    └── agents/
        ├── manager_agent.py
        └── expert_agents.py
```

`★★` = ميزة رئيسية | `★` = ميزة جديدة أو محدّثة

---

## التثبيت السريع

```bash
# استنساخ المشروع
git clone https://github.com/your-username/stars-ai
cd stars-ai

# تثبيت المكتبات
pip install -r requirements.txt

# إعداد متغيرات البيئة
cp .env.example .env
# ثم عدّل .env وأضف مفاتيحك
```

**المكتبات الاختيارية حسب الحاجة:**
```bash
pip install llama-cpp-python   # تشغيل GGUF على CPU (لـ chat.py --gguf)
pip install bitsandbytes       # تكميم 4-bit على GPU
pip install wandb              # تتبع التدريب (train_all.py --wandb)
pip install huggingface_hub    # رفع النماذج (sync_to_hf.py)
```

---

## المسار الكامل — من الصفر إلى نموذج منشور

```
1. train_all.py       ← تدريب النموذج (6 مراحل تلقائية)
        ↓
2. evaluate.py        ← قياس الأداء على 50 سؤال
        ↓
3. benchmark.py       ← مقارنة مع النماذج العالمية
        ↓
4. sync_to_hf.py      ← نشر على HuggingFace Hub
        ↓
5. chat.py            ← محادثة تفاعلية
```

---

## الملفات الرئيسية — دليل الاستخدام

---

### `train_all.py` — التدريب الشامل

يُنفّذ 6 مراحل متتالية بأمر واحد:

```
المرحلة 1: جمع 38,000+ مثال (CodeAlpaca + HumanEval + عربي)
المرحلة 2: Fine-tuning بـ LoRA على بيانات البرمجة
المرحلة 3: Fine-tuning إضافي للغة العربية
المرحلة 4: تقييم تلقائي على 50 سؤال
المرحلة 5: ضغط النموذج (Pruning)
المرحلة 6: تحويل إلى GGUF
```

```bash
# تشغيل كل المراحل
python train_all.py

# مع تتبع التدريب على Weights & Biases
python train_all.py --wandb

# استئناف تلقائي من آخر نقطة تفتيش عند الانقطاع
python train_all.py --resume

# مراحل محددة فقط
python train_all.py --stages 1,2,3

# نموذج مختلف
python train_all.py --model mistralai/Mistral-7B-v0.1

# إضافة بيانات محلية
python train_all.py --local-data ./data/my_data.jsonl

# تخصيص كامل
python train_all.py \
    --model   mistralai/Mistral-7B-v0.1 \
    --batch   8 \
    --epochs  5 \
    --wandb   \
    --resume
```

---

### `distill_model.py` — تقطير المعرفة ★★

يدرّب نموذجاً صغيراً ليتعلم من نموذج كبير (Teacher → Student).
النتيجة: نموذج صغير أذكى بكثير من تدريبه المباشر.

```
Teacher (كبير): Mistral-7B / GPT-4 / Llama-70B
      ↓  يوجّه التدريب
Student (صغير): Phi-2 / TinyLlama
```

```bash
# تقطير من Mistral-7B إلى Phi-2 (محلياً)
python distill_model.py \
    --teacher mistralai/Mistral-7B-v0.1 \
    --student microsoft/phi-2

# تقطير باستخدام GPT-4 عبر API
python distill_model.py \
    --teacher gpt-4o \
    --student microsoft/phi-2 \
    --use-api

# استئناف التقطير من حيث توقف
python distill_model.py \
    --teacher mistralai/Mistral-7B-v0.1 \
    --student microsoft/phi-2 \
    --resume

# مع مقارنة Teacher vs Student بعد التدريب
python distill_model.py \
    --teacher mistralai/Mistral-7B-v0.1 \
    --student microsoft/phi-2 \
    --compare

# بيانات إضافية خاصة
python distill_model.py \
    --teacher mistralai/Mistral-7B-v0.1 \
    --student microsoft/phi-2 \
    --data    ./data/my_prompts.txt \
    --epochs  5
```

---

### `sync_to_hf.py` — رفع إلى HuggingFace Hub ★★

يرفع النموذج بشكل تلقائي مع توليد Model Card احترافي.

```bash
# إعداد Token
export HF_TOKEN=hf_...   # من huggingface.co/settings/tokens

# رفع نموذج HuggingFace
python sync_to_hf.py \
    --model ./models/stars_expert_merged \
    --repo  your-username/stars-ai

# رفع ملف GGUF
python sync_to_hf.py \
    --gguf ./models/stars_expert.gguf \
    --repo your-username/stars-ai-gguf

# رفع كليهما معاً
python sync_to_hf.py \
    --model ./models/stars_expert_merged \
    --gguf  ./models/stars_expert.gguf  \
    --repo  your-username/stars-ai

# رفع خاص (private)
python sync_to_hf.py \
    --model   ./models/stars_expert_merged \
    --repo    your-username/stars-ai \
    --private
```

---

### `chat.py` — المحادثة التفاعلية ★

يدعم ثلاثة محركات للتشغيل:

```bash
# GGUF (أسرع — يعمل على CPU بدون GPU)
python chat.py --gguf ./models/stars_expert.gguf

# HuggingFace (تحميل مباشر)
python chat.py --model ./models/stars_expert_merged
python chat.py --model microsoft/phi-2

# Ollama (أسهل — بدون تحميل يدوي)
python chat.py --ollama llama3
python chat.py --ollama mistral
python chat.py --ollama phi3 --ollama-host http://192.168.1.5:11434

# تخصيص الإخراج
python chat.py --gguf ./models/stars_expert.gguf \
    --temp 0.7 --max-tokens 600
```

**أوامر داخل المحادثة:**
```
مسح / clear  ← مسح تاريخ المحادثة
حفظ / save   ← حفظ المحادثة في ملف JSON
خروج / quit  ← إنهاء البرنامج
مساعدة       ← عرض الأوامر
```

**إعداد Ollama:**
```bash
# تثبيت Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# تحميل نموذج
ollama pull llama3
ollama pull mistral
ollama pull phi3

# تشغيل الخادم (إن لم يكن يعمل)
ollama serve
```

---

### `test_suite.py` — الاختبارات التلقائية ★

يتحقق من صحة كل مكونات المشروع:

```bash
# كل الاختبارات
python test_suite.py

# اختبارات سريعة (بدون تحميل نماذج)
python test_suite.py --fast

# مجموعة محددة
python test_suite.py --module imports   # فحص المكتبات
python test_suite.py --module package   # فحص حزمة stars_ai
python test_suite.py --module model     # فحص StarsLM
python test_suite.py --module data      # فحص معالجة البيانات
python test_suite.py --module eval      # فحص نظام التقييم
python test_suite.py --module files     # فحص وجود الملفات
python test_suite.py --module env       # فحص متغيرات البيئة
```

**مثال على المخرجات:**
```
══════════════════════════════════════════════════════════════════
  Stars AI — Test Suite
══════════════════════════════════════════════════════════════════

[1] اختبار المكتبات الأساسية
  ✓  import PyTorch                                        (0.31ث)
  ✓  import HuggingFace Transformers                      (0.18ث)
  ✓  import PEFT / LoRA                                   (0.12ث)

[5] اختبار نظام التقييم
  ✓  Evaluator — 50 سؤال بالضبط                         (0.01ث)
  ✓  Evaluator — صيغة الأسئلة صحيحة                      (0.02ث)

══════════════════════════════════════════════════════════════════
  النتائج: 24/24 اختبار ناجح | 0 فاشل
══════════════════════════════════════════════════════════════════
  ✅ جميع الاختبارات ناجحة — المشروع جاهز!
```

---

### `evaluate.py` — التقييم التلقائي

```bash
# تقييم نموذجك
python evaluate.py --model ./models/stars_expert_merged
python evaluate.py --gguf  ./models/stars_expert.gguf

# مقارنة قبل وبعد التدريب
python evaluate.py \
    --before microsoft/phi-2 \
    --after  ./models/stars_expert_merged

# تقييم نموذج HuggingFace مباشرة
python evaluate.py --model microsoft/phi-2
```

**الفئات (50 سؤال):**

| الفئة | عدد الأسئلة |
|-------|-------------|
| Python أساسي | 10 |
| البرمجة كائنية التوجه | 10 |
| الخوارزميات وهياكل البيانات | 10 |
| Python متقدم | 10 |
| قواعد البيانات والـ API | 10 |

---

### `benchmark.py` — المقارنة العالمية

```bash
# تصنيف نموذجك مقارنةً بالنماذج العالمية
python benchmark.py --my-model ./models/stars_expert_merged

# مع ملف GGUF
python benchmark.py --my-model ./models/stars_expert.gguf --gguf

# حفظ النتائج
python benchmark.py --my-model ./models/stars_expert_merged \
    --output ./results/benchmark.json
```

---

### `generate_data.py` — توليد البيانات

```bash
export OPENAI_API_KEY=sk-...

python generate_data.py --count 500  --topic python
python generate_data.py --count 1000 --topic algorithms
python generate_data.py --count 300  --topic arabic_python
python generate_data.py --count 800  --topic mixed \
    --output ./data/mixed.jsonl
```

**المواضيع:** `python` | `algorithms` | `web` | `arabic_python` | `mixed`

---

### `distill_model.py` + `auto_improve.py` + `compare_models.py`

```bash
# التحسين التلقائي (يكتشف نقاط الضعف ويعالجها)
python auto_improve.py --model ./models/stars_expert_merged --rounds 3

# مقارنة نماذج جنباً إلى جنب
python compare_models.py --models phi-2 mistral llama3
python compare_models.py --gguf ./models/before.gguf ./models/after.gguf

# مساعد RAG يقرأ كودك
python rag_code.py --project ./my_project --gguf ./models/stars_expert.gguf
```

---

## إعداد البيئة

```bash
# انسخ ملف الإعدادات
cp .env.example .env
```

ثم عدّل `.env` وأضف مفاتيحك:

```env
# OpenAI (للتوليد والتقطير عبر API)
OPENAI_API_KEY=sk-...

# HuggingFace (للرفع + النماذج المحمية)
HF_TOKEN=hf_...

# Weights & Biases (تتبع التدريب)
WANDB_API_KEY=...
WANDB_PROJECT=stars-ai
```

---

## إعداد Google Cloud (اختياري)

```bash
gcloud iam service-accounts create stars-ai-sa
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:stars-ai-sa@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
gcloud iam service-accounts keys create key.json \
    --iam-account=stars-ai-sa@PROJECT_ID.iam.gserviceaccount.com
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/key.json"
echo -n "sk-..." | gcloud secrets create openai_api_key --data-file=-
```

---

## النشر على Ubuntu (Systemd)

```bash
sudo mkdir -p /opt/stars-ai
sudo cp -r ./* /opt/stars-ai/
pip install -r /opt/stars-ai/requirements.txt
```

`/etc/systemd/system/stars-ai.service`:
```ini
[Unit]
Description=Stars AI
After=network.target

[Service]
WorkingDirectory=/opt/stars-ai
EnvironmentFile=/opt/stars-ai/.env
ExecStart=/usr/bin/python3 chat.py --gguf ./models/stars_expert.gguf
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now stars-ai
sudo journalctl -u stars-ai -f
```

---

## ملاحظات تقنية

| الميزة | التفاصيل |
|--------|----------|
| معمارية StarsLM | Decoder-only + RoPE + RMSNorm + SwiGLU |
| LoRA | r=16, alpha=32 — يدرّب 0.1% من الأوزان فقط |
| تقطير المعرفة | Teacher→Student — KL Divergence على logits |
| RAG | TF-IDF بسيط — لا يحتاج embeddings أو GPU |
| Pruning | Magnitude + Structured + INT8 Dynamic |
| تنسيق GGUF | v3 — متوافق مع llama.cpp، Ollama، LM Studio |
| التكميم | f32 / f16 / Q8_0 / Q4_0 |
| التقييم | 50 سؤال، 5 فئات، معيار النجاح 50% |
| W&B | تتبع Loss + LR + GPU في الوقت الحقيقي |
| Ollama | HTTP API محلي — بدون تحميل يدوي للنماذج |

---

## متطلبات النظام

| الاستخدام | الحد الأدنى | الموصى به |
|-----------|-------------|-----------|
| محادثة GGUF (CPU) | 8 GB RAM | 16 GB RAM |
| Fine-tuning LoRA | GPU 8 GB | GPU 16+ GB |
| تقطير المعرفة | GPU 16 GB | GPU 24+ GB |
| توليد البيانات (API) | أي جهاز | أي جهاز |

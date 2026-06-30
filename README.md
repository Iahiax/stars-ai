# stars-ai
# Stars AI — نظام الوكيل الذكي المتعدد النماذج

مشروع Python متكامل يشمل كل ما تحتاجه للعمل مع نماذج الذكاء الاصطناعي.

---

## هيكل المشروع الكامل

```
python/
├── train_all.py           ← التدريب الشامل الكامل (كل المصادر + كل المراحل) ★★
├── benchmark.py           ← Benchmark احترافي — تصنيف نموذجك عالمياً  ★★
├── auto_improve.py        ← تحسين تلقائي — يكتشف الضعف ويعالجه وحده ★★
├── main.py                ← استعراض النماذج + تدريب + تحويل + Swarm
├── train_custom.py        ← تدريب StarsLM على بياناتك الخاصة خطوة بخطوة
├── finetune_lora.py       ← Fine-tuning بـ LoRA (برمجة - إنجليزي)
├── finetune_arabic.py     ← Fine-tuning للغة العربية  ★
├── generate_data.py       ← توليد بيانات التدريب تلقائياً بـ GPT-4  ★
├── compare_models.py      ← مقارنة عدة نماذج على نفس السؤال  ★
├── rag_code.py            ← مساعد يقرأ كودك ويجيب عنه (RAG)  ★
├── chat.py                ← محادثة تفاعلية من Terminal
├── evaluate.py            ← تقييم تلقائي على 50 سؤال برمجي
├── prune_model.py         ← ضغط النموذج وتصغيره  ★
├── requirements.txt
└── stars_ai/
    ├── key_manager.py     ← إدارة مفاتيح API (GCP + .env)
    ├── model_registry.py  ← 17 مزود، 70+ نموذج
    ├── model_builder.py   ← بناء StarsLM من الصفر
    ├── gguf_converter.py  ← تحويل إلى GGUF
    └── agents/
        ├── manager_agent.py
        └── expert_agents.py
```
★★ = الملفات الرئيسية | ★ = ميزة جديدة

---

## التثبيت

```bash
cd python
python3 -m venv venv
source venv/bin/activate

# المكتبات الأساسية
pip install -r requirements.txt

# إضافات اختيارية
pip install llama-cpp-python    # لتشغيل ملفات GGUF
pip install sentencepiece       # لـ BPE Tokenizer
pip install bitsandbytes        # تكميم 4-bit (GPU فقط)
```

---

## المسار الكامل — من الصفر إلى نموذج جاهز

### الأمر الواحد الذي يفعل كل شيء تلقائياً:
```bash
python train_all.py
```

هذا الأمر يُنفّذ تلقائياً 6 مراحل متتالية:

```
المرحلة 1: جمع 38,000+ مثال من 4 مصادر + بيانات عربية
     ↓
المرحلة 2: Fine-tuning بـ LoRA على كل البيانات
     ↓
المرحلة 3: Fine-tuning إضافي للغة العربية
     ↓
المرحلة 4: تقييم تلقائي على 50 سؤال (قبل/بعد)
     ↓
المرحلة 5: ضغط النموذج (Pruning 30%)
     ↓
المرحلة 6: تحويل إلى GGUF للاستخدام المحلي
```

---

### أو يدوياً خطوة بخطوة:

```
generate_data.py → finetune_lora.py → finetune_arabic.py → evaluate.py → prune_model.py → main.py convert → chat.py
```

---

## الملفات الجديدة — الدليل السريع

---

### 1. توليد بيانات تلقائياً (`generate_data.py`)

يستخدم GPT-4 لإنشاء آلاف الأمثلة البرمجية تلقائياً.

```bash
# توليد 500 مثال Python
python generate_data.py --count 500 --topic python

# توليد 1000 مثال للخوارزميات
python generate_data.py --count 1000 --topic algorithms

# توليد بيانات عربية
python generate_data.py --count 300 --topic arabic_python

# توليد مواضيع مختلطة
python generate_data.py --count 800 --topic mixed --output ./data/mixed.jsonl
```

**المواضيع المتاحة:** `python` | `algorithms` | `web` | `arabic_python` | `mixed`

بعد التوليد، استخدم الملف مباشرة في التدريب:
```python
# في finetune_lora.py:
"dataset_source": "local_file",
"local_file": "./data/data_python.jsonl",
```

---

### 2. مقارنة نماذج متعددة (`compare_models.py`)

يسأل عدة نماذج نفس السؤال في نفس الوقت ويعرضها جنباً إلى جنب.

```bash
# مقارنة نموذجين بالأسماء المختصرة
python compare_models.py --models phi-2 mistral

# مقارنة ثلاثة نماذج
python compare_models.py --models phi-2 mistral llama3

# مقارنة ملفات GGUF
python compare_models.py --gguf ./models/before.gguf ./models/after.gguf

# سؤال واحد مباشر
python compare_models.py --models phi-2 mistral \
    --question "اكتب دالة Binary Search"

# وضع تفاعلي مع حفظ النتائج
python compare_models.py --models phi-2 mistral --save results.jsonl
```

**الأسماء المختصرة:** `phi-2` | `phi-3` | `mistral` | `llama3` | `llama2` | `deepseek` | `gemma` | `qwen` | `dolphin`

**المخرجات:**
```
  ┌─ phi-2 (2.3ث | 85 كلمة | يحتوي كود) ✓
  │  def binary_search(arr, target):
  │      left, right = 0, len(arr) - 1
  │      ...
  └────────────────────────────────────────

  ┌─ mistral (4.1ث | 120 كلمة | يحتوي كود) ✓
  │  ...
  └────────────────────────────────────────

  الأسرع: phi-2 (2.3ث)
```

---

### 3. مساعد يقرأ كودك ويجيب عنه (`rag_code.py`)

يفهرس مشروعك البرمجي كاملاً ثم يجيب على أسئلتك بناءً على كودك أنت.

```bash
# فهرسة مشروعك والمحادثة عنه
python rag_code.py --project ./my_project --model microsoft/phi-2

# مع ملف GGUF (أسرع)
python rag_code.py --project ./my_project --gguf ./models/code_expert.gguf

# سؤال واحد مباشر
python rag_code.py --project . --model microsoft/phi-2 \
    --question "ما الذي تفعله دالة process_data؟"

# ضبط حجم القطع
python rag_code.py --project . --model microsoft/phi-2 --chunk 30
```

**يدعم:** `.py .js .ts .java .cpp .go .rs .sql .md` وغيرها

**كيف يعمل:**
```
1. يقرأ جميع ملفات مشروعك
2. يقسّمها إلى قطع صغيرة
3. عند سؤالك → يبحث عن أقرب القطع لسؤالك
4. يعطيها للنموذج مع السؤال → النموذج يجيب بناءً عليها
```

---

### 4. Fine-tuning للغة العربية (`finetune_arabic.py`)

يدرّب النموذج على الإجابة بالعربي على أسئلة البرمجة.

```bash
# بيانات عربية تجريبية مدمجة (8 أمثلة × 25)
python finetune_arabic.py

# بياناتك العربية الخاصة
python finetune_arabic.py --data ./data/arabic_coding.jsonl

# تخصيص إعدادات التدريب
python finetune_arabic.py \
    --base-model microsoft/phi-2 \
    --data ./data/arabic.jsonl \
    --epochs 5 \
    --output ./models/arabic_expert
```

**صيغة البيانات العربية:**
```json
{"instruction": "اكتب دالة تجمع رقمين", "output": "def جمع(أ, ب):\n    return أ + ب"}
{"instruction": "ما هو الـ decorator؟", "output": "شرح بالعربي مع مثال..."}
```

---

### 5. ضغط النموذج (`prune_model.py`)

يحذف الأوزان غير المهمة لتصغير النموذج قبل التحويل إلى GGUF.

```bash
# ضغط 30% (آمن — جودة ممتازة)
python prune_model.py --model ./models/code_expert --ratio 0.3

# ضغط 50% مع قياس الأداء قبل وبعد
python prune_model.py --model ./models/code_expert --ratio 0.5 --benchmark

# ضغط هيكلي (أسرع في التنفيذ)
python prune_model.py --model ./models/code_expert --structured --ratio 0.25

# ضغط + تكميم INT8 (أقصى تصغير)
python prune_model.py --model ./models/code_expert --ratio 0.3 --quantize

# ثم حوّله إلى GGUF
python main.py convert --model-dir ./models/code_expert_pruned --quant q4_0
```

**مثال على نتيجة الضغط:**
```
╔════════════════════════════════════════════════════════╗
║                    تقرير الضغط                        ║
╠════════════════════════════════════════════════════════╣
║  المقياس              قبل          بعد                ║
║  المعاملات           2700.0M      1890.0M              ║
║  نسبة الإزالة          0.0%        30.0%              ║
║  وقت الاستدلال         4.2ث         2.9ث              ║
║  Tokens/ثانية         11.9         17.2               ║
║  تسريع الاستدلال      1.45x                           ║
╚════════════════════════════════════════════════════════╝
```

**نصيحة لاختيار النسبة:**
| النسبة | الجودة | الحجم | الاستخدام |
|--------|--------|-------|-----------|
| 20-30% | ممتازة | -25%  | موصى به |
| 40-50% | جيدة   | -40%  | للأجهزة المحدودة |
| 60-70% | مقبولة | -55%  | للأجهزة الضعيفة جداً |

---

## الملفات القديمة — مرجع سريع

### استعراض النماذج
```bash
python main.py registry --gguf-only
python main.py registry --provider meta
python main.py registry --search llama
```

### التدريب من الصفر
```bash
python train_custom.py                          # بيانات تجريبية
python train_custom.py  # غيّر data_file في CONFIG
```

### Fine-tuning (برمجة)
```bash
python finetune_lora.py                         # بيانات تلقائية
```

### التقييم
```bash
python evaluate.py --model ./models/code_expert
python evaluate.py --before microsoft/phi-2 --after ./models/code_expert
```

### المحادثة
```bash
python chat.py --model ./models/code_expert
python chat.py --gguf ./models/code_expert.gguf
```

### التحويل إلى GGUF
```bash
# تثبيت llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j4 && cd ..

python main.py convert \
    --model-dir ./models/code_expert_merged \
    --output    ./models/code_expert.gguf \
    --quant     q4_0 \
    --llama-cpp ./llama.cpp
```

### Multi-Agent Swarm
```bash
export OPENAI_API_KEY=sk-...
python main.py swarm --task "قارن أفضل 3 نماذج للبرمجة العربية"
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
sudo cp -r ./python/* /opt/stars-ai/
python3 -m venv /opt/stars-ai/venv
/opt/stars-ai/venv/bin/pip install -r /opt/stars-ai/requirements.txt
```

`/etc/systemd/system/stars-ai.service`:
```ini
[Unit]
Description=Stars AI
After=network.target

[Service]
WorkingDirectory=/opt/stars-ai
Environment="OPENAI_API_KEY=sk-..."
ExecStart=/opt/stars-ai/venv/bin/python chat.py --model ./models/code_expert
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
| RAG | TF-IDF بسيط — لا يحتاج embeddings أو GPU |
| Pruning | Magnitude + Structured + INT8 Dynamic |
| تنسيق GGUF | v3 — متوافق مع llama.cpp، Ollama، LM Studio |
| التكميم | f32 / f16 / Q8_0 / Q4_0 |
| التقييم | 50 سؤال، 5 فئات، معيار النجاح 50% |
| توليد البيانات | GPT-4o — 5 مواضيع — batch كل 10 أمثلة |

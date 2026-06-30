# stars-ai
# Multi-Agent Swarm System

نظام وكيل ذكي متعدد النماذج (Multi-Agent Swarm) مبني على CrewAI، بحيث يدير
“مدير” (Manager Agent) نقاشاً بين “خبراء” (Expert Agents) يمثّل كل منهم
شركة/مزوداً مختلفاً من مزودي نماذج الذكاء الاصطناعي، عبر `Process.hierarchical`.
مفاتيح API تُجلب بأمان من **Google Secret Manager** ولا تُخزَّن أبداً داخل الكود.

## هيكل الملفات

```
multi-agent-swarm/
├── key_manager.py              # جلب المفاتيح من Secret Manager
├── model_registry.py           # سجل النماذج ومزوديها
├── main.py                     # تعريف الوكلاء والـ Crew وتشغيل النظام
├── requirements.txt            # المكتبات المطلوبة
├── deploy.sh                   # سكربت النشر على Ubuntu
├── multi-agent-swarm.service   # ملف خدمة systemd
└── README.md
```

## 1. إعداد Google Cloud (مرة واحدة فقط)

### أ. تفعيل واجهة Secret Manager

```bash
gcloud services enable secretmanager.googleapis.com --project=YOUR_PROJECT_ID
```

### ب. تخزين المفاتيح كأسرار (Secrets)

كرّر هذا الأمر لكل مفتاح (الاسم يجب أن يطابق القيم في `SECRET_MAP` داخل `key_manager.py`):

```bash
echo -n "sk-xxxxxxxx" | gcloud secrets create OPENAI_API_KEY \
  --data-file=- --project=YOUR_PROJECT_ID

echo -n "sk-ant-xxxxxxxx" | gcloud secrets create ANTHROPIC_API_KEY \
  --data-file=- --project=YOUR_PROJECT_ID
```

### ج. إنشاء Service Account وإعطاؤه صلاحية `Secret Manager Secret Accessor`

```bash
# 1. إنشاء الحساب
gcloud iam service-accounts create swarm-runner \
  --display-name="Multi-Agent Swarm Runner" \
  --project=YOUR_PROJECT_ID

# 2. ربط الصلاحية المطلوبة فقط (مبدأ أقل الصلاحيات)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:swarm-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# 3. إصدار مفتاح JSON لاستخدامه على خادم Ubuntu (خارج بيئة GCP)
gcloud iam service-accounts keys create service-account.json \
  --iam-account=swarm-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

> إن كان الخادم نفسه VM على GCP (Compute Engine)، يمكنك تجاهل الخطوة 3
> وربط الـ Service Account مباشرة بالـ VM بدلاً من نسخ ملف JSON.

## 2. نشر المشروع على خادم Ubuntu

```bash
# من جهازك المحلي: نقل الملفات
scp -r ./multi-agent-swarm user@SERVER_IP:/tmp/

# على الخادم: نقل المشروع لمكانه النهائي وتشغيل سكربت النشر
ssh user@SERVER_IP
sudo mv /tmp/multi-agent-swarm /opt/multi-agent-swarm
cd /opt/multi-agent-swarm
chmod +x deploy.sh
./deploy.sh
```

سكربت `deploy.sh` يقوم بـ:

1. إنشاء `venv` (بيئة افتراضية معزولة).
1. تثبيت المكتبات من `requirements.txt`.
1. إعادة المحاولة تلقائياً (حتى 5 مرات) عند فشل التثبيت بسبب ضعف الشبكة
   (`ReadTimeoutError`)، مع فاصل 10 ثوانٍ بين كل محاولة.

ضع ملف `service-account.json` (إن استخدمته) داخل `/opt/multi-agent-swarm/`.

## 3. ضبط متغيرات البيئة

```bash
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/opt/multi-agent-swarm/service-account.json"
```

## 4. تشغيل النظام كخدمة دائمة (systemd)

```bash
# 1. عدّل القيم __YOUR_LINUX_USER__ و __YOUR_GCP_PROJECT_ID__ في الملف
sudo nano /opt/multi-agent-swarm/multi-agent-swarm.service

# 2. نسخ ملف الخدمة لمكانه الصحيح
sudo cp /opt/multi-agent-swarm/multi-agent-swarm.service /etc/systemd/system/

# 3. إعادة تحميل systemd لقراءة الخدمة الجديدة
sudo systemctl daemon-reload

# 4. تفعيل الخدمة لتعمل تلقائياً عند إعادة تشغيل الخادم
sudo systemctl enable multi-agent-swarm.service

# 5. تشغيل الخدمة
sudo systemctl start multi-agent-swarm.service

# 6. التحقق من الحالة
sudo systemctl status multi-agent-swarm.service

# 7. متابعة السجلات (logs) مباشرة
journalctl -u multi-agent-swarm.service -f
```

بهذا يعمل النظام في الخلفية بشكل دائم، ويُعاد تشغيله تلقائياً عند أي تعطل
(`Restart=on-failure`) أو عند إعادة تشغيل الخادم بالكامل.

## 5. تشغيل تجريبي مباشر (بدون systemd)

```bash
source /opt/multi-agent-swarm/venv/bin/activate
export SWARM_TOPIC="موضوعك هنا"
python main.py
```

## ملاحظات أمنية

- لا تضع أي مفتاح API داخل الكود مباشرة؛ كل المفاتيح تُجلب حصرياً من
  Secret Manager عبر `KeyManager`.
- الصلاحية الممنوحة للـ Service Account هي `secretmanager.secretAccessor`
  فقط (قراءة الأسرار)، وليست صلاحية إدارة كاملة.
- في حال تسريب `service-account.json`، يجب إبطال المفتاح فوراً:
  `gcloud iam service-accounts keys delete KEY_ID --iam-account=...`

#!/usr/bin/env bash

# deploy.sh

# سكربت نشر نظام الوكيل الذكي المتعدد النماذج على خادم Ubuntu.

# يقوم بإنشاء بيئة افتراضية (venv) وتثبيت المكتبات مع إعادة محاولة تلقائية

# عند فشل التثبيت بسبب ضعف الشبكة (ReadTimeoutError).

set -euo pipefail

PROJECT_DIR=”/opt/multi-agent-swarm”
VENV_DIR=”$PROJECT_DIR/venv”
PYTHON_BIN=“python3”
MAX_RETRIES=5
RETRY_DELAY=10

echo “>>> [1/4] إنشاء مجلد المشروع: $PROJECT_DIR”
sudo mkdir -p “$PROJECT_DIR”
sudo chown “$(whoami)”:”$(whoami)” “$PROJECT_DIR”

echo “>>> [2/4] ملاحظة: نقل الملفات إلى الخادم”
echo “    نفّذ هذا الأمر من جهازك المحلي (وليس من الخادم) قبل تشغيل باقي السكربت:”
echo “    scp -r ./multi-agent-swarm/* user@SERVER_IP:$PROJECT_DIR/”

cd “$PROJECT_DIR”

echo “>>> [3/4] إنشاء البيئة الافتراضية (venv)”
if [ ! -d “$VENV_DIR” ]; then
$PYTHON_BIN -m venv “$VENV_DIR”
fi

# shellcheck disable=SC1091

source “$VENV_DIR/bin/activate”

echo “>>> [4/4] تثبيت المكتبات (مع إعادة محاولة تلقائية عند مشاكل الشبكة)”
pip install –upgrade pip –timeout 120

attempt=1
while [ “$attempt” -le “$MAX_RETRIES” ]; do
if pip install -r requirements.txt –timeout 120 –retries 5; then
echo “تم تثبيت جميع المكتبات بنجاح.”
break
else
echo “فشلت المحاولة $attempt من $MAX_RETRIES (ربما بسبب ReadTimeoutError).”
if [ “$attempt” -eq “$MAX_RETRIES” ]; then
echo “تعذّر تثبيت المكتبات بعد $MAX_RETRIES محاولات. تأكد من استقرار الإنترنت ثم أعد المحاولة.”
exit 1
fi
echo “إعادة المحاولة بعد $RETRY_DELAY ثواني…”
sleep “$RETRY_DELAY”
attempt=$((attempt + 1))
fi
done

echo “”
echo “============================================================”
echo “المشروع جاهز في: $PROJECT_DIR”
echo “لتفعيل البيئة الافتراضية يدوياً: source $VENV_DIR/bin/activate”
echo “الخطوة التالية: إعداد خدمة systemd (راجع README.md)”
echo “============================================================”

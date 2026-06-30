#!/usr/bin/env bash

# install-gcloud.sh

# تثبيت Google Cloud CLI على Ubuntu عبر مستودع APT الرسمي من Google.

set -euo pipefail

echo “>>> [1/4] تثبيت المتطلبات الأساسية”
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl

echo “>>> [2/4] إضافة مفتاح Google ومستودع APT الرسمي”
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg |   
sudo gpg –dearmor -o /usr/share/keyrings/cloud.google.gpg

echo “deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main” |   
sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

echo “>>> [3/4] تثبيت الحزمة”
sudo apt-get update
sudo apt-get install -y google-cloud-cli

echo “>>> [4/4] التحقق من التثبيت”
gcloud version

echo “”
echo “تم التثبيت بنجاح. الخطوة التالية: تسجيل الدخول عبر:”
echo “    gcloud init”

## “””
key_manager.py

وحدة إدارة المفاتيح (KeyManager) — مسؤولة عن جلب مفاتيح API الخاصة بمختلف
مزودي نماذج الذكاء الاصطناعي من Google Secret Manager بطريقة آمنة، مع
تخزين مؤقت (cache) داخل الذاكرة لتقليل عدد طلبات الشبكة، ونسخة احتياطية
من متغيرات البيئة في حال عدم توافر السر في Secret Manager.
“””

import os
import logging
from functools import lru_cache

from google.cloud import secretmanager
from google.api_core.exceptions import NotFound, PermissionDenied, GoogleAPICallError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(“KeyManager”)

class KeyManager:
“””
كلاس مسؤول عن جلب مفاتيح API لمزودي النماذج المختلفين من Google Secret
Manager. كل مزود مرتبط باسم سر (secret_id) محدد مسبقاً في SECRET_MAP،
ويمكن توسيع هذه الخريطة بسهولة لإضافة مزودين جدد دون تعديل منطق الكلاس.
“””

```
# خريطة: اسم المزود الداخلي -> اسم السر في Secret Manager
SECRET_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "meta": "META_API_KEY",
    "microsoft": "AZURE_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "alibaba": "ALIBABA_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "ibm": "IBM_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "runway": "RUNWAY_API_KEY",
    "stability": "STABILITY_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
}

def __init__(self, project_id: str = None):
    """
    :param project_id: رقم مشروع GCP. إن لم يُمرر، يُقرأ من متغير البيئة
                        GCP_PROJECT_ID.
    """
    self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
    if not self.project_id:
        raise ValueError(
            "يجب تحديد GCP_PROJECT_ID عبر متغير البيئة أو عند إنشاء الكلاس"
        )

    try:
        # العميل يعتمد تلقائياً على Application Default Credentials (ADC)
        # سواء عبر GOOGLE_APPLICATION_CREDENTIALS أو Service Account المرفق بالـ VM
        self.client = secretmanager.SecretManagerServiceClient()
    except Exception as exc:
        logger.error(f"فشل تهيئة عميل Secret Manager: {exc}")
        raise

@lru_cache(maxsize=64)
def _fetch_secret(self, secret_id: str, version: str = "latest") -> str:
    """
    جلب قيمة سر معين من Secret Manager. النتيجة تُخزَّن مؤقتاً (lru_cache)
    لتجنّب استدعاء الشبكة المتكرر لنفس السر خلال عمر العملية.
    """
    name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
    try:
        response = self.client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8").strip()
    except NotFound:
        logger.warning(
            f"السر '{secret_id}' غير موجود في Secret Manager، "
            f"سيتم البحث في متغيرات البيئة كنسخة احتياطية."
        )
        return os.environ.get(secret_id, "")
    except PermissionDenied:
        logger.error(f"صلاحيات غير كافية للوصول إلى السر '{secret_id}'.")
        raise
    except GoogleAPICallError as exc:
        logger.error(f"خطأ في الاتصال بـ Secret Manager عند جلب '{secret_id}': {exc}")
        raise

def get_key(self, provider: str) -> str:
    """
    إرجاع مفتاح API لمزود معين (مثل: openai, anthropic, google ...).
    يرفع ValueError إذا كان اسم المزود غير مسجل في SECRET_MAP.
    """
    provider = provider.lower().strip()
    secret_id = self.SECRET_MAP.get(provider)
    if not secret_id:
        raise ValueError(f"المزود '{provider}' غير مسجل في SECRET_MAP")

    key = self._fetch_secret(secret_id)
    if not key:
        logger.warning(f"تم إرجاع مفتاح فارغ للمزود '{provider}'.")
    return key

def get_all_keys(self) -> dict:
    """إرجاع جميع المفاتيح المسجلة دفعة واحدة (مفيد لفحص الصلاحية الأولي)."""
    return {provider: self.get_key(provider) for provider in self.SECRET_MAP}

def clear_cache(self):
    """تفريغ التخزين المؤقت — مفيد بعد تدوير (rotate) أحد المفاتيح."""
    self._fetch_secret.cache_clear()
    logger.info("تم تفريغ التخزين المؤقت لجميع المفاتيح.")
```

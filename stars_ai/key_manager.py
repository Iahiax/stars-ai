"""
Stars AI — KeyManager
إدارة مفاتيح API عبر Google Secret Manager أو متغيرات البيئة.
"""

import os
from typing import Optional


class KeyManager:
    """
    يتعامل مع جلب مفاتيح API من:
      1. Google Cloud Secret Manager  (بيئة الإنتاج)
      2. متغيرات البيئة / ملف .env   (بيئة التطوير)
    """

    PROVIDERS = [
        "openai", "anthropic", "google", "meta", "microsoft",
        "xai", "mistral", "deepseek", "moonshot", "alibaba",
        "zhipu", "ibm", "elevenlabs", "midjourney", "stability",
        "runway", "pika",
    ]

    def __init__(self, project_id: Optional[str] = None, use_gcp: bool = False):
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.use_gcp    = use_gcp
        self._cache:    dict[str, str] = {}
        self._client    = None

        # تحميل .env تلقائياً إذا وُجد
        self._load_dotenv()

        if self.use_gcp:
            self._init_gcp_client()

    def _load_dotenv(self):
        """يحمّل متغيرات البيئة من ملف .env إذا وُجد."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    def _init_gcp_client(self):
        """يهيئ عميل Google Secret Manager."""
        try:
            from google.cloud import secretmanager
            self._client = secretmanager.SecretManagerServiceClient()
        except ImportError:
            raise ImportError(
                "pip install google-cloud-secret-manager"
            )

    def get_key(self, provider: str, version: str = "latest") -> str:
        """
        يجلب مفتاح API للمزود المحدد.

        :param provider: openai | anthropic | google | mistral | ...
        :raises ValueError: إذا لم يُعثر على المفتاح
        """
        cache_key = f"{provider}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        api_key = (
            self._fetch_from_gcp(provider, version) if self.use_gcp
            else self._fetch_from_env(provider)
        )

        if not api_key:
            env_var = self._env_var_name(provider)
            raise ValueError(
                f"لم يُعثر على مفتاح API للمزود '{provider}'.\n"
                f"الحل: export {env_var}=<مفتاحك>\n"
                f"أو أضفه في ملف .env"
            )

        self._cache[cache_key] = api_key
        return api_key

    def get_key_safe(self, provider: str) -> Optional[str]:
        """يجلب المفتاح بدون رمي استثناء — يُعيد None إذا لم يُعثر عليه."""
        try:
            return self.get_key(provider)
        except ValueError:
            return None

    def _fetch_from_gcp(self, provider: str, version: str) -> Optional[str]:
        if not self.project_id:
            raise ValueError("project_id مطلوب عند استخدام GCP.")
        name = f"projects/{self.project_id}/secrets/{provider}_api_key/versions/{version}"
        try:
            resp = self._client.access_secret_version(request={"name": name})
            return resp.payload.data.decode("utf-8").strip()
        except Exception as e:
            raise RuntimeError(f"فشل جلب السر '{name}': {e}") from e

    def _fetch_from_env(self, provider: str) -> Optional[str]:
        return os.getenv(self._env_var_name(provider))

    @staticmethod
    def _env_var_name(provider: str) -> str:
        return f"{provider.upper().replace('-', '_')}_API_KEY"

    def clear_cache(self):
        self._cache.clear()

    def list_available_providers(self) -> list[str]:
        """يُعيد قائمة المزودين الذين تم ضبط مفاتيحهم."""
        return [p for p in self.PROVIDERS if os.getenv(self._env_var_name(p))]

    def show_status(self):
        """يطبع حالة كل المفاتيح."""
        print("  حالة مفاتيح API:")
        for p in self.PROVIDERS:
            key = os.getenv(self._env_var_name(p))
            status = "✓ موجود" if key else "✗ مفقود"
            print(f"    {p:15s}: {status}")

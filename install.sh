#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Stars AI — سكريبت التثبيت التلقائي
#  يعمل على Replit (NixOS) وLinux وmacOS
#
#  الاستخدام:
#    bash install.sh           # تثبيت المكتبات الأساسية
#    bash install.sh --full    # تثبيت كل المكتبات (بما فيها GGUF وCrewAI)
#    bash install.sh --check   # فحص ما هو مثبّت فقط
# ══════════════════════════════════════════════════════════════

set -e

# ── الألوان ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $1${NC}"; }
err()  { echo -e "${RED}  ✗ $1${NC}"; }

FULL_INSTALL=false
CHECK_ONLY=false

for arg in "$@"; do
    case $arg in
        --full)  FULL_INSTALL=true ;;
        --check) CHECK_ONLY=true ;;
    esac
done

echo "══════════════════════════════════════════════════════"
echo "  Stars AI — التثبيت"
echo "══════════════════════════════════════════════════════"

# ── فحص Python ───────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=$(command -v python3)
    ok "Python: $($PYTHON --version)"
else
    err "Python غير موجود!"
    exit 1
fi

# ── وضع الفحص فقط ────────────────────────────────────────────
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "  الحالة الحالية:"
    for pkg in torch transformers datasets peft accelerate \
               huggingface_hub wandb openai anthropic sentencepiece \
               numpy tqdm requests aiohttp python_dotenv; do
        if $PYTHON -c "import $pkg" 2>/dev/null; then
            ver=$($PYTHON -c "import $pkg; print(getattr($pkg, '__version__', 'موجود'))" 2>/dev/null)
            ok "$pkg ($ver)"
        else
            warn "$pkg — غير مثبّت"
        fi
    done
    exit 0
fi

# ── اكتشاف طريقة التثبيت ─────────────────────────────────────
echo ""
echo "  اكتشاف بيئة التثبيت..."

# Replit — استخدام uv مع venv
if command -v uv &>/dev/null; then
    ok "تم اكتشاف uv"
    INSTALL_METHOD="uv"

    # إنشاء venv إن لم تكن موجودة
    if [ ! -d ".venv" ]; then
        echo "  إنشاء بيئة افتراضية..."
        uv venv .venv 2>/dev/null
        ok "تم إنشاء .venv"
    else
        ok ".venv موجودة"
    fi

    PYTHON=".venv/bin/python"
    INSTALL_CMD="uv pip install --python .venv/bin/python"

# pip عادي (Linux/macOS)
elif $PYTHON -m pip --version &>/dev/null 2>&1; then
    warn "uv غير متاح — استخدام pip"
    INSTALL_METHOD="pip"
    INSTALL_CMD="$PYTHON -m pip install"

else
    err "لا يوجد pip ولا uv!"
    echo "  ثبّت uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── المكتبات الأساسية ─────────────────────────────────────────
echo ""
echo "  [1/4] تثبيت مكتبات HuggingFace..."
$INSTALL_CMD \
    torch \
    transformers \
    datasets \
    peft \
    accelerate \
    huggingface-hub \
    tokenizers \
    sentencepiece \
    && ok "HuggingFace Stack" || warn "فشل بعض المكتبات"

echo ""
echo "  [2/4] تثبيت مكتبات APIs..."
$INSTALL_CMD \
    openai \
    anthropic \
    aiohttp \
    requests \
    && ok "APIs" || warn "فشل بعض المكتبات"

echo ""
echo "  [3/4] تثبيت أدوات التدريب..."
$INSTALL_CMD \
    wandb \
    python-dotenv \
    tqdm \
    numpy \
    && ok "أدوات التدريب" || warn "فشل بعض المكتبات"

# ── المكتبات الاختيارية ───────────────────────────────────────
if [ "$FULL_INSTALL" = true ]; then
    echo ""
    echo "  [4/4] تثبيت المكتبات الاختيارية..."

    echo "    llama-cpp-python (GGUF)..."
    $INSTALL_CMD llama-cpp-python \
        && ok "llama-cpp-python" \
        || warn "فشل تثبيت llama-cpp-python — يحتاج مترجم C++"

    echo "    bitsandbytes (تكميم 4-bit)..."
    $INSTALL_CMD bitsandbytes \
        && ok "bitsandbytes" \
        || warn "فشل تثبيت bitsandbytes — GPU فقط"

    echo "    CrewAI + LangChain..."
    $INSTALL_CMD crewai langchain langchain-openai langchain-anthropic \
        && ok "CrewAI + LangChain" \
        || warn "فشل تثبيت CrewAI/LangChain"

    echo "    Google Cloud..."
    $INSTALL_CMD google-cloud-secret-manager \
        && ok "Google Cloud" \
        || warn "فشل تثبيت Google Cloud"
else
    ok "[4/4] تخطّي المكتبات الاختيارية (شغّل --full لتثبيتها)"
fi

# ── التحقق النهائي ────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════"
echo "  التحقق من التثبيت:"

FAILED=0
for pkg in torch transformers datasets peft huggingface_hub wandb openai numpy tqdm; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        ver=$($PYTHON -c "import $pkg; print(getattr($pkg, '__version__', 'OK'))" 2>/dev/null)
        ok "$pkg ($ver)"
    else
        err "$pkg — فشل التثبيت"
        FAILED=$((FAILED + 1))
    fi
done

echo "══════════════════════════════════════════════════════"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  ✅ كل المكتبات جاهزة!${NC}"
    echo ""
    echo "  الخطوة التالية:"
    echo "    cp .env.example .env   # ثم أضف مفاتيحك"
    echo "    python train_all.py    # بدء التدريب"
    echo "    python chat.py --ollama llama3"
else
    echo -e "${YELLOW}  ⚠ $FAILED مكتبة فشلت — راجع الأخطاء أعلاه${NC}"
fi

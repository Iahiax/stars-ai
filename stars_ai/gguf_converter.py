"""
GGUF Converter: تحويل نماذج HuggingFace / StarsLM إلى صيغة GGUF.

الصيغة GGUF (GGML Unified Format) تُستخدم لتشغيل النماذج محلياً
عبر llama.cpp و Ollama وأدوات مشابهة.

طريقتان للتحويل:
  1. GGUFWriter مضمّن (للنماذج المبنية بـ StarsLM)
  2. llama.cpp convert_hf_to_gguf.py (لنماذج HuggingFace الخارجية)
"""

import os
import struct
import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

import torch


# ─── ثوابت GGUF ──────────────────────────────────────────────────────────────

GGUF_MAGIC   = 0x46554747  # "GGUF" بالـ little-endian
GGUF_VERSION = 3

GGUF_TYPE = {
    "uint8":   0,
    "int8":    1,
    "uint16":  2,
    "int16":   3,
    "uint32":  4,
    "int32":   5,
    "float32": 6,
    "bool":    7,
    "string":  8,
    "array":   9,
    "uint64":  10,
    "int64":   11,
    "float64": 12,
}

QuantType = Literal["f32", "f16", "q8_0", "q4_0", "q4_1"]


# ─── كاتب GGUF الأساسي ────────────────────────────────────────────────────────

class GGUFWriter:
    """
    يكتب ملف GGUF يدوياً من tensors PyTorch.
    يدعم الأوزان كاملة (f32 / f16) والكمية الأساسية (q8_0 / q4_0).
    """

    def __init__(self, output_path: str, arch: str = "starslm"):
        self.path    = output_path
        self.arch    = arch
        self._meta: list[tuple] = []
        self._tensors: list[tuple] = []

    # ── إضافة البيانات الوصفية ────────────────────────────────────────

    def add_string(self, key: str, value: str):
        self._meta.append(("string", key, value))

    def add_uint32(self, key: str, value: int):
        self._meta.append(("uint32", key, value))

    def add_float32(self, key: str, value: float):
        self._meta.append(("float32", key, value))

    def add_bool(self, key: str, value: bool):
        self._meta.append(("bool", key, value))

    # ── إضافة الأوزان ─────────────────────────────────────────────────

    def add_tensor(self, name: str, tensor: torch.Tensor, quant: QuantType = "f32"):
        self._tensors.append((name, tensor.detach().cpu().float(), quant))

    # ── الكتابة النهائية ──────────────────────────────────────────────

    def write(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "wb") as f:
            self._write_header(f)
            self._write_metadata(f)
            self._write_tensor_info(f)
            self._align(f, 32)
            self._write_tensor_data(f)
        size_mb = os.path.getsize(self.path) / 1024 / 1024
        print(f"[GGUFWriter] تم الكتابة: {self.path} ({size_mb:.1f} MB)")

    def _write_header(self, f):
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(self._tensors)))
        f.write(struct.pack("<Q", len(self._meta)))

    def _write_str(self, f, s: str):
        enc = s.encode("utf-8")
        f.write(struct.pack("<Q", len(enc)))
        f.write(enc)

    def _write_metadata(self, f):
        for dtype, key, value in self._meta:
            self._write_str(f, key)
            if dtype == "string":
                f.write(struct.pack("<I", GGUF_TYPE["string"]))
                self._write_str(f, value)
            elif dtype == "uint32":
                f.write(struct.pack("<I", GGUF_TYPE["uint32"]))
                f.write(struct.pack("<I", value))
            elif dtype == "float32":
                f.write(struct.pack("<I", GGUF_TYPE["float32"]))
                f.write(struct.pack("<f", value))
            elif dtype == "bool":
                f.write(struct.pack("<I", GGUF_TYPE["bool"]))
                f.write(struct.pack("<B", int(value)))

    def _write_tensor_info(self, f):
        self._tensor_offsets = []
        current_offset = 0
        for name, tensor, quant in self._tensors:
            self._write_str(f, name)
            ndim = tensor.ndim
            f.write(struct.pack("<I", ndim))
            for dim in tensor.shape:
                f.write(struct.pack("<Q", dim))
            ggml_type = self._quant_to_ggml(quant)
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", current_offset))
            self._tensor_offsets.append(current_offset)
            nbytes = self._tensor_nbytes(tensor, quant)
            current_offset += self._pad(nbytes, 32)

    def _write_tensor_data(self, f):
        for (name, tensor, quant), _ in zip(self._tensors, self._tensor_offsets):
            data = self._quantize(tensor, quant)
            f.write(data)
            # محاذاة 32 بايت
            pad = (-len(data)) % 32
            f.write(b"\x00" * pad)

    def _quantize(self, tensor: torch.Tensor, quant: QuantType) -> bytes:
        flat = tensor.flatten()
        if quant == "f32":
            return flat.numpy().astype("float32").tobytes()
        elif quant == "f16":
            return flat.numpy().astype("float16").tobytes()
        elif quant == "q8_0":
            return self._quant_q8_0(flat)
        elif quant == "q4_0":
            return self._quant_q4_0(flat)
        else:
            return flat.numpy().astype("float32").tobytes()

    @staticmethod
    def _quant_q8_0(flat: torch.Tensor) -> bytes:
        """تكميم Q8_0: 32 عنصر لكل كتلة، عامل تدريج واحد (float16)."""
        block_size = 32
        n = flat.numel()
        # ضمان القابلية للقسمة
        pad = (-n) % block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        blocks = flat.reshape(-1, block_size)
        result = bytearray()
        for blk in blocks:
            max_val = blk.abs().max().item()
            scale = max_val / 127.0 if max_val != 0 else 1.0
            quantized = (blk / scale).round().clamp(-128, 127).to(torch.int8)
            # اكتب scale كـ float16 ثم الأوزان
            result += struct.pack("<e", scale)
            result += quantized.numpy().tobytes()
        return bytes(result)

    @staticmethod
    def _quant_q4_0(flat: torch.Tensor) -> bytes:
        """تكميم Q4_0: 32 عنصر لكل كتلة، 4 بت لكل عنصر."""
        block_size = 32
        n = flat.numel()
        pad = (-n) % block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad)])
        blocks = flat.reshape(-1, block_size)
        result = bytearray()
        for blk in blocks:
            max_val = blk.abs().max().item()
            scale = max_val / 7.0 if max_val != 0 else 1.0
            quantized = (blk / scale).round().clamp(-8, 7).to(torch.int8) + 8
            result += struct.pack("<e", scale)
            # حزم كل عنصرين في بايت واحد
            for i in range(0, block_size, 2):
                lo = int(quantized[i].item()) & 0x0F
                hi = int(quantized[i + 1].item()) & 0x0F
                result.append(lo | (hi << 4))
        return bytes(result)

    @staticmethod
    def _quant_to_ggml(quant: QuantType) -> int:
        mapping = {"f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q8_0": 8}
        return mapping.get(quant, 0)

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor, quant: QuantType) -> int:
        n = tensor.numel()
        block_size = 32
        n_padded = n + ((-n) % block_size)
        n_blocks = n_padded // block_size
        if quant == "f32":
            return n * 4
        elif quant == "f16":
            return n * 2
        elif quant == "q8_0":
            return n_blocks * (2 + block_size)
        elif quant == "q4_0":
            return n_blocks * (2 + block_size // 2)
        return n * 4

    @staticmethod
    def _pad(n: int, align: int) -> int:
        return n + ((-n) % align)

    @staticmethod
    def _align(f, align: int):
        pos = f.tell()
        pad = (-pos) % align
        f.write(b"\x00" * pad)


# ─── محوّل نماذج StarsLM ─────────────────────────────────────────────────────

class StarsLMToGGUF:
    """
    يحوّل نموذج StarsLM المحفوظ إلى ملف GGUF.
    """

    def __init__(self, model_dir: str, output_path: str, quant: QuantType = "q8_0"):
        self.model_dir   = model_dir
        self.output_path = output_path
        self.quant       = quant

    def convert(self):
        from stars_ai.model_builder import StarsLM, StarsConfig

        print(f"[StarsLMToGGUF] تحميل النموذج من: {self.model_dir}")
        model = StarsLM.from_pretrained(self.model_dir)
        cfg   = model.cfg

        writer = GGUFWriter(self.output_path, arch="starslm")

        # ── البيانات الوصفية ──────────────────────────────────────────
        writer.add_string("general.architecture",        "starslm")
        writer.add_string("general.name",               "StarsLM")
        writer.add_string("general.quantization_version", self.quant)
        writer.add_uint32("starslm.context_length",     cfg.max_seq_len)
        writer.add_uint32("starslm.embedding_length",   cfg.hidden_size)
        writer.add_uint32("starslm.feed_forward_length",cfg.ffn_hidden)
        writer.add_uint32("starslm.block_count",        cfg.num_layers)
        writer.add_uint32("starslm.attention.head_count", cfg.num_heads)
        writer.add_uint32("tokenizer.ggml.model",       cfg.vocab_size)

        # ── الأوزان ───────────────────────────────────────────────────
        state = model.state_dict()
        for name, tensor in state.items():
            # اختر التكميم المناسب حسب نوع الطبقة
            q = "f32" if "norm" in name else self.quant
            gguf_name = self._map_name(name)
            print(f"  تحويل: {name:50s} → {gguf_name} ({q})")
            writer.add_tensor(gguf_name, tensor, quant=q)

        writer.write()
        print(f"\n[تم] ملف GGUF جاهز: {self.output_path}")
        print(f"للتشغيل المحلي: llama-cli -m {self.output_path} -p 'مرحباً' -n 100")

    @staticmethod
    def _map_name(name: str) -> str:
        """يحوّل أسماء الأوزان إلى الصيغة المعيارية GGUF."""
        name = name.replace("layers.", "blk.")
        name = name.replace(".attn.qkv.", ".attn_qkv.")
        name = name.replace(".attn.out.", ".attn_output.")
        name = name.replace(".ffn.gate.",  ".ffn_gate.")
        name = name.replace(".ffn.up.",    ".ffn_up.")
        name = name.replace(".ffn.down.",  ".ffn_down.")
        name = name.replace(".norm1.",     ".attn_norm.")
        name = name.replace(".norm2.",     ".ffn_norm.")
        name = name.replace("embed.",      "token_embd.")
        name = name.replace("norm.",       "output_norm.")
        name = name.replace("lm_head.",    "output.")
        return name


# ─── محوّل نماذج HuggingFace (عبر llama.cpp) ─────────────────────────────────

class HuggingFaceToGGUF:
    """
    يحوّل أي نموذج HuggingFace إلى GGUF باستخدام سكربت llama.cpp.

    المتطلبات:
      - تثبيت llama.cpp: git clone https://github.com/ggerganov/llama.cpp
      - Python: pip install transformers sentencepiece
    """

    def __init__(
        self,
        hf_model_id: str,
        output_dir: str,
        llama_cpp_dir: str = "./llama.cpp",
        quant: QuantType = "q4_0",
    ):
        self.hf_model_id  = hf_model_id
        self.output_dir   = output_dir
        self.llama_cpp_dir = llama_cpp_dir
        self.quant        = quant

    def convert(self, hf_cache_dir: Optional[str] = None):
        """تنزيل النموذج وتحويله."""
        local_dir = self._download(hf_cache_dir)
        gguf_path = self._run_convert_script(local_dir)
        self._quantize(gguf_path)
        return gguf_path

    def _download(self, cache_dir: Optional[str]) -> str:
        """تنزيل النموذج من HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("pip install huggingface_hub")

        print(f"[HF→GGUF] تنزيل: {self.hf_model_id}")
        local_dir = snapshot_download(
            self.hf_model_id,
            cache_dir=cache_dir,
            ignore_patterns=["*.msgpack", "flax_model*"],
        )
        print(f"[HF→GGUF] تم التنزيل: {local_dir}")
        return local_dir

    def _run_convert_script(self, model_dir: str) -> str:
        """يُشغّل سكربت التحويل من llama.cpp."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(
            self.output_dir,
            self.hf_model_id.replace("/", "-") + "-f16.gguf"
        )
        convert_script = os.path.join(self.llama_cpp_dir, "convert_hf_to_gguf.py")
        if not os.path.exists(convert_script):
            raise FileNotFoundError(
                f"سكربت التحويل غير موجود: {convert_script}\n"
                "نفّذ: git clone https://github.com/ggerganov/llama.cpp"
            )
        cmd = [
            "python3", convert_script,
            model_dir,
            "--outfile", output_file,
            "--outtype", "f16",
        ]
        print(f"[HF→GGUF] تحويل: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return output_file

    def _quantize(self, gguf_f16_path: str) -> str:
        """تكميم الملف الناتج باستخدام llama-quantize."""
        quantize_bin = os.path.join(self.llama_cpp_dir, "llama-quantize")
        if not os.path.exists(quantize_bin):
            print(f"[HF→GGUF] تحذير: llama-quantize غير موجود، سيتم الاحتفاظ بـ f16.")
            return gguf_f16_path

        quantized_path = gguf_f16_path.replace("-f16.gguf", f"-{self.quant}.gguf")
        cmd = [quantize_bin, gguf_f16_path, quantized_path, self.quant.upper()]
        print(f"[HF→GGUF] تكميم {self.quant}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[تم] ملف GGUF مُكمَّم: {quantized_path}")
        return quantized_path

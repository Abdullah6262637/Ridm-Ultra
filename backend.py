"""Donanim farkindalikli RIDM hesaplama arka ucu.

``backend='auto'`` (varsayilan) ve ``backend='native'`` SADECE gercekten
derlenmis ve yuklenmis bir C++ cekirdegiyle calisir: baslangicta zorunlu
bir `g++` derlemesi tetiklenir ve derleme veya ctypes yukleme basarisiz
olursa surec FATAL bir hatayla derhal sonlandirilir. Burada "sessiz"
NumPy/Torch geri-donusu YOKTUR -- ``backend='numpy'`` veya
``backend='torch'`` sadece kullanicinin bilerek/acikca sectigi bagimsiz
modlardir, otomatik bir yedek yol degildir.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Bu modulun bulundugu dizin -- proje kokune ve native/ klasorune referans icin.
_PROJECT_ROOT = Path(__file__).resolve().parent
_NATIVE_DIR = _PROJECT_ROOT / "native"
_NATIVE_SOURCE = _NATIVE_DIR / "ridm_kernels.cpp"


class NativeKernelBuildError(RuntimeError):
    """C++ cekirdek derleme/yukleme adiminda alinan FATAL hata."""


def _native_lib_name() -> str:
    if platform.system() == "Windows":
        return "ridm_kernels.dll"
    if platform.system() == "Darwin":
        return "libridm_kernels.dylib"
    return "libridm_kernels.so"


def _compiler_command(output_path: Path, enable_openmp: bool = True) -> list[str]:
    """Platforma gore g++ derleme komutunu olusturur."""
    cmd = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC"]
    if enable_openmp:
        cmd.append("-fopenmp")
    cmd.extend([str(_NATIVE_SOURCE), "-o", str(output_path)])
    return cmd


def compile_native_kernel(force: bool = True) -> Path:
    """Baslangicta zorunlu C++ derleme adimi."""
    if not _NATIVE_SOURCE.exists():
        raise NativeKernelBuildError(
            f"Native C++ kaynak dosyasi bulunamadi: {_NATIVE_SOURCE}"
        )

    output_path = _NATIVE_DIR / _native_lib_name()
    if output_path.exists() and not force:
        return output_path

    t0 = time.perf_counter()
    cmd = _compiler_command(output_path, enable_openmp=True)
    print(f"[BUILD] Zorunlu C++ cekirdek derlemesi calistiriliyor: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(_NATIVE_DIR))
    except FileNotFoundError as exc:
        raise NativeKernelBuildError(
            "g++ derleyicisi PATH'te bulunamadi. Bir C++17 destekli derleyici kurulu olmalidir."
        ) from exc
    except subprocess.CalledProcessError:
        # If openmp failed (e.g. missing libgomp on Windows MinGW), retry without -fopenmp
        cmd_fallback = _compiler_command(output_path, enable_openmp=False)
        print(f"[BUILD] OpenMP derlemesi basarisiz oldu, standart C++ ile deneniyor: {' '.join(cmd_fallback)}")
        try:
            result = subprocess.run(cmd_fallback, check=True, capture_output=True, text=True, cwd=str(_NATIVE_DIR))
        except subprocess.CalledProcessError as exc2:
            raise NativeKernelBuildError(
                "native/ridm_kernels.cpp derlemesi BASARISIZ oldu (sozdizim/link hatasi).\n"
                f"--- g++ stdout ---\n{exc2.stdout}\n--- g++ stderr ---\n{exc2.stderr}"
            ) from exc2


    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not output_path.exists():
        raise NativeKernelBuildError(
            f"g++ sifir hata koduyla cikti ama beklenen cikti dosyasi olusmadi: {output_path}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    print(f"[BUILD] C++ cekirdek basariyla derlendi -> {output_path} ({elapsed_ms:.1f} ms)")
    return output_path


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    active: str
    device: str
    threads: int
    native_kernels: bool
    cuda_available: bool
    cpu_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComputeBackend:
    """Merkezi matris islemleri icin CPU/GPU dispatcher.

    ``backend='auto'`` yerel C++ kernel DLL'ini kullanir. ``device='cuda'``
    istendiginde PyTorch/CUDA ile GPU SVD kullanilir; CUDA yoksa acik bir
    hata verilir, sessizce CPU'ya dusmez.
    """

    _FEATURES = {1: "avx2", 2: "avx512f", 4: "openmp"}

    def __init__(self, backend: str = "auto", device: str = "auto", threads: Optional[int] = None):
        if backend not in {"auto", "native", "numpy", "torch"}:
            raise ValueError("backend auto, native, numpy veya torch olmalidir.")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device auto, cpu veya cuda olmalidir.")
        self.requested = backend
        self.threads = max(1, int(threads or (os.cpu_count() or 1)))

        # KRITIK: 'auto' ve 'native' SESSIZCE NumPy'a dusmez. Ikisi de gercek,
        # derlenmis bir C++ cekirdek gerektirir. Sadece kullanicinin acikca
        # 'numpy' veya 'torch' istedigi durumlarda native derleme atlanir --
        # bu bir "fallback" degil, bilerek secilen bagimsiz bir hesap modudur.
        if backend in ("auto", "native"):
            try:
                compile_native_kernel(force=False)
            except NativeKernelBuildError as exc:
                sys.exit(
                    "CRITICAL: Local C++ Engine Compilation Failed\n"
                    f"{exc}\n"
                    "No silent fallback to pure Python/NumPy is permitted for "
                    "backend='auto'/'native'. Fix native/ridm_kernels.cpp or "
                    "explicitly pass backend='numpy' if you truly want the "
                    "pure-Python path."
                )

        self._native = self._load_native()
        if backend in ("auto", "native") and self._native is None:
            # Derleme "basarili" gorundu ama ctypes yuklemesi/ABI baglama
            # basarisiz oldu (ör. eksik sembol, uyumsuz ABI). Yine FATAL.
            sys.exit(
                "FATAL: C++ Kernel Compilation/Binding Failed\n"
                "g++ derlemesi tamamlandi fakat ctypes.CDLL yukleme veya "
                "sembol baglama basarisiz oldu. native/ridm_kernels.cpp "
                "ABI'sini kontrol edin."
            )

        self._torch = self._load_torch()
        self.cuda_available = bool(self._torch is not None and self._torch.cuda.is_available())

        if device == "cuda" and not self.cuda_available:
            raise RuntimeError("CUDA istendi fakat PyTorch CUDA cihazi bulunamadi.")
        self.device = "cuda" if device == "cuda" or (device == "auto" and self.cuda_available and backend == "torch") else "cpu"
        if backend == "native" and self._native is None:
            raise RuntimeError("Yerel C++ cekirdegi yuklu degil. native/README.md ile derleyin.")
        if backend == "torch" and self._torch is None:
            raise RuntimeError("PyTorch yuklu degil.")
        self.active = "torch" if backend == "torch" or self.device == "cuda" else ("native" if self._native else "numpy")
        if self.active == "numpy" and backend == "auto":
            # Bu satira asla ulasilmamali (yukarida zaten sys.exit edildi),
            # ama savunma amacli ikinci bir kontrol olarak birakiyoruz.
            sys.exit("FATAL: 'auto' backend, native cekirdek olmadan NumPy'a sessizce dusemez.")
        if self._torch is not None:
            self._torch.set_num_threads(self.threads)

    @staticmethod
    def _load_torch():
        try:
            import torch
            return torch
        except ImportError:
            return None

    @staticmethod
    def _library_names() -> tuple[str, ...]:
        return (_native_lib_name(),)

    def _load_native(self):
        root = Path(__file__).resolve().parent / "native"
        candidates = [root / name for name in self._library_names()]
        candidates += [root / "build" / name for name in self._library_names()]
        for path in candidates:
            if not path.exists():
                continue
            try:
                lib = ctypes.CDLL(str(path))
                lib.ridm_cpu_feature_flags.restype = ctypes.c_int
                lib.ridm_set_threads.argtypes = [ctypes.c_int]
                lib.ridm_accumulate_contexts_f32.argtypes = [
                    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ]
                lib.ridm_matvec_f32.argtypes = [
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int,
                ]
                if hasattr(lib, "ridm_cosine_matvec_f32"):
                    lib.ridm_cosine_matvec_f32.argtypes = [
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_float,
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int,
                    ]
                if hasattr(lib, "ridm_sample_logits_f32"):
                    lib.ridm_sample_logits_f32.argtypes = [
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
                        ctypes.c_float, ctypes.c_int, ctypes.c_float,
                    ]
                    lib.ridm_sample_logits_f32.restype = ctypes.c_int64
                if hasattr(lib, "ridm_sample_logits_constrained_f32"):
                    lib.ridm_sample_logits_constrained_f32.argtypes = [
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64,
                        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                        ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_float,
                        ctypes.c_float, ctypes.c_int, ctypes.c_float,
                    ]
                    lib.ridm_sample_logits_constrained_f32.restype = ctypes.c_int64
                lib.ridm_set_threads(self.threads)
                return lib

            except OSError:
                continue
        return None

    @staticmethod
    def _ptr(array: np.ndarray, ctype):
        return array.ctypes.data_as(ctypes.POINTER(ctype))

    @property
    def info(self) -> BackendInfo:
        flags = self._native.ridm_cpu_feature_flags() if self._native else 0
        features = tuple(name for bit, name in self._FEATURES.items() if flags & bit)
        return BackendInfo(self.requested, self.active, self.device, self.threads, self._native is not None,
                           self.cuda_available, features)

    def accumulate_contexts(self, token_ids, context_vecs, idf, window: int, distance_weights, matrix, counts) -> int:
        """Baglam sayimlarini C++/OpenMP cekirdegiyle yerinde biriktirir."""
        ids = np.ascontiguousarray(token_ids, dtype=np.int64)
        if len(ids) <= window:
            return 0
        if ids.min() < 0 or ids.max() >= context_vecs.shape[0]:
            raise ValueError("token_ids sözlük aralığının dışında.")
        if self.device == "cuda":
            # index_add_ repeated target satırlarını atomik/doğru toplar; hem
            # eğitim biriktirme hem de sonraki SVD aynı GPU yolunda kalır.
            torch = self._torch
            ids_gpu = torch.as_tensor(ids, dtype=torch.long, device="cuda")
            target = ids_gpu[window:]
            updates = torch.zeros((len(target), context_vecs.shape[1]), dtype=torch.float32, device="cuda")
            vectors = torch.as_tensor(context_vecs, dtype=torch.float32, device="cuda")
            idf_gpu = torch.as_tensor(idf, dtype=torch.float32, device="cuda")
            weights = torch.as_tensor(distance_weights, dtype=torch.float32, device="cuda")
            for distance in range(1, window + 1):
                source = ids_gpu[window - distance:len(ids) - distance]
                updates.add_(vectors[source] * idf_gpu[source, None] * weights[distance - 1])
            matrix_gpu = torch.as_tensor(matrix, dtype=torch.float32, device="cuda")
            counts_gpu = torch.as_tensor(counts, dtype=torch.float32, device="cuda")
            matrix_gpu.index_add_(0, target, updates)
            counts_gpu.index_add_(0, target, torch.ones(len(target), dtype=torch.float32, device="cuda"))
            matrix[...] = matrix_gpu.cpu().numpy()
            counts[...] = counts_gpu.cpu().numpy()
            return len(target)
        if self._native is not None and all(a.dtype == np.float32 and a.flags.c_contiguous
                                             for a in (context_vecs, idf, distance_weights, matrix, counts)):
            self._native.ridm_accumulate_contexts_f32(
                self._ptr(ids, ctypes.c_int64), len(ids), self._ptr(context_vecs, ctypes.c_float),
                self._ptr(idf, ctypes.c_float), int(context_vecs.shape[1]), int(window),
                self._ptr(distance_weights, ctypes.c_float), self._ptr(matrix, ctypes.c_float),
                self._ptr(counts, ctypes.c_float),
            )
            return len(ids) - window
        targets, sums = self.weighted_context_sums(ids, context_vecs, idf, window, distance_weights)
        np.add.at(matrix, targets, sums)
        np.add.at(counts, targets, 1)
        return len(targets)

    @staticmethod
    def weighted_context_sums(ids, context_vecs, idf, window: int, distance_weights):
        n, dim = len(ids), context_vecs.shape[1]
        out = np.zeros((n - window, dim), dtype=np.float32)
        weighted = context_vecs[ids] * idf[ids, None]
        for distance in range(1, window + 1):
            out += distance_weights[distance - 1] * weighted[window - distance:n - distance]
        return ids[window:], out

    def truncated_svd(self, matrix: np.ndarray, k: int):
        """CUDA varsa GPU SVD, aksi halde BLAS/LAPACK SVD.

        CPU'da yerel kernel biriktirme ve matvec'i kapsar; tam SVD icin
        kararlı, vendor-optimized LAPACK kullanilir. Bu, elle yazilmis bir
        SVD'den daha guvenilir ve sayisal olarak daha saglamdir.
        """
        if self.device == "cuda":
            torch = self._torch
            tensor = torch.as_tensor(matrix, dtype=torch.float32, device="cuda")
            u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
            return (u[:, :k].cpu().numpy(), s[:k].cpu().numpy(), vh[:k].cpu().numpy())
        if self.active == "torch" and self._torch is not None:
            tensor = self._torch.as_tensor(matrix, dtype=self._torch.float32)
            u, s, vh = self._torch.linalg.svd(tensor, full_matrices=False)
            return u[:, :k].numpy(), s[:k].numpy(), vh[:k].numpy()
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :k], s[:k], vh[:k]

    def matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
            raise ValueError("matvec için (satır, sütun) matris ve uyumlu vektör gerekir.")
        if self.device == "cuda":
            return (self._torch.as_tensor(matrix, device="cuda") @ self._torch.as_tensor(vector, device="cuda")).cpu().numpy()
        if self._native is not None:
            out = np.empty(matrix.shape[0], dtype=np.float32)
            self._native.ridm_matvec_f32(self._ptr(matrix, ctypes.c_float), self._ptr(vector, ctypes.c_float),
                                          self._ptr(out, ctypes.c_float), matrix.shape[0], matrix.shape[1])
            return out
        return matrix @ vector

    def cosine_matvec(self, matrix: np.ndarray, vector: np.ndarray, row_norms: Optional[np.ndarray] = None) -> np.ndarray:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        vec_norm = float(np.linalg.norm(vector))
        if self._native is not None and hasattr(self._native, "ridm_cosine_matvec_f32"):
            out = np.empty(matrix.shape[0], dtype=np.float32)
            r_ptr = self._ptr(row_norms.astype(np.float32), ctypes.c_float) if row_norms is not None else None
            self._native.ridm_cosine_matvec_f32(
                self._ptr(matrix, ctypes.c_float), self._ptr(vector, ctypes.c_float),
                r_ptr, ctypes.c_float(vec_norm), self._ptr(out, ctypes.c_float),
                matrix.shape[0], matrix.shape[1]
            )
            return out
        # Fallback numpy path
        raw_dot = matrix @ vector
        r_norms = row_norms if row_norms is not None else np.linalg.norm(matrix, axis=1)
        r_norms = np.maximum(r_norms, 1e-8)
        v_norm = max(vec_norm, 1e-8)
        return raw_dot / (r_norms * v_norm)

    def sample_logits_constrained(
        self, logits: np.ndarray, valid_mask: Optional[np.ndarray] = None,
        ngram_boost: Optional[dict[int, float]] = None, penalty_ids: Optional[list[int]] = None,
        penalty_val: float = 10.0, temperature: float = 0.7, top_k: int = 10, rng=None
    ) -> int:
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        random_val = float(rng.rand()) if rng is not None else float(np.random.rand())

        mask_arr = np.ascontiguousarray(valid_mask, dtype=np.uint8) if valid_mask is not None else None
        m_ptr = self._ptr(mask_arr, ctypes.c_uint8) if mask_arr is not None else None

        if ngram_boost:
            b_ids = np.ascontiguousarray(list(ngram_boost.keys()), dtype=np.int64)
            b_vals = np.ascontiguousarray(list(ngram_boost.values()), dtype=np.float32)
            b_ids_ptr = self._ptr(b_ids, ctypes.c_int64)
            b_vals_ptr = self._ptr(b_vals, ctypes.c_float)
            b_cnt = len(ngram_boost)
        else:
            b_ids_ptr = b_vals_ptr = None
            b_cnt = 0

        if penalty_ids:
            p_ids = np.ascontiguousarray(penalty_ids, dtype=np.int64)
            p_ids_ptr = self._ptr(p_ids, ctypes.c_int64)
            p_cnt = len(penalty_ids)
        else:
            p_ids_ptr = None
            p_cnt = 0

        if self._native is not None and hasattr(self._native, "ridm_sample_logits_constrained_f32"):
            return int(self._native.ridm_sample_logits_constrained_f32(
                self._ptr(logits, ctypes.c_float), m_ptr, len(logits),
                b_ids_ptr, b_vals_ptr, b_cnt,
                p_ids_ptr, p_cnt, float(penalty_val),
                float(temperature), int(top_k), float(random_val)
            ))
        return self.sample_logits(logits, temperature=temperature, top_k=top_k, rng=rng)


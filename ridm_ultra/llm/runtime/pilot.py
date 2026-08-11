"""GPU pilot eğitiminden önce zorunlu doğrulama ve koşum manifesti."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from ..data.streaming import JSONLDocumentStream, PackedCausalDataset
from ..data.tokenizer import HuggingFaceTokenizer
from ..model.config import estimate_parameter_count
from ..model.presets import model_preset
from .benchmark import run_smoke_test


@dataclass(frozen=True)
class PilotSpec:
    preset: str = "turkish-50m"
    sequence_length: int = 1024
    batch_size: int = 1
    grad_accum_steps: int = 16
    max_steps: int = 2_000
    min_vram_gib: int = 8
    smoke_steps: int = 2


@dataclass
class PilotPreflightReport:
    ready: bool
    preset: str
    parameters: int | None = None
    device: str | None = None
    vram_gib: float | None = None
    data_documents: int | None = None
    manifest_sha256: str | None = None
    issues: list[str] = field(default_factory=list)
    smoke: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_pilot_preflight(data_files: Sequence[str | Path], tokenizer_path: str | Path, manifest_path: str | Path,
                        output_dir: str | Path, spec: PilotSpec = PilotSpec(), *, run_smoke: bool = False) -> PilotPreflightReport:
    """Pilot eğitimi için veri sözleşmesi ve donanım sözleşmesini doğrular.

    Büyük veri dosyasını yeniden hash'lemez; `prepare-data` ile üretilen
    manifestin içerdiği hash ve artifact yolunu doğrular. Böylece preflight
    saniyeler sürer, 1B tokenlik korpusu tekrar taramaz.
    """
    report = PilotPreflightReport(ready=False, preset=spec.preset)
    files = [Path(path) for path in data_files]
    manifest_file, tokenizer_file = Path(manifest_path), Path(tokenizer_path)
    if not files or any(not path.is_file() for path in files):
        report.issues.append("Eğitim shard'larından en az biri yok.")
    if not manifest_file.is_file():
        report.issues.append("Veri manifesti bulunamadı.")
    if not tokenizer_file.is_file():
        report.issues.append("Tokenizer artifact'i bulunamadı.")
    manifest = None
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            report.manifest_sha256 = manifest.get("sha256")
            if manifest.get("format_version") != 1 or not report.manifest_sha256:
                report.issues.append("Manifest formatı veya çıktı hash'i geçersiz.")
        except (OSError, json.JSONDecodeError) as exc:
            report.issues.append(f"Manifest okunamadı: {exc}")
    try:
        import torch
    except ImportError:
        report.issues.append("PyTorch kurulu değil: `pip install .[llm]` gerekli.")
        torch = None
    if torch is not None:
        if not torch.cuda.is_available():
            report.issues.append("CUDA GPU bulunamadı; pilot eğitim CPU'da başlatılmayacak.")
        else:
            properties = torch.cuda.get_device_properties(0)
            report.device = properties.name
            report.vram_gib = properties.total_memory / 1024 ** 3
            if report.vram_gib < spec.min_vram_gib:
                report.issues.append(f"VRAM yetersiz: {report.vram_gib:.1f} GiB < {spec.min_vram_gib} GiB.")
    tokenizer = None
    if tokenizer_file.is_file():
        try:
            tokenizer = HuggingFaceTokenizer.from_file(tokenizer_file)
            model_config = model_preset(spec.preset, tokenizer.vocab_size, spec.sequence_length, gradient_checkpointing=True)
            report.parameters = estimate_parameter_count(model_config)
        except Exception as exc:
            report.issues.append(f"Tokenizer/model config doğrulanamadı: {exc}")
    if files and tokenizer is not None:
        try:
            sample_dataset = PackedCausalDataset(JSONLDocumentStream(files), tokenizer, spec.sequence_length, shuffle_buffer=1)
            sample = next(iter(sample_dataset))
            if sample.numel() != spec.sequence_length + 1:
                report.issues.append("Sequence packing beklenen uzunlukta örnek üretmedi.")
            report.data_documents = manifest.get("stats", {}).get("kept") if manifest else None
        except Exception as exc:
            report.issues.append(f"Veri/sequence packing doğrulanamadı: {exc}")
    if run_smoke and not report.issues and tokenizer is not None:
        smoke = run_smoke_test(model_preset(spec.preset, tokenizer.vocab_size, min(256, spec.sequence_length),
                                            gradient_checkpointing=True), batch_size=spec.batch_size,
                               steps=spec.smoke_steps, device="cuda")
        report.smoke = smoke.to_dict()
    report.ready = not report.issues
    output = Path(output_dir)
    _atomic_json(output / "pilot.preflight.json", {"spec": asdict(spec), "report": report.to_dict(),
                  "data_files": [str(path) for path in files], "tokenizer": str(tokenizer_file),
                  "manifest": str(manifest_file)})
    return report

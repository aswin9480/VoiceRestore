from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4a", ".aac", ".mp3", ".wav", ".flac", ".ogg"]


def resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load_config() -> dict:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    local = ROOT / "config.local.json"
    if local.exists():
        config.update(json.loads(local.read_text(encoding="utf-8")))
    return config


def save_config(config: dict) -> None:
    path = ROOT / "config.local.json"
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(config, indent=2), encoding="utf-8")
    pending.replace(path)


def prepare() -> None:
    for folder in ["logs", "temp", "models/uvr", "models/cache", "outputs"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["GRADIO_TEMP_DIR"] = str(ROOT / "temp" / "gradio")


@dataclass
class Settings:
    preset: str = "Auditorium Speech"
    channel: str = "Auto"
    clearvoice: bool = True
    dereverb: bool = False
    vocal: bool = False
    resemble: bool = True
    dereverb_model: str = ""
    vocal_model: str = ""
    uvr_directory: str = "models/uvr"
    dereverb_stem: str = "No Reverb"
    vocal_stem: str = "Vocals"
    aggression: str = "Conservative"
    restoration: str = "Balanced"
    denoise: bool = False
    solver: str = "midpoint"
    nfe: int = 64
    tau: float = 0.5
    lambd: float = 0.25
    highpass: bool = True
    highpass_hz: int = 80
    eq: bool = True
    eq_amount: float = 1.0
    compressor: bool = True
    limiter: bool = True
    loudness: bool = True
    target_lufs: float = -16.0
    device: str = "auto"
    low_vram: bool = True
    chunk_seconds: int = 30
    overlap_seconds: float = 1.0
    save_intermediates: bool = False

    def validate(self) -> None:
        if not 5 <= self.chunk_seconds <= 120:
            raise ValueError("Chunk length must be 5–120 seconds.")
        if not 0 <= self.overlap_seconds < self.chunk_seconds / 2:
            raise ValueError("Overlap must be less than half the chunk length.")
        if self.channel not in ["Auto", "Stereo → Mono", "Left Channel", "Right Channel", "Keep Stereo"]:
            raise ValueError("Unknown channel mode.")
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError("Unknown compute device.")
        if self.solver not in ["midpoint", "rk4", "euler"] or not 1 <= self.nfe <= 128:
            raise ValueError("Invalid Resemble solver or NFE.")
        if not (0 <= self.tau <= 1 and 0 <= self.lambd <= 1):
            raise ValueError("Tau and Lambda must be between zero and one.")
        if not -30 <= self.target_lufs <= -10 or not 20 <= self.highpass_hz <= 250:
            raise ValueError("Invalid loudness target or high-pass frequency.")
        for enabled, name in [(self.dereverb, self.dereverb_model), (self.vocal, self.vocal_model)]:
            if enabled:
                from .models.uvr import model_path

                model_path(self.uvr_directory, name)

    def to_dict(self) -> dict:
        return asdict(self)

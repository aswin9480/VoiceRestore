from dataclasses import replace

from ..config import Settings

RESTORATION = {"Natural": (32, 0.3, 0.1), "Balanced": (64, 0.5, 0.25), "Strong Restore": (96, 0.65, 0.5)}
PRESETS = ["Auditorium Speech", "Natural Speech", "Podcast Clear", "Strong Cleanup", "Custom"]


def preset(name: str) -> Settings:
    s = Settings(preset=name)
    if name == "Natural Speech":
        s = replace(s, restoration="Natural", nfe=32, tau=0.3, lambd=0.1, eq_amount=0.5, compressor=False)
    elif name == "Podcast Clear":
        s = replace(s, highpass_hz=70, eq_amount=0.7)
    elif name == "Strong Cleanup":
        s = replace(s, dereverb=True, restoration="Strong Restore", nfe=96, tau=0.65, lambd=0.5)
    return s


def pipeline_text(s: Settings) -> str:
    stages = [
        ("FFmpeg preprocessing", True),
        ("UVR Vocal Isolation", s.vocal),
        ("ClearVoice · MossFormer2_SE_48K", s.clearvoice),
        ("UVR De-Reverb", s.dereverb),
        (f"Resemble Enhance · {s.restoration}", s.resemble),
        ("High-pass filter", s.highpass),
        ("Speech clarity EQ", s.eq),
        ("Compressor", s.compressor),
        ("Limiter · -1 dBTP", s.limiter),
        (f"Loudness · {s.target_lufs:g} LUFS", s.loudness),
        ("Export", True),
    ]
    return "\n".join(f"{i}. {name} — {'Enabled' if on else 'Disabled'}" for i, (name, on) in enumerate(stages, 1))

import numpy as np
import soundfile as sf

CSS = """
.gradio-container { max-width: 1460px !important; }
body, .gradio-container { background: #0b1019 !important; }
#hero { padding: 26px 28px; border: 1px solid #263448; border-radius: 18px;
background: linear-gradient(115deg, #142a38, #151c31 65%, #231a35); margin-bottom: 18px; }
#hero h1 {font-size: 34px; letter-spacing: -1px; margin-bottom: 2px; color: #f1f6ff;}
#hero .eyebrow {color: #65dfcb; letter-spacing: 2px; font-size: 11px; font-weight: 700;}
#hero p {color: #b7c5d9;}
#process { min-height: 52px; font-weight: 700; }
#console textarea {font-family: Consolas, monospace; font-size: 12px; background: #080d15;}
.pipeline-card {border: 1px solid #293349; border-radius: 14px; padding: 16px;}
.footer {color: #8493ab; font-size: 12px; text-align: center; padding: 20px;}
"""


def waveform(path: str, label: str) -> str:
    with sf.SoundFile(path) as reader:
        data = reader.read(min(len(reader), reader.samplerate * 60), dtype="float32", always_2d=True)
    if len(data) == 0:
        return ""
    mono = np.max(np.abs(data), axis=1)
    peaks = [float(np.max(block)) if len(block) else 0 for block in np.array_split(mono, 220)]
    bars = "".join(
        f'<rect x="{i * 3}" y="{35 - max(1, p * 32):.2f}" width="2" height="{max(2, p * 64):.2f}" rx="1"/>'
        for i, p in enumerate(peaks)
    )
    return f'<div style="color:#94a9c2;font-size:12px">{label} · first 60 seconds</div><svg role="img" aria-label="{label} waveform" viewBox="0 0 660 70" style="width:100%;height:70px;fill:#61d9bd">{bars}</svg>'

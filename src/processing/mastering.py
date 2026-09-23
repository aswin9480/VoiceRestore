import json
import math
import re
from pathlib import Path

from ..media.ffmpeg import convert
from ..process import run


def measure(path: Path, config: dict, cancel, target: float = -16) -> dict:
    output = run(
        [
            config["ffmpeg"],
            "-hide_banner",
            "-nostdin",
            "-i",
            str(path),
            "-af",
            f"loudnorm=I={target}:TP=-1:LRA=11:print_format=json",
            "-f",
            "null",
            "-",
        ],
        cancel,
    )
    matches = re.findall(r'\{\s*"input_i".*?\}', output, re.S)
    if not matches:
        raise RuntimeError("FFmpeg did not return a loudness measurement.")
    return json.loads(matches[-1])


def master(source: Path, target: Path, settings, config, cancel, log) -> dict:
    filters = []
    if settings.highpass:
        filters += [f"highpass=f={settings.highpass_hz}:p=2"]
    if settings.eq:
        a = settings.eq_amount
        filters += [
            f"equalizer=f=280:t=q:w=0.8:g={-2 * a}",
            f"equalizer=f=650:t=q:w=1:g={-a}",
            f"equalizer=f=3000:t=q:w=0.8:g={1.5 * a}",
            f"equalizer=f=5000:t=q:w=1:g={a}",
        ]
    if settings.compressor:
        filters += ["acompressor=threshold=0.1258925:ratio=2.5:attack=10:release=120:makeup=1"]
    if settings.limiter:
        # Oversampling before the limiter controls intersample peaks, then return to 48 kHz.
        filters += ["aresample=192000", "alimiter=limit=0.8912509:level=false:latency=true", "aresample=48000"]
    premaster = target.with_name("premaster.wav")
    convert(source, premaster, config, cancel, ",".join(filters))
    measured = measure(premaster, config, cancel, settings.target_lufs)
    norm = ""
    if settings.loudness and all(
        math.isfinite(float(measured[k])) for k in ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset"]
    ):
        norm = (
            f"loudnorm=I={settings.target_lufs}:TP=-1:LRA=11:linear=true:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}"
        )
        log(f"Applying measured two-pass loudness normalization: {settings.target_lufs} LUFS / -1 dBTP.")
    elif settings.loudness:
        log("Silence or unmeasurable loudness: preserving audio without gain normalization.")
    args = [config["ffmpeg"], "-nostdin", "-hide_banner", "-v", "error", "-y", "-i", str(premaster)]
    if norm:
        args += ["-af", norm]
    run(args + ["-ar", "48000", "-c:a", "pcm_s24le", str(target)], cancel)
    premaster.unlink(missing_ok=True)
    actual = measure(target, config, cancel, settings.target_lufs)
    return {
        "integrated_lufs": actual["input_i"],
        "true_peak_dbtp": actual["input_tp"],
        "loudness_range_lu": actual["input_lra"],
        "normalization_applied": bool(norm),
    }

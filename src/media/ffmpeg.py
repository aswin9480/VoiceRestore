from pathlib import Path

from ..process import run


def convert(
    source: Path, target: Path, config: dict, cancel, filters: str = "", extra: list[str] | None = None
) -> None:
    args = [config["ffmpeg"], "-hide_banner", "-nostdin", "-y", "-v", "error", "-i", str(source), "-vn"]
    if filters:
        args += ["-af", filters]
    args += extra or []
    args += ["-ar", "48000", "-c:a", "pcm_f32le", str(target)]
    run(args, cancel)


def extract(source: Path, target: Path, info: dict, mode: str, config: dict, cancel, log) -> None:
    import numpy as np
    import soundfile as sf

    channels = info["channels"]
    filters = ""
    output_channels = 2 if mode == "Keep Stereo" and channels > 1 else 1
    if mode == "Right Channel":
        if channels < 2:
            raise ValueError("Right Channel requires an input with at least two channels.")
        filters = "pan=mono|c0=c1"
    elif mode == "Left Channel":
        filters = "pan=mono|c0=c0"
    elif mode == "Auto" and channels == 2:
        # Inspect bounded excerpts at the start, middle and end for phase cancellation / dead channels.
        sample = target.with_name("channel_analysis.wav")
        energy = np.zeros(2)
        mono_energy = 0.0
        for offset in sorted(set([0, max(0, info["duration"] / 2 - 5), max(0, info["duration"] - 10)])):
            run(
                [
                    config["ffmpeg"],
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(offset),
                    "-i",
                    str(source),
                    "-map",
                    f"0:{info['audio_index']}",
                    "-t",
                    "10",
                    "-ac",
                    "2",
                    "-ar",
                    "16000",
                    str(sample),
                ],
                cancel,
            )
            data, _ = sf.read(sample, always_2d=True)
            energy += np.sum(data * data, axis=0)
            mono_energy += float(np.sum(np.mean(data, axis=1) ** 2))
        sample.unlink(missing_ok=True)
        if min(energy) < max(energy) * 0.01 or mono_energy < sum(energy) * 0.025:
            selected = int(np.argmax(energy))
            filters = f"pan=mono|c0=c{selected}"
            log(
                f"Auto channel analysis: selected {'left' if selected == 0 else 'right'} to avoid a dead channel or phase cancellation."
            )
        else:
            log("Auto channel analysis: stereo mixed to mono (recommended for speech).")
    convert(source, target, config, cancel, filters, ["-map", f"0:{info['audio_index']}", "-ac", str(output_channels)])
    log(f"Extracted 48 kHz / {output_channels} channel / float WAV.")

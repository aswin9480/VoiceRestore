from pathlib import Path

from ..process import run


def remux(source: Path, audio: Path, folder: Path, info: dict, config: dict, cancel, log) -> Path:
    # Copy codec bitstream. Matroska fallback handles codecs unsupported by MP4.
    offset = info["audio_start"] - info["video"]["start"]
    common = [
        config["ffmpeg"],
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-itsoffset",
        str(offset),
        "-i",
        str(audio),
        "-map",
        f"0:{info['video']['index']}",
        "-map",
        "1:a:0",
        "-map_metadata",
        "0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
    ]
    target = folder / "final_restored.mp4"
    try:
        run(common + ["-movflags", "+faststart", str(target)], cancel)
    except RuntimeError as error:
        target.unlink(missing_ok=True)
        log(f"MP4 stream-copy remux failed; trying MKV without video re-encoding. {error}")
        target = folder / "final_restored.mkv"
        run(common + [str(target)], cancel)
    return target

import json
from pathlib import Path

from ..process import run


def probe(path: Path, config: dict, cancel=None) -> dict:
    data = json.loads(
        run(
            [config["ffprobe"], "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            cancel,
            timeout=30,
        )
    )
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    video = next(
        (s for s in data["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if not audio:
        raise ValueError("This file has no audio stream.")
    return {
        "filename": path.name,
        "duration": float(data["format"].get("duration", audio.get("duration", 0))),
        "format": data["format"]["format_name"],
        "sample_rate": int(audio["sample_rate"]),
        "channels": audio["channels"],
        "size_bytes": path.stat().st_size,
        "audio_index": audio["index"],
        "audio_start": float(audio.get("start_time", 0)),
        "video": {
            "index": video["index"],
            "codec": video["codec_name"],
            "width": video["width"],
            "height": video["height"],
            "start": float(video.get("start_time", 0)),
        }
        if video
        else None,
    }

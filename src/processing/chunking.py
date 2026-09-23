"""Bounded-memory overlap crossfades. A transform sees at most one chunk."""

from pathlib import Path

import numpy as np
import soundfile as sf

from ..process import check_cancel


def process_chunks(source: Path, target: Path, transform, seconds: float, overlap: float, cancel, log) -> None:
    with sf.SoundFile(source) as reader:
        sr, channels, total = reader.samplerate, reader.channels, len(reader)
        length, fade = int(seconds * sr), int(overlap * sr)
        if length < 1 or fade < 0 or fade * 2 >= length:
            raise ValueError("Invalid chunk/overlap lengths.")
        step = length - fade
        tail = None
        with sf.SoundFile(target, "w", samplerate=sr, channels=channels, subtype="FLOAT") as writer:
            start, index = 0, 0
            while start < total:
                check_cancel(cancel)
                reader.seek(start)
                data = reader.read(min(length, total - start), dtype="float32", always_2d=True)
                index += 1
                log(f"Chunk {index}: {start / sr:.1f}–{(start + len(data)) / sr:.1f}s / {total / sr:.1f}s")
                result = np.asarray(transform(data, sr), dtype=np.float32)
                if result.shape != data.shape or not np.isfinite(result).all():
                    raise ValueError(f"Model produced invalid audio: expected {data.shape}, got {result.shape}.")
                check_cancel(cancel)
                if tail is not None:
                    n = min(len(tail), len(result))
                    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
                    result[:n] = tail[:n] * (1 - ramp) + result[:n] * ramp
                last = start + len(data) >= total
                if last or fade == 0:
                    writer.write(result)
                    tail = None
                else:
                    writer.write(result[:-fade])
                    tail = result[-fade:].copy()
                if last:
                    break
                start += step

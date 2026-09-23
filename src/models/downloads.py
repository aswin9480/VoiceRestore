"""Atomic, visible model downloads. Processing never calls this with network enabled."""

import os
import time
import urllib.request
from pathlib import Path


def fingerprint(path: Path) -> dict:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size:
        return
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(f"Missing cached model/config: {target.name}. Use Download / Verify Models first.")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".part")
    received = 0
    last = 0.0
    try:
        with urllib.request.urlopen(url, timeout=120) as response, pending.open("wb") as output:
            expected = int(response.headers.get("Content-Length", 0))
            while block := response.read(1024 * 1024):
                output.write(block)
                received += len(block)
                now = time.monotonic()
                if now - last > 1:
                    suffix = (
                        f" / {expected / 1024**2:.1f} MiB ({received / expected:.0%})"
                        if expected
                        else " (total size unknown)"
                    )
                    print(f"Downloading {target.name}: {received / 1024**2:.1f} MiB{suffix}", flush=True)
                    last = now
            if received == 0 or (expected and expected != received):
                raise IOError(f"Incomplete download: {received}/{expected} bytes for {target.name}")
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)

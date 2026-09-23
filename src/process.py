"""Cancellable child processes, with no shell interpolation."""

from __future__ import annotations

import subprocess
import threading
from collections import deque
from typing import Callable


class Cancelled(Exception):
    pass


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled("Job cancelled.")


def run(
    args: list[str],
    cancel: threading.Event | None = None,
    log: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> str:
    cancel = cancel or threading.Event()
    check_cancel(cancel)
    process = subprocess.Popen(
        [str(x) for x in args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    lines: deque[str] = deque(maxlen=4000)

    def read() -> None:
        assert process.stdout
        for line in process.stdout:
            lines.append(line)
            if log:
                log(line.rstrip())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    import time

    start = time.monotonic()
    try:
        while process.poll() is None:
            check_cancel(cancel)
            if timeout and time.monotonic() - start > timeout:
                raise TimeoutError(f"{args[0]} exceeded {timeout}s.")
            cancel.wait(0.1)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=5)
        if process.stdout:
            process.stdout.close()
    check_cancel(cancel)
    output = "".join(lines)
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed (exit {process.returncode}):\n{output[-5000:]}")
    return output

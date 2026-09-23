"""Build a documented inference-only wheel from the SHA256-verified upstream 0.0.1 wheel.

Only training imports/visualization are removed. Neural networks, solvers, inference,
and upstream licenses are unchanged. Never stub DeepSpeed or mock model inference.
"""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.0.1+voicerestore.1"
DEPENDENCIES = [
    "celluloid==0.2.0",
    "librosa==0.10.2.post1",
    "matplotlib==3.8.4",
    "numpy==1.26.4",
    "omegaconf==2.3.0",
    "pandas==2.2.3",
    "rich==13.9.4",
    "scipy==1.14.1",
    "soundfile==0.12.1",
    "torch==2.5.1",
    "torchaudio==2.5.1",
    "torchvision==0.20.1",
    "tqdm>=4.66",
    "resampy==0.4.3",
    "tabulate==0.9.0",
]


def patch_source(name: str, source: str) -> str:
    if name.endswith("enhancer/inference.py"):
        old = "from .train import Enhancer, HParams"
        if old not in source:
            raise RuntimeError("Upstream Resemble inference changed; compatibility patch requires review.")
        return source.replace(old, "from .enhancer import Enhancer\nfrom .hparams import HParams")
    if name.endswith("denoiser/inference.py"):
        return source.replace(
            "from .train import Denoiser, HParams", "from .denoiser import Denoiser\nfrom .hparams import HParams"
        )
    if name.endswith("enhancer/enhancer.py"):
        source = source.replace("from ..utils.distributed import global_leader_only\n", "").replace(
            "from ..utils.train_loop import TrainLoop\n", ""
        )
        source = source.replace("    @global_leader_only\n", "")
        # Remove training-only visualization without rewriting other source or Unicode identifiers.
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "_visualize")
        lines = source.splitlines(keepends=True)
        lines[node.body[0].lineno - 1 : node.end_lineno] = [
            '        raise RuntimeError("This distribution supports inference only, not training.")\n'
        ]
        return "".join(lines)
    return source


def build(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"resemble_enhance-{VERSION}-py3-none-any.whl"
    if target.exists():
        return target
    metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/resemble-enhance/0.0.1/json", timeout=60))
    release = next(item for item in metadata["urls"] if item["filename"].endswith(".whl"))
    data = urllib.request.urlopen(release["url"], timeout=120).read()
    if hashlib.sha256(data).hexdigest() != release["digests"]["sha256"]:
        raise RuntimeError("Upstream Resemble wheel checksum mismatch.")
    old_info = "resemble_enhance-0.0.1.dist-info"
    new_info = f"resemble_enhance-{VERSION}.dist-info"
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for old_name in wheel.namelist():
            if old_name.endswith("/RECORD"):
                continue
            content = wheel.read(old_name)
            name = old_name.replace(old_info, new_info)
            if name.endswith(".py"):
                content = patch_source(name, content.decode("utf-8")).encode("utf-8")
            if name.endswith("/METADATA"):
                lines = content.decode("utf-8").splitlines()
                lines = [line for line in lines if not line.startswith("Requires-Dist:")]
                lines[lines.index("Version: 0.0.1")] = f"Version: {VERSION}"
                at = lines.index("")
                lines[at:at] = [f"Requires-Dist: {item}" for item in DEPENDENCIES]
                content = "\n".join(lines).encode("utf-8")
            files[name] = content
    files["resemble_enhance/VOICERESTORE_PATCH.txt"] = (
        "Inference-only Windows adaptation of resemble-enhance 0.0.1.\n"
        "Training imports and training visualization removed; inference mathematics unchanged.\n"
        f"Upstream wheel SHA256: {release['digests']['sha256']}\n"
    ).encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(content)])
    writer.writerow([new_info + "/RECORD", "", ""])
    files[new_info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, content in files.items():
            wheel.writestr(name, content)
    return target

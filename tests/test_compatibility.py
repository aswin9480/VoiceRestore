import ast

from scripts.build_resemble import patch_source


def test_resemble_patch_preserves_inference_math():
    source = "from .train import Enhancer, HParams\nresult = model(x)\n"
    patched = patch_source("enhancer/inference.py", source)
    assert "from .enhancer import Enhancer" in patched
    assert "result = model(x)" in patched
    ast.parse(patched)


def test_port_fallback():
    import socket

    from app import available_port

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        assert available_port(port) != port

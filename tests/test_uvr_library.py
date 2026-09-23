import json

import pytest

from src.models.uvr import model_path, scan, stage_local_model


def test_recursive_uvr5_scan(tmp_path):
    for name in [
        "MDX_Net_Models/Kim_Vocal_2.onnx",
        "VR_Models/UVR-DeEcho-DeReverb.pth",
        "VR_Models/UVR-DeNoise-Lite.pth",
        "Demucs_Models/weights.th",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test fixture, not a real model")
    dereverb, vocal = scan(str(tmp_path))
    assert "MDX_Net_Models/Kim_Vocal_2.onnx" in vocal
    assert "VR_Models/UVR-DeEcho-DeReverb.pth" in dereverb
    assert not any("DeNoise" in name or name.endswith(".th") for name in dereverb + vocal)
    assert model_path(str(tmp_path), vocal[0]).is_file()


def test_uvr_path_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        model_path(str(tmp_path), "../outside.pth")


def test_uvr5_metadata_staging_is_read_only_to_source(tmp_path):
    source = tmp_path / "UVR5/MDX_Net_Models/model.ckpt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture weights")
    metadata = source.parent / "model_data/model_data.json"
    metadata.parent.mkdir()
    metadata.write_text('{"known_hash": {"primary_stem": "Vocals"}}')
    yaml = source.parent / "model_data/mdx_c_configs/model.yaml"
    yaml.parent.mkdir()
    yaml.write_text("audio:\n  sample_rate: 44100\n")
    before = {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    cache = stage_local_model(source, tmp_path / "cache")
    assert (cache / source.name).read_bytes() == source.read_bytes()
    assert (cache / "model.yaml").read_bytes() == yaml.read_bytes()
    assert json.loads((cache / "mdx_model_data.json").read_text())["known_hash"]["primary_stem"] == "Vocals"
    after = {p.relative_to(source.parent): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
    assert before == after
    target_mtime = (cache / source.name).stat().st_mtime_ns
    stage_local_model(source, tmp_path / "cache")
    assert (cache / source.name).stat().st_mtime_ns == target_mtime

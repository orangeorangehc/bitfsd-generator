import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def test_multiseed_preserves_all_seeds_and_refuses_overwrite(tmp_path):
    config = {
        "real_tracks": ["FSE22_test.yaml"],
        "synthetic_tracks": ["triangle"],
        "ego": {"spacing": 100.0},
        "augmentation": {"flip": False, "seeds": [42, 43]},
        "output": {"dir": str(tmp_path / "corpus"), "coordinate_frame": "ego"},
    }
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(config))
    command = [sys.executable, str(PROJECT / "collect_multiseed.py"), "--config", str(config_path)]
    subprocess.run(command, cwd=PROJECT, check=True, capture_output=True, text=True)
    root = tmp_path / "corpus"
    manifest_path = root / "dataset_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    expected = {"FSE22_test_s42", "FSE22_test_s43", "triangle_s42", "triangle_s43"}
    assert {item["name"] for item in manifest["tracks"]} == expected
    assert manifest["coordinate_frame"] == "ego"
    assert manifest["collection_config"]["augmentation"]["seeds"] == [42, 43]
    for name in expected:
        meta = yaml.safe_load((root / name / "metadata.yaml").read_text())
        assert meta["track_name"] == name
        assert name == f"{meta['source']['track']}_s{meta['source']['seed']}"
        assert meta["source"]["type"] == ("real" if name.startswith("FSE") else "synthetic")
        assert all((root / name / frame["file"]).is_file() for frame in meta["frames"])
    assert (root / "triangle_s42/cloud_0.txt").read_bytes() != (root / "triangle_s43/cloud_0.txt").read_bytes()
    before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    second = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == before

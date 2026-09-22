"""Check the collection entry point using small temporary outputs."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

PROJECT = Path(__file__).resolve().parents[1]


class GenerateScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="generator-script-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "corpus"
        self.config = self.root / "generate.yaml"
        self.recipe = {
            "real_tracks": ["FSE22_test.yaml"], "synthetic_tracks": ["triangle"],
            "ego": {"spacing": 100.0},
            "augmentation": {"flip": False, "seeds": [42, 43]},
            "output": {"dir": str(self.output), "coordinate_frame": "ego"},
        }
        self.config.write_text(yaml.safe_dump(self.recipe))

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(PROJECT / "generate_data.sh"), "--config", str(self.config), *args],
            cwd=self.root, env=dict(os.environ, PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=60,
        )

    def test_preview_from_foreign_directory_creates_no_output(self):
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("variants=4", result.stdout)
        self.assertFalse(self.output.exists())

    def test_generate_all_variants_and_refuse_overwrite(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest_path = self.output / "dataset_manifest.yaml"
        original = manifest_path.read_bytes()
        manifest = yaml.safe_load(original)
        self.assertEqual({t["name"] for t in manifest["tracks"]},
                         {"FSE22_test_s42", "FSE22_test_s43", "triangle_s42", "triangle_s43"})
        again = self.run_script()
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(manifest_path.read_bytes(), original)

    def test_missing_source_is_reported_before_output_is_created(self):
        self.recipe["real_tracks"] = ["missing.yaml"]
        self.config.write_text(yaml.safe_dump(self.recipe))
        result = self.run_script("--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Real track not found", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

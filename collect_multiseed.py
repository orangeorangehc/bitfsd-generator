#!/usr/bin/env python3
"""Collect several observation seeds into one manifest-backed SideNet corpus."""

import argparse
import copy
import tempfile
from pathlib import Path

import yaml

from main import cmd_collect
from export import write_sidenet_manifest


def collect_multiseed(config_path, output_dir=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    seeds = cfg.get("augmentation", {}).get("seeds", [])
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("augmentation.seeds must be a nonempty list of unique nonnegative integers")
    if cfg.get("augmentation", {}).get("flip", False):
        raise ValueError("This collection recipe requires augmentation.flip: false")
    if cfg.get("output", {}).get("coordinate_frame", "ego") != "ego":
        raise ValueError("This collection recipe requires ego coordinates")

    root = Path(output_dir or cfg["output"]["dir"]).resolve()
    if root.exists():
        raise FileExistsError(f"Choose a new output directory; already exists: {root}")
    real_names = [Path(name).stem for name in cfg.get("real_tracks", [])]
    synthetic_names = cfg.get("synthetic_tracks", [])
    if len(set(real_names)) != len(real_names) or len(set(synthetic_names)) != len(synthetic_names):
        raise ValueError("Duplicate track/preset names in collection configuration")
    if set(real_names) & set(synthetic_names):
        raise ValueError("Real and synthetic track names must be distinct")

    root.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every seed finishes; existing datasets are never overwritten.
    with tempfile.TemporaryDirectory(prefix=".multiseed-", dir=root.parent) as temporary:
        staging = Path(temporary)
        corpus = staging / "corpus"
        corpus.mkdir()
        names = []
        for seed in seeds:
            seed_dir = staging / f"seed_{seed}"
            cmd_collect(argparse.Namespace(config=str(config_path), seed=seed, output=str(seed_dir)))
            manifest = yaml.safe_load((seed_dir / "dataset_manifest.yaml").read_text())
            expected = {
                **{name: ("real", name) for name in real_names},
                **{f"{name}_s{seed}": ("synthetic", name) for name in synthetic_names},
            }
            actual = [item["name"] for item in manifest["tracks"]]
            if len(actual) != len(expected) or set(actual) != set(expected):
                raise ValueError(f"Collection omitted or duplicated configured tracks for seed {seed}")
            for old_name in actual:
                source_type, source_track = expected[old_name]
                new_name = f"{source_track}_s{seed}"
                if new_name in names:
                    raise ValueError(f"Output track name collision: {new_name}")
                track_dir = seed_dir / old_name
                metadata_path = track_dir / "metadata.yaml"
                metadata = yaml.safe_load(metadata_path.read_text())
                metadata["track_name"] = new_name
                metadata["source"] = {"type": source_type, "track": source_track, "seed": seed}
                metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
                track_dir.rename(corpus / new_name)
                names.append(new_name)

        manifest_path = write_sidenet_manifest(corpus, names, coordinate_frame="ego")
        manifest = yaml.safe_load(manifest_path.read_text())
        resolved_cfg = copy.deepcopy(cfg)
        resolved_cfg["output"]["dir"] = str(root)
        manifest["collection_config"] = resolved_cfg
        manifest["seed_semantics"] = "observation_noise_only; geometry fixed by track or preset"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
        corpus.rename(root)
    print(f"Published {len(names)} track variants, seeds={seeds}: {root}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/perceive_mixed.yaml")
    parser.add_argument("--output", help="New output directory (must not already exist)")
    args = parser.parse_args()
    collect_multiseed(args.config, args.output)

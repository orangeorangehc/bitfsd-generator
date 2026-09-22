#!/usr/bin/env python3
"""BITFSD Generator CLI: track generation, perception simulation, and web server."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "config"


def cmd_generate(args):
    from track_generator import TrackGenerator, TrackPreset
    from track import TrackData
    from export import export_track_yaml
    import yaml

    config_path = Path(args.config)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults", {})
    preset_dict = data.get("presets", {}).get(args.preset)
    if preset_dict is None:
        available = list(data.get("presets", {}).keys())
        print(f"Preset '{args.preset}' not found. Available: {available}")
        sys.exit(1)

    seed = args.seed if args.seed is not None else preset_dict.get("seed", 42)
    preset = TrackPreset(
        name=args.preset,
        segments=preset_dict["segments"],
        track_width=preset_dict.get("track_width", defaults.get("track_width", 4.0)),
        cone_spacing_min=preset_dict.get("cone_spacing_min", defaults.get("cone_spacing_min", 5.0)),
        cone_spacing_max=preset_dict.get("cone_spacing_max", defaults.get("cone_spacing_max", 15.0)),
    )

    gen = TrackGenerator(preset, seed=seed)
    track_dict = gen.generate()
    track = TrackData(f"{args.preset}_s{seed}", track_dict)
    print(f"Generated: {track}")

    output_dir = Path(args.output)
    output_path = output_dir / f"{track.name}.yaml"
    export_track_yaml(track_dict, output_path)
    print(f"Saved to {output_path}")


def cmd_perceive(args):
    from track import load_track, load_all_tracks
    from perception import PerceptionPipeline, PerceptionConfig
    from export import sidenet_sequence_to_dir

    track_path = Path(args.track)
    if track_path.is_dir():
        tracks = load_all_tracks(track_path)
        if not tracks:
            print(f"No tracks found in {track_path}")
            sys.exit(1)
        track = list(tracks.values())[0]
        print(f"Loaded: {track}")
    else:
        track = load_track(track_path)
        print(f"Loaded: {track}")

    config = PerceptionConfig(
        ego_spacing=args.ego_spacing,
        position_noise_xy=args.noise_xy,
        drop_rate=args.drop_rate,
        fp_rate=args.fp_rate,
        lidar_range_max=args.lidar_range,
        lidar_fov_deg=args.lidar_fov,
        seed=args.seed,
    )
    pipeline = PerceptionPipeline(config)
    sequence = pipeline.run(track)
    print(f"Perception: {sequence.stats}")

    output_dir = Path(args.output)
    paths = sidenet_sequence_to_dir(sequence, output_dir)
    print(f"Saved {len(paths)} frames to {output_dir / sequence.track_name}/")


def cmd_serve(args):
    from server import main as server_main
    server_main()


def cmd_infer(args):
    """Run SideNet/DGCNN inference on a track and compare with GT labels."""
    from track import load_track
    from perception import PerceptionPipeline, PerceptionConfig
    from sidenet_predictor import SideNetPredictor

    track = load_track(args.track)
    print(f"Track: {track}")

    config = PerceptionConfig(
        ego_spacing=args.ego_spacing,
        position_noise_xy=0.0 if args.no_noise else args.noise_xy,
        position_noise_z=0.0 if args.no_noise else 0.03,
        drop_rate=0.0 if args.no_noise else args.drop_rate,
        fp_rate=0.0 if args.no_noise else args.fp_rate,
        lidar_range_max=args.lidar_range,
        lidar_fov_deg=args.lidar_fov,
        seed=args.seed,
    )

    pipeline = PerceptionPipeline(config)
    sequence = pipeline.run(track)
    print(f"Frames: {len(sequence.frames)}, cones: {sum(len(f.cones) for f in sequence.frames)}")

    predictor = SideNetPredictor(args.ckpt)

    correct, total = 0, 0
    l_to_r, r_to_l = 0, 0  # confusion counts
    per_frame_acc = []

    for frame in sequence.frames:
        cones = frame.cones
        if not cones:
            continue

        gt = np.array([0 if "Left" in c.side else 1 for c in cones])
        pred = predictor.predict(cones)

        n_correct = (pred == gt).sum()
        n_total = len(gt)
        correct += n_correct
        total += n_total
        per_frame_acc.append(n_correct / n_total)

        # Confusion
        l_to_r += ((gt == 0) & (pred == 1)).sum()
        r_to_l += ((gt == 1) & (pred == 0)).sum()

    if total == 0:
        print("No cones to evaluate.")
        return

    per_frame_acc = np.array(per_frame_acc)
    print(f"\n{'='*50}")
    print(f"Results for {track.name}")
    print(f"{'='*50}")
    print(f"Frames evaluated: {len(per_frame_acc)}")
    print(f"Cones evaluated:  {total}")
    print(f"Overall accuracy: {correct/total:.4f} ({correct}/{total})")
    print(f"Per-frame accuracy: min={per_frame_acc.min():.4f} "
          f"mean={per_frame_acc.mean():.4f} max={per_frame_acc.max():.4f}")
    print(f"Confusion: Left→Right={l_to_r}, Right→Left={r_to_l}")


def cmd_collect(args):
    """YAML-driven batch data collection for SideNet training."""
    from track import load_track, TrackData
    from track_generator import TrackGenerator, TrackPreset
    from perception import PerceptionPipeline, PerceptionConfig
    from export import (
        sidenet_sequence_to_dir,
        write_sidenet_manifest,
        write_sidenet_track_metadata,
    )
    import yaml

    # ── Load config ──────────────────────────────────────────────────
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    seed = args.seed if args.seed is not None else cfg.get("augmentation", {}).get("seed", 42)
    output_cfg = cfg.get("output", {})
    output_dir = Path(args.output) if args.output else Path(
        output_cfg.get("dir", "output/sidenet_data_ego")
    )
    coordinate_frame = output_cfg.get("coordinate_frame", "ego")
    if coordinate_frame not in {"ego", "world"}:
        raise ValueError("output.coordinate_frame must be 'ego' or 'world'")

    sensor = cfg.get("sensor", {})
    ego = cfg.get("ego", {})
    perc = cfg.get("perception", {})

    perc_config = PerceptionConfig(
        ego_spacing=ego.get("spacing", 5.0),
        ego_noise_x=ego.get("noise_x", 0.1),
        ego_noise_y=ego.get("noise_y", 0.1),
        ego_noise_yaw_deg=ego.get("noise_yaw_deg", 1.0),
        lidar_range_min=sensor.get("lidar_range_min", 0.5),
        lidar_range_max=sensor.get("lidar_range_max", 30.0),
        lidar_fov_deg=sensor.get("lidar_fov_deg", 360.0),
        position_noise_xy=perc.get("position_noise_xy", 0.10),
        position_noise_z=perc.get("position_noise_z", 0.03),
        drop_rate=perc.get("drop_rate", 0.10),
        fp_rate=perc.get("fp_rate", 0.05),
        seed=seed,
    )

    flip = cfg.get("augmentation", {}).get("flip", False)
    pipeline = PerceptionPipeline(perc_config)

    # ── Load presets config for synthetic tracks ─────────────────────
    presets_path = CONFIG_DIR / "track_presets.yaml"
    with open(presets_path) as f:
        presets_data = yaml.safe_load(f)
    presets_defaults = presets_data.get("defaults", {})

    # ── Collect all tracks ───────────────────────────────────────────
    tracks = []

    for name in cfg.get("real_tracks", []):
        tpath = DATA_DIR / name
        if not tpath.exists():
            print(f"  SKIP real track not found: {tpath}")
            continue
        track = load_track(tpath)
        tracks.append(track)
        print(f"  Loaded real: {track}")

    for preset_name in cfg.get("synthetic_tracks", []):
        preset_dict = presets_data.get("presets", {}).get(preset_name)
        if preset_dict is None:
            print(f"  SKIP preset not found: {preset_name}")
            continue
        preset = TrackPreset(
            name=preset_name,
            segments=preset_dict["segments"],
            track_width=preset_dict.get("track_width", presets_defaults.get("track_width", 4.0)),
            cone_spacing_min=preset_dict.get("cone_spacing_min", presets_defaults.get("cone_spacing_min", 5.0)),
            cone_spacing_max=preset_dict.get("cone_spacing_max", presets_defaults.get("cone_spacing_max", 15.0)),
        )
        gen = TrackGenerator(preset, seed=seed)
        track_dict = gen.generate()
        track = TrackData(f"{preset_name}_s{seed}", track_dict)
        tracks.append(track)
        print(f"  Generated synthetic: {track}")

    if not tracks:
        print("No tracks to process. Check your config.")
        sys.exit(1)

    # ── Run perception + export ──────────────────────────────────────
    print(f"\nProcessing {len(tracks)} tracks, flip={flip}, seed={seed}")
    print(f"Output: {output_dir}\n")

    total_frames = 0
    exported_track_names = []
    for track in tracks:
        seq = pipeline.run(track, flip=False)
        paths = sidenet_sequence_to_dir(
            seq, output_dir, coordinate_frame=coordinate_frame
        )
        write_sidenet_track_metadata(
            seq,
            output_dir,
            paths,
            coordinate_frame=coordinate_frame,
        )
        exported_track_names.append(track.name)
        total_frames += len(paths)
        print(f"  {track.name}: {len(paths)} frames -> {output_dir / track.name}/")

        if flip:
            seq_f = pipeline.run(track, flip=True)
            flip_name = f"{track.name}_flip"
            # Override track_name in sequence for output directory naming
            seq_f.track_name = flip_name
            paths_f = sidenet_sequence_to_dir(
                seq_f, output_dir, coordinate_frame=coordinate_frame
            )
            write_sidenet_track_metadata(
                seq_f,
                output_dir,
                paths_f,
                coordinate_frame=coordinate_frame,
            )
            exported_track_names.append(flip_name)
            total_frames += len(paths_f)
            print(f"  {flip_name}: {len(paths_f)} frames -> {output_dir / flip_name}/")

    write_sidenet_manifest(
        output_dir,
        exported_track_names,
        coordinate_frame=coordinate_frame,
    )
    print(f"\nDone. {total_frames} total frames exported to {output_dir}")
    print(f"SideNet manifest: {output_dir / 'dataset_manifest.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        "bitfsd-generator",
        description="FSD track generation and perception data platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen = subparsers.add_parser("generate", help="Generate a synthetic track")
    gen.add_argument("--preset", required=True, help="Preset name from config")
    gen.add_argument("--config", default="config/track_presets.yaml", help="Presets config file")
    gen.add_argument("--seed", type=int, default=None, help="Random seed")
    gen.add_argument("--output", default="output/", help="Output directory")

    # perceive
    perc = subparsers.add_parser("perceive", help="Run perception simulation on a track")
    perc.add_argument("--track", required=True, help="Track YAML file or data directory")
    perc.add_argument("--output", default="output/", help="Output directory")
    perc.add_argument("--ego-spacing", type=float, default=5.0, help="Vehicle pose spacing (m)")
    perc.add_argument("--noise-xy", type=float, default=0.10, help="Position noise sigma (m)")
    perc.add_argument("--drop-rate", type=float, default=0.10, help="Detection dropout rate")
    perc.add_argument("--fp-rate", type=float, default=0.05, help="False positive rate")
    perc.add_argument("--lidar-range", type=float, default=30.0, help="LiDAR max range (m)")
    perc.add_argument("--lidar-fov", type=float, default=360.0, help="LiDAR total FoV (degrees)")
    perc.add_argument("--seed", type=int, default=42, help="Random seed")

    # collect
    col = subparsers.add_parser("collect", help="Batch collect SideNet training data from YAML config")
    col.add_argument("--config", default="config/perceive.yaml", help="Collection config file")
    col.add_argument("--seed", type=int, default=None, help="Override random seed")
    col.add_argument("--output", default=None, help="Override output directory")

    # serve
    serve = subparsers.add_parser("serve", help="Start web server")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--host", default="0.0.0.0")

    # infer
    inf = subparsers.add_parser("infer", help="Run SideNet/DGCNN inference on a track")
    inf.add_argument("--track", required=True, help="Track YAML file")
    inf.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    inf.add_argument("--ego-spacing", type=float, default=5.0)
    inf.add_argument("--noise-xy", type=float, default=0.10)
    inf.add_argument("--drop-rate", type=float, default=0.10)
    inf.add_argument("--fp-rate", type=float, default=0.05)
    inf.add_argument("--lidar-range", type=float, default=30.0)
    inf.add_argument("--lidar-fov", type=float, default=360.0)
    inf.add_argument("--seed", type=int, default=42)
    inf.add_argument("--no-noise", action="store_true",
                     help="Disable noise injection (use clean GT cones)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "perceive":
        cmd_perceive(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Export utilities for OpenPCDet, SideNet, and track YAML."""

import math
from pathlib import Path
from typing import List

import yaml

# Fixed cone bounding-box dimensions and heading for synthetic data
_DX = 0.200
_DY = 0.200
_DZ = 0.300
_HEADING = 0.000
_SCORE = 1.0


def _world_to_ego(
    x: float, y: float, z: float, ego_pose
) -> tuple[float, float, float]:
    """Transform a world point into the ego/LiDAR frame of ``ego_pose``."""
    px, py, pz, yaw = (float(value) for value in ego_pose)
    dx, dy = x - px, y - py
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * dx + sin_yaw * dy,
        -sin_yaw * dx + cos_yaw * dy,
        z - pz,
    )


def frame_to_sidenet(
    cones, ego_pose=None, *, coordinate_frame: str = "world"
) -> str:
    """Convert PerceptionCone list to SideNet training text format.

    Output per line: x y z dx dy dz heading score class_name
    Only ground_truth and gaussian_noise cones are included (false_positive skipped).
    """
    if coordinate_frame not in {"ego", "world"}:
        raise ValueError("coordinate_frame must be 'ego' or 'world'")
    if coordinate_frame == "ego" and ego_pose is None:
        raise ValueError("ego_pose is required for ego-frame SideNet export")

    lines = []
    for c in cones:
        if c.source == "false_positive":
            continue
        cls = "Cone_Left" if "Left" in c.side else "Cone_Right"
        x, y, z = c.x, c.y, c.z
        if coordinate_frame == "ego":
            x, y, z = _world_to_ego(x, y, z, ego_pose)
        lines.append(
            f"{x:.4f} {y:.4f} {z:.4f} "
            f"{_DX:.3f} {_DY:.3f} {_DZ:.3f} {_HEADING:.3f} {_SCORE:.1f} {cls}"
        )
    return "\n".join(lines)


def sidenet_sequence_to_dir(
    sequence, output_dir, *, coordinate_frame: str = "world"
) -> List[Path]:
    """Write all frames of a PerceptionSequence as SideNet training files.

    Creates: output_dir/{track_name}/cloud_0.txt, cloud_1.txt, ...
    Returns list of written file paths.
    """
    output_dir = Path(output_dir) / sequence.track_name
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for frame in sequence.frames:
        fname = output_dir / f"cloud_{frame.frame_id}.txt"
        text = frame_to_sidenet(
            frame.cones,
            frame.ego_pose,
            coordinate_frame=coordinate_frame,
        )
        fname.write_text(text)
        paths.append(fname)
    return paths


def write_sidenet_track_metadata(
    sequence,
    output_dir,
    frame_paths: List[Path],
    *,
    coordinate_frame: str = "ego",
) -> Path:
    """Write the SideNet v1 contract for one track directory."""
    if coordinate_frame not in {"ego", "world"}:
        raise ValueError("coordinate_frame must be 'ego' or 'world'")

    track_dir = Path(output_dir) / sequence.track_name
    metadata = {
        "schema_version": 1,
        "track_name": sequence.track_name,
        "coordinate_frame": coordinate_frame,
        "side_semantics": "track_global",
        "frames": [
            {
                "frame_id": int(frame.frame_id),
                "file": path.name,
                "ego_pose_world": [float(value) for value in frame.ego_pose],
            }
            for frame, path in zip(sequence.frames, frame_paths)
        ],
        "config": {
            "ego_spacing": float(sequence.config.ego_spacing),
            "lidar_range_min": float(sequence.config.lidar_range_min),
            "lidar_range_max": float(sequence.config.lidar_range_max),
            "lidar_fov_deg": float(sequence.config.lidar_fov_deg),
            "drop_rate": float(sequence.config.drop_rate),
            "position_noise_xy": float(sequence.config.position_noise_xy),
            "position_noise_z": float(sequence.config.position_noise_z),
            "seed": int(sequence.config.seed),
        },
    }
    path = track_dir / "metadata.yaml"
    path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return path


def write_sidenet_manifest(
    output_dir,
    track_names: List[str],
    *,
    coordinate_frame: str = "ego",
) -> Path:
    """Write the dataset-level manifest consumed by SideNet."""
    if coordinate_frame not in {"ego", "world"}:
        raise ValueError("coordinate_frame must be 'ego' or 'world'")

    root = Path(output_dir)
    manifest = {
        "schema_version": 1,
        "coordinate_frame": coordinate_frame,
        "side_semantics": "track_global",
        "format": "sidenet_v1",
        "tracks": [{"name": name} for name in track_names],
    }
    path = root / "dataset_manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path


def export_track_yaml(track_dict: dict, output_path: Path | str) -> None:
    """Write a track dict to YAML file (strips centerline to save space)."""
    import yaml
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in track_dict.items()}
    if "track" in out and "centerline" in out["track"]:
        out = {
            "track": {k: v for k, v in out["track"].items() if k != "centerline"}
        }
    with open(output_path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, allow_unicode=True)

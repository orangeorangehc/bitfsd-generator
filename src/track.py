"""Track loader: reads FSD competition YAML track files."""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


class TrackData:
    """Ground-truth track with left/right cone boundaries."""

    def __init__(self, name: str, data: dict):
        track = data.get("track", data)
        self.name = name
        self.version = track.get("version", "1.0")
        self.closed_loop = track.get("lanesFirstWithLastConnected", True)

        # start pose
        start = track.get("start", {})
        self.start_pos = np.array(start.get("position", [0, 0, 0]), dtype=np.float64)
        self.start_ori = np.array(start.get("orientation", [0, 0, 0]), dtype=np.float64)

        # left / right boundaries
        self.left = self._parse_cones(track.get("left", []))
        self.right = self._parse_cones(track.get("right", []))
        self.unknown = self._parse_cones(track.get("unknown", []))

        # Optional: true centerline from generator (stored in YAML)
        cl_raw = track.get("centerline", [])
        if cl_raw:
            self._centerline = np.array(cl_raw, dtype=np.float32)
        else:
            self._centerline = None

        # classification for OpenPCDet export
        self._class_map = {
            "blue": "Cone_Left",
            "yellow": "Cone_Right",
            "big-orange": "Cone",
            "small-orange": "Cone",
            "unknown": "Cone",
            "invisible": None,  # skip
        }

    @staticmethod
    def _parse_cones(cone_list: list) -> np.ndarray:
        """Convert cone list to (M, 4) array [x, y, z, class_str]."""
        if not cone_list:
            return np.empty((0, 4), dtype=object)
        rows = []
        for c in cone_list:
            pos = c["position"]
            rows.append([float(pos[0]), float(pos[1]), float(pos[2]), c.get("class", "unknown")])
        return np.array(rows, dtype=object)

    @property
    def left_xyz(self) -> np.ndarray:
        """(N, 3) float array of left cone positions."""
        if len(self.left) == 0:
            return np.empty((0, 3))
        return np.array(self.left[:, :3].tolist(), dtype=np.float32)

    @property
    def right_xyz(self) -> np.ndarray:
        """(N, 3) float array of right cone positions."""
        if len(self.right) == 0:
            return np.empty((0, 3))
        return np.array(self.right[:, :3].tolist(), dtype=np.float32)

    @property
    def all_cones(self) -> np.ndarray:
        """(N, 3) all cone positions."""
        l = self.left_xyz
        r = self.right_xyz
        if len(l) == 0 and len(r) == 0:
            return np.empty((0, 3))
        parts = [p for p in [l, r] if len(p) > 0]
        return np.concatenate(parts, axis=0)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "closed_loop": self.closed_loop,
            "left_cones": int(len(self.left)),
            "right_cones": int(len(self.right)),
            "unknown_cones": int(len(self.unknown)),
            "track_length_est_m": self.total_length,
        }

    def _estimate_length(self) -> float:
        """Track length from centerline or left boundary fallback."""
        return self.total_length

    @property
    def centerline(self) -> np.ndarray:
        """(M, 3) centerline: stored (generated) or estimated (loaded)."""
        if self._centerline is not None:
            return self._centerline
        return self._estimate_centerline()

    @property
    def width(self) -> float:
        """Average half-width: mean distance from each cone to the centerline."""
        left = self.left_xyz
        right = self.right_xyz
        cl = self.centerline
        if len(cl) == 0:
            return 3.0

        all_cones = []
        if len(left) > 0:
            all_cones.append(left)
        if len(right) > 0:
            all_cones.append(right)
        if not all_cones:
            return 3.0

        pts = np.concatenate(all_cones, axis=0)
        # For each cone, distance to nearest centerline point
        dists = np.linalg.norm(
            pts[:, None, :2] - cl[None, :, :2], axis=2
        ).min(axis=1)
        return float(2.0 * np.mean(dists))

    @property
    def total_length(self) -> float:
        """Centerline length (meters), preferring centerline over left boundary."""
        cl = self.centerline
        if len(cl) >= 2:
            return round(float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()), 1)
        return self._estimate_length()

    def _estimate_centerline(self) -> np.ndarray:
        """Estimate centerline as midpoint between matched left/right cones."""
        left = self.left_xyz
        right = self.right_xyz
        if len(left) == 0 and len(right) == 0:
            return np.empty((0, 3))
        if len(left) == 0:
            return right.copy()
        if len(right) == 0:
            return left.copy()

        n = min(len(left), len(right))
        result = (left[:n] + right[:n]) / 2.0
        return result.astype(np.float32)

    def __repr__(self):
        s = self.stats()
        return f"Track({s['name']}, {s['left_cones']}L/{s['right_cones']}R, ~{s['track_length_est_m']}m)"


def load_track(path: Path | str) -> TrackData:
    """Load a single track from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    name = Path(path).stem
    return TrackData(name, data)


def load_all_tracks(data_dir: Path | str) -> Dict[str, TrackData]:
    """Load all track YAMLs from directory."""
    tracks = {}
    for p in sorted(Path(data_dir).glob("*.yaml")):
        track = load_track(p)
        if len(track.left) > 0 or len(track.right) > 0:
            tracks[track.name] = track
    return tracks

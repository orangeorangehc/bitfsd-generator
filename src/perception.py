"""Perception simulation: vehicle poses, LiDAR filtering, noise injection."""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PerceptionConfig:
    ego_spacing: float = 2.0
    ego_noise_x: float = 0.1
    ego_noise_y: float = 0.1
    ego_noise_yaw_deg: float = 1.0
    lidar_range_min: float = 0.5
    lidar_range_max: float = 30.0
    lidar_fov_deg: float = 360.0  # total FoV angle
    position_noise_xy: float = 0.10
    position_noise_z: float = 0.03
    drop_rate: float = 0.10
    fp_rate: float = 0.05
    seed: int = 42


@dataclass
class PerceptionCone:
    x: float
    y: float
    z: float
    side: str  # "Left" | "Right"
    source: str  # "ground_truth" | "gaussian_noise" | "false_positive"
    original_gt_x: Optional[float] = None
    original_gt_y: Optional[float] = None
    original_gt_z: Optional[float] = None

    def to_list(self) -> list:
        base = [self.x, self.y, self.z, self.side, self.source]
        if self.source == "gaussian_noise":
            base.extend([
                self.original_gt_x or 0.0,
                self.original_gt_y or 0.0,
                self.original_gt_z or 0.0,
            ])
        return base


@dataclass
class PerceptionFrame:
    frame_id: int
    ego_pose: np.ndarray  # (4,) [x, y, z, yaw]
    cones: List[PerceptionCone] = field(default_factory=list)


@dataclass
class PerceptionSequence:
    track_name: str
    frames: List[PerceptionFrame]
    config: PerceptionConfig
    stats: dict = field(default_factory=dict)


class PerceptionPipeline:
    """Simulate perception on any TrackData object."""

    def __init__(self, config: Optional[PerceptionConfig] = None):
        self.config = config or PerceptionConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def run(self, track, flip: bool = False) -> PerceptionSequence:
        """Full pipeline: poses → filter → noise → frames.

        Args:
            track: TrackData object.
            flip: If True, drive the track in reverse direction. This generates
                  different ego poses (and therefore different visible cone subsets)
                  without swapping labels — cone labels remain track-global
                  (Cone_Left = left boundary, Cone_Right = right boundary).
        """
        left = track.left_xyz
        right = track.right_xyz
        centerline = track.centerline

        if flip:
            centerline = centerline[::-1].copy()
            # Labels are NOT swapped — they are track-global, not ego-relative.
            # The ego perspective is captured by ego_pose, not by cone labels.

        all_gt = np.concatenate([left, right], axis=0) if len(left) > 0 or len(right) > 0 else np.empty((0, 3))
        gt_sides = ["Left"] * len(left) + ["Right"] * len(right)

        poses = self._generate_poses(centerline)
        frames = []
        for frame_id, pose in enumerate(poses):
            visible = self._lidar_filter(pose, all_gt, gt_sides)
            noisy = self._inject_noise(pose, visible)
            frames.append(PerceptionFrame(
                frame_id=frame_id, ego_pose=pose, cones=noisy,
            ))

        stats = self._compute_stats(frames)
        return PerceptionSequence(
            track_name=track.name, frames=frames, config=self.config, stats=stats,
        )

    # ── vehicle poses ──────────────────────────────────────────────────

    def _generate_poses(self, centerline: np.ndarray) -> List[np.ndarray]:
        """Sample vehicle poses along centerline with noise.

        Uses central-difference for yaw (more accurate at curves and endpoints).
        """
        if len(centerline) < 2:
            return [np.array([0.0, 0.0, 0.0, 0.0])]

        # Compute arc-length sampled points
        seg_lens = np.linalg.norm(np.diff(centerline[:, :2], axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cumulative[-1]

        spacing = self.config.ego_spacing
        n_samples = max(1, int(total_len / spacing))
        target_dists = np.linspace(0, total_len, n_samples, endpoint=False)

        # Delta for central difference: 1m or 2% of track, whichever is smaller
        delta = min(1.0, total_len * 0.02)

        poses = []
        for dist in target_dists:
            pt = self._interpolate_at(centerline, cumulative, dist)

            # Central difference: interpolate ahead and behind, wrap around for closed loop
            dist_ahead = (dist + delta) % total_len
            dist_behind = (dist - delta) % total_len
            pt_ahead = self._interpolate_at(centerline, cumulative, dist_ahead)
            pt_behind = self._interpolate_at(centerline, cumulative, dist_behind)

            yaw = math.atan2(pt_ahead[1] - pt_behind[1], pt_ahead[0] - pt_behind[0])

            # Add noise
            noise_x = self.rng.normal(0, self.config.ego_noise_x)
            noise_y = self.rng.normal(0, self.config.ego_noise_y)
            noise_yaw = math.radians(self.rng.normal(0, self.config.ego_noise_yaw_deg))

            pose = np.array([
                pt[0] + noise_x,
                pt[1] + noise_y,
                pt[2],
                yaw + noise_yaw,
            ], dtype=np.float32)
            poses.append(pose)

        return poses

    @staticmethod
    def _interpolate_at(centerline: np.ndarray, cumulative: np.ndarray, dist: float) -> np.ndarray:
        """Linearly interpolate centerline at a given arc-length distance."""
        idx = int(np.searchsorted(cumulative, dist))
        idx = min(idx, len(centerline) - 1)
        if idx == 0:
            return centerline[0].copy()
        t0 = cumulative[idx - 1]
        t1 = cumulative[idx]
        frac = (dist - t0) / max(t1 - t0, 1e-8)
        frac = np.clip(frac, 0.0, 1.0)
        return centerline[idx - 1] * (1 - frac) + centerline[idx] * frac

    # ── LiDAR filter ────────────────────────────────────────────────────

    def _lidar_filter(
        self, pose: np.ndarray, cones: np.ndarray, sides: List[str]
    ) -> List[dict]:
        """Filter cones by LiDAR range and FoV. Returns visible cones in world frame."""
        if len(cones) == 0:
            return []

        px, py, pz, yaw = pose[0], pose[1], pose[2], pose[3]

        # Transform to ego frame
        dx = cones[:, 0] - px
        dy = cones[:, 1] - py
        dz = cones[:, 2] - pz

        # Rotate by -yaw
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        dx_local = dx * cos_yaw - dy * sin_yaw
        dy_local = dx * sin_yaw + dy * cos_yaw

        dist = np.sqrt(dx_local**2 + dy_local**2)
        angle = np.abs(np.arctan2(dy_local, dx_local))

        r_min = self.config.lidar_range_min
        r_max = self.config.lidar_range_max
        half_fov = math.radians(self.config.lidar_fov_deg / 2.0)

        mask = (dist >= r_min) & (dist <= r_max) & (angle <= half_fov)

        visible = []
        for i in np.where(mask)[0]:
            visible.append({
                "x": float(cones[i, 0]),
                "y": float(cones[i, 1]),
                "z": float(cones[i, 2]),
                "side": sides[i],
            })
        return visible

    # ── noise injection ─────────────────────────────────────────────────

    def _inject_noise(self, pose: np.ndarray, visible: List[dict]) -> List[PerceptionCone]:
        """Add position noise, dropouts, and false positives."""
        cones: List[PerceptionCone] = []

        for cone in visible:
            # Dropout
            if self.rng.random() < self.config.drop_rate:
                continue

            # Position jitter
            if self.config.position_noise_xy > 0 or self.config.position_noise_z > 0:
                nx = self.rng.normal(0, self.config.position_noise_xy)
                ny = self.rng.normal(0, self.config.position_noise_xy)
                nz = self.rng.normal(0, self.config.position_noise_z)
                cones.append(PerceptionCone(
                    x=cone["x"] + nx,
                    y=cone["y"] + ny,
                    z=cone["z"] + nz,
                    side=cone["side"],
                    source="gaussian_noise",
                    original_gt_x=cone["x"],
                    original_gt_y=cone["y"],
                    original_gt_z=cone["z"],
                ))
            else:
                cones.append(PerceptionCone(
                    x=cone["x"], y=cone["y"], z=cone["z"],
                    side=cone["side"], source="ground_truth",
                ))

        # False positives
        if self.config.fp_rate > 0 and len(cones) > 0:
            n_fp = max(1, int(len(cones) * self.config.fp_rate))
            # Sample from bounding box of detected cones
            detected = np.array([[c.x, c.y, c.z] for c in cones])
            mins = detected.min(axis=0) - 3.0
            maxs = detected.max(axis=0) + 3.0
            for _ in range(n_fp):
                fx = self.rng.uniform(mins[0], maxs[0])
                fy = self.rng.uniform(mins[1], maxs[1])
                fz = self.rng.uniform(mins[2], maxs[2])
                # Re-filter by LiDAR range/FoV so FP appears "visible"
                dx = fx - pose[0]
                dy = fy - pose[1]
                cos_yaw = math.cos(-pose[3])
                sin_yaw = math.sin(-pose[3])
                dx_local = dx * cos_yaw - dy * sin_yaw
                dy_local = dx * sin_yaw + dy * cos_yaw
                if math.sqrt(dx_local**2 + dy_local**2) > self.config.lidar_range_max:
                    continue
                side = "Left" if self.rng.random() < 0.5 else "Right"
                cones.append(PerceptionCone(
                    x=fx, y=fy, z=fz, side=side, source="false_positive",
                ))

        return cones

    # ── stats ───────────────────────────────────────────────────────────

    def _compute_stats(self, frames: List[PerceptionFrame]) -> dict:
        gt_count = 0
        gn_count = 0
        fp_count = 0
        for f in frames:
            for c in f.cones:
                if c.source == "ground_truth":
                    gt_count += 1
                elif c.source == "gaussian_noise":
                    gn_count += 1
                elif c.source == "false_positive":
                    fp_count += 1
        total = gt_count + gn_count + fp_count
        return {
            "num_frames": len(frames),
            "total_cones": total,
            "ground_truth_count": gt_count,
            "gaussian_noise_count": gn_count,
            "false_positive_count": fp_count,
        }

"""Synthetic closed-loop track generator from segment-based presets."""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrackPreset:
    name: str
    segments: list  # list of dict: {type: straight|curve, length?, radius?, angle?}
    track_width: float = 4.0
    cone_spacing_min: float = 3.0
    cone_spacing_max: float = 5.0
    centerline_resolution: float = 0.2  # meters between centerline sample points


class TrackGenerator:
    """Generate a closed-loop track from a segment-based preset."""

    def __init__(self, preset: TrackPreset, seed: int = 42):
        self.preset = preset
        self.rng = np.random.default_rng(seed)

    def generate(self) -> dict:
        """Build track and return YAML-compatible dict."""
        centerline = self._build_centerline()
        left_boundary, right_boundary = self._offset_boundaries(centerline)
        left_cones, right_cones = self._place_cones(
            centerline, left_boundary, right_boundary
        )
        return self._build_track_dict(left_cones, right_cones, centerline)

    # ── centerline ─────────────────────────────────────────────────────

    def _build_centerline(self) -> np.ndarray:
        """Walk through segments producing a dense (M, 3) centerline polyline."""
        resolution = self.preset.centerline_resolution
        points = [(0.0, 0.0)]
        heading = 0.0  # radians, 0 = +x

        for seg in self.preset.segments:
            seg_type = seg["type"]
            if seg_type == "straight":
                length = seg["length"]
                n_pts = max(2, int(length / resolution))
                start_x, start_y = points[-1]
                for i in range(1, n_pts + 1):
                    t = i / n_pts * length
                    x = start_x + t * math.cos(heading)
                    y = start_y + t * math.sin(heading)
                    points.append((x, y))
            elif seg_type == "curve":
                radius = abs(seg["radius"])
                angle_deg = seg["angle"]
                angle_rad = math.radians(angle_deg)
                direction = 1 if angle_deg > 0 else -1  # +1=left, -1=right

                # Arc center is perpendicular to heading at distance radius
                cx = points[-1][0] - direction * radius * math.sin(heading)
                cy = points[-1][1] + direction * radius * math.cos(heading)

                # Start angle (pointing from center to current position)
                start_angle = math.atan2(
                    points[-1][1] - cy, points[-1][0] - cx
                )

                arc_len = radius * abs(angle_rad)
                n_pts = max(4, int(arc_len / resolution))
                for i in range(1, n_pts + 1):
                    frac = i / n_pts
                    a = start_angle + direction * frac * abs(angle_rad)
                    x = cx + radius * math.cos(a)
                    y = cy + radius * math.sin(a)
                    points.append((x, y))

                heading += direction * abs(angle_rad)

        # Close loop
        pts = np.array(points, dtype=np.float32)
        gap = np.linalg.norm(pts[-1] - pts[0])
        if gap > 0.1:
            n_close = max(2, int(gap / resolution))
            start = pts[-1].copy()
            direction = pts[0] - pts[-1]
            for i in range(1, n_close):
                t = i / n_close
                pt = start + t * direction
                pts = np.vstack([pts, pt.reshape(1, 2)])

        # Ensure closed
        pts = np.vstack([pts, pts[0:1]])
        z = np.zeros((len(pts), 1), dtype=np.float32)
        return np.concatenate([pts, z], axis=1)

    # ── boundaries ─────────────────────────────────────────────────────

    def _offset_boundaries(
        self, centerline: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Offset centerline by +/- width/2 along normals."""
        hw = self.preset.track_width / 2.0
        tangents = np.diff(centerline[:, :2], axis=0)
        tangents = np.vstack([tangents, tangents[-1:]])
        lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
        lengths[lengths < 1e-8] = 1e-8
        tangents = tangents / lengths

        # Normal: rotate tangent 90° counterclockwise
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]

        left_boundary = centerline.copy()
        left_boundary[:, :2] += normals * hw
        right_boundary = centerline.copy()
        right_boundary[:, :2] -= normals * hw
        return left_boundary, right_boundary

    # ── cone placement ─────────────────────────────────────────────────

    def _place_cones(
        self,
        centerline: np.ndarray,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Place cones along boundaries with adaptive spacing."""
        curvatures = self._compute_curvature(centerline)
        spacings = self._curvature_to_spacing(curvatures)

        left_cones = self._walk_boundary(left_boundary, spacings)
        right_cones = self._walk_boundary(right_boundary, spacings)
        return left_cones, right_cones

    def _compute_curvature(self, centerline: np.ndarray) -> np.ndarray:
        """Three-point curvature at each centerline point."""
        pts = centerline[:, :2]
        n = len(pts)
        k = np.zeros(n)
        for i in range(1, n - 1):
            a = pts[i - 1]
            b = pts[i]
            c = pts[i + 1]
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ac = np.linalg.norm(c - a)
            if ab < 1e-8 or bc < 1e-8 or ac < 1e-8:
                continue
            # Curvature = 4 * area / (ab * bc * ac)
            area = 0.5 * abs(
                a[0] * (b[1] - c[1])
                + b[0] * (c[1] - a[1])
                + c[0] * (a[1] - b[1])
            )
            k[i] = 4.0 * area / (ab * bc * ac)

        # Smooth with a 5-point window
        if n >= 5:
            kernel = np.ones(5) / 5.0
            k = np.convolve(k, kernel, mode="same")
        return k

    def _curvature_to_spacing(self, curvature: np.ndarray) -> np.ndarray:
        """Map curvature to cone spacing: high curvature → tight spacing."""
        smin = self.preset.cone_spacing_min
        smax = self.preset.cone_spacing_max
        kmax = np.percentile(curvature, 90)
        if kmax < 1e-8:
            return np.full_like(curvature, smax)
        ratio = np.clip(curvature / kmax, 0.0, 1.0)
        return smax - (smax - smin) * ratio

    def _walk_boundary(
        self, boundary: np.ndarray, spacings: np.ndarray
    ) -> np.ndarray:
        """Walk along boundary and place cones at adaptive spacing intervals."""
        if len(boundary) < 2:
            return np.empty((0, 3), dtype=np.float32)

        cones = [boundary[0].copy()]
        accumulated = 0.0

        for i in range(1, len(boundary)):
            seg_len = float(np.linalg.norm(boundary[i] - boundary[i - 1]))
            if seg_len < 1e-8:
                continue

            direction = (boundary[i] - boundary[i - 1]) / seg_len

            # Interpolate spacing for this segment
            spacing = float(spacings[min(i, len(spacings) - 1)])
            while accumulated + seg_len >= spacing and spacing > 0:
                # Place cone at interpolated position
                remaining = spacing - accumulated
                t = remaining / seg_len
                cone_pt = boundary[i - 1] + t * (boundary[i] - boundary[i - 1])
                cones.append(cone_pt.copy())
                accumulated = -remaining
            accumulated += seg_len

        return np.array(cones, dtype=np.float32)

    # ── output ──────────────────────────────────────────────────────────

    def _build_track_dict(
        self,
        left_cones: np.ndarray,
        right_cones: np.ndarray,
        centerline: np.ndarray,
    ) -> dict:
        """Assemble YAML-compatible track dict."""
        left_list = [
            {"position": [float(c[0]), float(c[1]), float(c[2])], "class": "blue"}
            for c in left_cones
        ]
        right_list = [
            {"position": [float(c[0]), float(c[1]), float(c[2])], "class": "yellow"}
            for c in right_cones
        ]

        return {
            "track": {
                "version": "1.0",
                "lanesFirstWithLastConnected": True,
                "start": {
                    "position": [float(centerline[0, 0]), float(centerline[0, 1]), 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                },
                "earthToTrack": {
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0],
                },
                "centerline": [
                    [float(p[0]), float(p[1]), float(p[2])] for p in centerline
                ],
                "left": left_list,
                "right": right_list,
                "unknown": [],
            }
        }

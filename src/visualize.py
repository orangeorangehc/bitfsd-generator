"""Track visualizer: plots GT cones and simulated perception output."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Optional

from track import TrackData


def plot_track(track: TrackData, ax: Optional[plt.Axes] = None,
               title: str = None, show_start: bool = True,
               show_grid: bool = True):
    """Plot track ground-truth: left (blue), right (red), start (green arrow)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 10))

    left = track.left_xyz
    right = track.right_xyz

    # Boundaries
    for pts, color, label in [(left, 'royalblue', 'Left'), (right, 'crimson', 'Right')]:
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=15, alpha=0.8, label=label, zorder=3)
            # Connect adjacent cones to show track shape
            ax.plot(pts[:, 0], pts[:, 1], c=color, linewidth=1.5, alpha=0.4, zorder=2)

    # Unknown cones
    unk = track.unknown
    if len(unk) > 0:
        uk_xyz = np.array(unk[:, :3].tolist(), dtype=np.float32)
        ax.scatter(uk_xyz[:, 0], uk_xyz[:, 1], c='grey', s=10, alpha=0.5, marker='x', label='Unknown')

    # Start position
    if show_start:
        ax.arrow(track.start_pos[0], track.start_pos[1],
                 2.0, 0, head_width=1.5, head_length=2.0, fc='green', ec='green', lw=2, zorder=5)
        ax.scatter(track.start_pos[0], track.start_pos[1], c='green', s=60, marker='*', zorder=5, label='Start')

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title or f'{track.name}  —  {track.stats()["track_length_est_m"]}m')
    if show_grid:
        ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    return ax


def plot_perception_overlay(track: TrackData, perception: np.ndarray,
                             ax: Optional[plt.Axes] = None, title: str = None):
    """
    Overlay simulated perception output on track GT.

    Args:
        track: ground truth track
        perception: (N, 4) array [x, y, z, side] where side=0 for Left, 1 for Right
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 10))

    # GT in light colors
    left = track.left_xyz
    right = track.right_xyz
    for pts, color in [(left, 'lightblue'), (right, 'lightcoral')]:
        if len(pts) > 0:
            ax.plot(pts[:, 0], pts[:, 1], c=color, linewidth=1, alpha=0.5, zorder=1)

    # Perception cones
    if perception.shape[0] > 0:
        is_left = perception[:, 3] == 0
        is_right = perception[:, 3] == 1

        ax.scatter(perception[is_left, 0], perception[is_left, 1],
                   c='blue', s=30, edgecolors='darkblue', linewidth=0.5,
                   label=f'Perception Left ({is_left.sum()})', zorder=4)
        ax.scatter(perception[is_right, 0], perception[is_right, 1],
                   c='red', s=30, edgecolors='darkred', linewidth=0.5,
                   label=f'Perception Right ({is_right.sum()})', zorder=4)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title or f'{track.name} — GT + Perception')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    return ax


def save_track_image(track: TrackData, output_path: Path, perception: np.ndarray = None):
    """Save track visualization to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    plot_track(track, ax=ax1, title=f'{track.name} — Ground Truth')
    if perception is not None:
        plot_perception_overlay(track, perception, ax=ax2,
                                title=f'{track.name} — Simulated Perception')
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def show_track(track: TrackData, perception: np.ndarray = None):
    """Interactive display."""
    if perception is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        plot_track(track, ax=ax1, title=f'{track.name} — GT')
        plot_perception_overlay(track, perception, ax=ax2, title='Simulated Perception')
    else:
        fig, ax1 = plt.subplots(figsize=(12, 10))
        plot_track(track, ax=ax1)
    fig.tight_layout()
    plt.show()


def plot_perception_frame(
    track: TrackData, frame, ax: Optional[plt.Axes] = None, title: str = None
):
    """Plot a single PerceptionFrame with source-based color coding.

    - green: ground_truth
    - blue/red: gaussian_noise (per side)
    - orange: false_positive
    - yellow star: ego pose
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 10))

    # GT in light colors
    left = track.left_xyz
    right = track.right_xyz
    for pts, color in [(left, 'lightblue'), (right, 'lightcoral')]:
        if len(pts) > 0:
            ax.plot(pts[:, 0], pts[:, 1], c=color, linewidth=1, alpha=0.4, zorder=1)

    # Centerline
    cl = track.centerline
    if len(cl) > 1:
        ax.plot(cl[:, 0], cl[:, 1], c='grey', linewidth=0.5, alpha=0.3, linestyle='--', zorder=1)

    for cone in frame.cones:
        if cone.source == "false_positive":
            color, marker, size = "orange", "x", 25
        elif cone.source == "gaussian_noise":
            color = "#44ccff" if cone.side == "Left" else "#ff4444"
            marker, size = "o", 20
        else:
            color = "#228844" if cone.side == "Left" else "#882222"
            marker, size = "o", 20
        ax.scatter(cone.x, cone.y, c=color, s=size, marker=marker, zorder=4, edgecolors='k', linewidth=0.3)

    # Ego pose
    pose = frame.ego_pose
    ax.scatter(pose[0], pose[1], c='yellow', s=80, marker='*', zorder=5, label='Ego')
    ax.arrow(pose[0], pose[1], 3 * np.cos(pose[3]), 3 * np.sin(pose[3]),
             head_width=1.0, head_length=1.5, fc='yellow', ec='yellow', lw=1.5, zorder=5)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title or f'{track.name} — Frame {frame.frame_id}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    return ax

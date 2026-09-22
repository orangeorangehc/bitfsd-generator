#!/usr/bin/env python3
"""Visualize SideNet training data as a grid of frames in a single PNG.

Usage:
    uv run python visualize_data.py output/sidenet_data/FSE22
    uv run python visualize_data.py output/sidenet_data/FSE22 --rows 3 --cols 4
    uv run python visualize_data.py output/sidenet_data/FSE22 --start 20
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLOR_LEFT = "#4488ff"
COLOR_RIGHT = "#ee5533"


def load_cloud(path: Path):
    """Load a cloud_N.txt file. Returns (points, labels)."""
    points, labels = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            x, y = float(parts[0]), float(parts[1])
            cls = parts[-1]
            side = 0 if "Left" in cls else 1
            points.append((x, y))
            labels.append(side)
    if not points:
        return np.empty((0, 2)), np.empty(0, dtype=int)
    return np.array(points), np.array(labels, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Visualize SideNet training data")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        print(f"Not a directory: {data_dir}")
        sys.exit(1)

    cloud_files = sorted(data_dir.glob("cloud_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    if not cloud_files:
        print(f"No cloud_*.txt files in {data_dir}")
        sys.exit(1)

    total = len(cloud_files)
    end = min(args.start + args.rows * args.cols, total)
    selected = cloud_files[args.start:end]

    rows, cols = args.rows, args.cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    fig.suptitle(f"{data_dir.name}  (frames {args.start}-{end-1} of {total})", fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(selected):
            ax.set_visible(False)
            continue

        pts, labels = load_cloud(selected[idx])
        frame_id = selected[idx].stem.split("_")[1]

        if len(pts) > 0:
            left_mask = labels == 0
            right_mask = labels == 1
            ax.scatter(pts[left_mask, 0], pts[left_mask, 1], c=COLOR_LEFT, s=10, label="Left")
            ax.scatter(pts[right_mask, 0], pts[right_mask, 1], c=COLOR_RIGHT, s=10, label="Right")
            ax.set_aspect("equal")
            ax.axhline(0, color="#444", linewidth=0.3)
            ax.axvline(0, color="#444", linewidth=0.3)
            ax.plot(0, 0, "w+", markersize=6, markeredgewidth=1.2)

        ax.set_title(f"frame {frame_id} ({len(pts)} cones)", fontsize=8)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out_path = args.output or f"{data_dir.name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

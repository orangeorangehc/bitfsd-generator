# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: BITFSD Generator

Synthetic FSD (Formula Student Driverless) racing-track + perception data platform. Produces cone-level outputs (positions + Left/Right side labels) for perception-to-planning closed-loop testing. Output is compatible with OpenPCDet `CustomDataset` and the sibling `bitfsd-annotator` tool.

Track-geometry constraints come from `FSD_RULE.md` (closed loop 200–500 m, min width 3 m, hairpin outer radius ≥ 9 m, cone spacing 5–15 m tighter on curves).

## Commands

Python 3.12 + `uv`. The project has no test suite — validation is by running the CLI and inspecting the web UI.

```bash
uv sync                                          # install (proxy may be needed: export http_proxy=http://127.0.0.1:7890)

./start.sh serve [PORT]                          # web UI at http://localhost:8001
./start.sh generate <preset> [--seed N]          # synth a track from config/track_presets.yaml
./start.sh perceive <track.yaml> [opts]          # run perception sim, write per-frame label files
./start.sh presets                               # list available presets

uv run python main.py {generate,perceive,serve} ...   # equivalent without the wrapper
uv run python src/server.py --port 8001               # direct server entry
```

Smoke check after changes: `./start.sh generate simple_oval` (writes `output/simple_oval_s42.yaml`) then load it in the web UI.

## Architecture

Two independent inputs feed the same downstream pipeline:

```
config/track_presets.yaml ──┐
                            ├──► TrackData ──► PerceptionPipeline ──► export.py ──► .txt frames
data/*.yaml (real FSD)   ──┘    (track.py)    (perception.py)
```

Real-world tracks (`data/FSE22.yaml`, `FSG23.yaml`, …) and synthetic tracks both load into the same `TrackData` object, so all downstream code (perception, export, web UI, visualizer) is source-agnostic.

### TrackGenerator (`src/track_generator.py`)

Segment-based, not Bezier/B-spline (the old top-level diagram is misleading):

1. Walk a list of `{type: straight, length}` / `{type: curve, radius, angle}` segments to produce a dense polyline centerline at `centerline_resolution` (default 0.2 m).
2. Auto-close the loop by linearly interpolating from last point back to first.
3. Offset the centerline ±`track_width/2` along the normal to get left/right boundaries.
4. Place cones along each boundary with **curvature-adaptive spacing**: 3-point curvature is computed at every centerline sample, smoothed with a 5-point box kernel, then mapped linearly to a spacing in `[cone_spacing_min, cone_spacing_max]` — high curvature → tight spacing.

The output dict matches the on-disk YAML schema (left/right cones tagged `blue`/`yellow`).

### TrackData class invariant (`src/track.py`)

`TrackData` accepts both dict-as-loaded and the raw `{"track": {...}}` wrapper. Cone class strings are mapped to OpenPCDet sides via `_class_map`:

| YAML `class`              | side          |
|---------------------------|---------------|
| `blue`                    | `Cone_Left`   |
| `yellow`                  | `Cone_Right`  |
| `big-orange`/`small-orange`/`unknown` | `Cone` |
| `invisible`               | dropped       |

Centerline handling has two paths: synthetic tracks store the generator's exact centerline in YAML; real-world tracks omit it and `_estimate_centerline` falls back to midpoint-of-matched-pairs (correct only when left/right cone counts are similar). `total_length` and `width` derive from whichever centerline is available. `export_track_yaml` strips the centerline before writing to keep YAML small.

### PerceptionPipeline (`src/perception.py`)

Three-stage pipeline run per track. Configured via `PerceptionConfig`:

1. **Pose generation** — sample ego positions along the arc-length-parameterized centerline at `ego_spacing`, compute yaw from local tangent, add Gaussian noise to (x, y, yaw).
2. **LiDAR filter** — for each pose, transform all GT cones into ego frame and keep only those in `[range_min, range_max]` and within `±lidar_fov_deg/2`.
3. **Noise injection** — apply per-cone dropout (`drop_rate`), per-cone Gaussian XY/Z jitter, then add false positives sampled from the bounding box of currently-detected cones (`fp_rate × N`). False positives are re-filtered by LiDAR range so they remain "plausible".

Each output `PerceptionCone` carries a `source` tag (`ground_truth` / `gaussian_noise` / `false_positive`); when the source is `gaussian_noise` the original GT position is preserved alongside the noisy one. The web UI and exporters key visualization color and output columns off this tag.

### Web server (`src/server.py`)

FastAPI with the entire frontend embedded in a single `HTML_PAGE` string (Canvas 2D, no build step). Exposes:

- `GET /api/tracks`, `GET /api/tracks/{name}` — list/fetch (real + generated)
- `POST /api/tracks/{name}/perceive` — has two modes:
  - `mode=simple` — flat list, no LiDAR filter, server-side noise. Used by sliders for live preview.
  - `mode=full` — runs the real `PerceptionPipeline` and returns multi-frame data. Arrow keys in the UI step through frames.
- `POST /api/generate` — instantiates `TrackPreset` from YAML and stores the result in the in-memory `GENERATED` dict (not persisted to disk by this endpoint — the CLI's `generate` subcommand is what writes `output/*.yaml`).
- `POST /api/export` — converts cone JSON to OpenPCDet or perception text format.

Real tracks are loaded once into `TRACKS` on first request (`get_tracks()` lazy-init); restart the server to pick up new files in `data/`.

### Output text formats (`src/export.py`)

Two distinct formats; do not conflate:

```
# OpenPCDet (cones_to_openpcdet) — fixed dx/dy/dz/heading
x y z 0.200 0.200 0.300 0.000 {Cone_Left|Cone_Right|Cone}

# Perception (perception_frame_to_text) — adds source provenance
x y z {Left|Right} {ground_truth|gaussian_noise|false_positive} [orig_x orig_y orig_z]
```

The trailing `orig_xyz` columns appear only for `gaussian_noise` rows.

## Notes

- `src/sidenet.py` is a standalone Transformer for left/right cone classification. It is not wired into `main.py` or `server.py` — treat it as a separate experiment.
- `data/skidpad.yaml`, `data/acceleration.yaml`, and `data/gripMap.yaml` are non-loop FSD events; `load_all_tracks` will still load them but `closed_loop` semantics in stats may be misleading.
- Adding a new preset = edit `config/track_presets.yaml`. Adding a new segment type = extend the `seg_type` branch in `TrackGenerator._build_centerline` and update `_compute_curvature` if the new shape needs special handling.
- When tracing a perception bug, follow `batch_dict`-style state through `PerceptionPipeline.run` → `_generate_poses` → `_lidar_filter` → `_inject_noise`; each stage adds info to a frame, none mutate prior stages.

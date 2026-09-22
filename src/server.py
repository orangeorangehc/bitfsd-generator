"""Web server for track visualization and generation. Run: python server.py --port 8001"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from track import load_track, load_all_tracks, TrackData
from perception import PerceptionPipeline, PerceptionConfig
from export import frame_to_sidenet
from track_generator import TrackGenerator, TrackPreset
import json
import argparse
import yaml
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
CONFIG_DIR = Path(__file__).parent.parent / "config"
TRACKS = {}
GENERATED = {}


def get_tracks():
    global TRACKS
    if not TRACKS:
        TRACKS = load_all_tracks(DATA_DIR)
    return TRACKS


def load_presets() -> dict:
    path = CONFIG_DIR / "track_presets.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {"presets": {}, "defaults": {}}


def create_app():
    from fastapi import FastAPI, Query, Body
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="BITFSD Track Generator & Visualizer")

    # ── Track listing ──────────────────────────────────────────────────

    @app.get("/api/tracks")
    async def list_tracks():
        tracks = get_tracks()
        result = [t.stats() for t in tracks.values()]
        for name, t in GENERATED.items():
            result.append(t.stats())
        return result

    @app.get("/api/tracks/{name}")
    async def get_track(name: str):
        tracks = get_tracks()
        t = tracks.get(name) or GENERATED.get(name)
        if t is None:
            return JSONResponse({"error": f"Track '{name}' not found"}, 404)

        def cone_to_dict(row):
            return {"x": float(row[0]), "y": float(row[1]), "z": float(row[2]), "class": str(row[3])}

        return JSONResponse({
            "name": t.name,
            "closed_loop": t.closed_loop,
            "start": [float(v) for v in t.start_pos],
            "left": [cone_to_dict(r) for r in t.left] if len(t.left) > 0 else [],
            "right": [cone_to_dict(r) for r in t.right] if len(t.right) > 0 else [],
            "unknown": [cone_to_dict(r) for r in t.unknown] if len(t.unknown) > 0 else [],
            "stats": t.stats(),
        })

    # ── Perception (full pipeline) ─────────────────────────────────────

    @app.post("/api/tracks/{name}/perceive")
    async def simulate_perception(
        name: str,
        noise_xy: float = Query(0.10, description="Position noise sigma (m)"),
        drop_rate: float = Query(0.10, description="Dropout rate"),
        fp_rate: float = Query(0.05, description="False positive rate"),
        lidar_range: float = Query(30.0, description="LiDAR max range (m)"),
        lidar_fov: float = Query(360.0, description="LiDAR total FoV (degrees)"),
        ego_spacing: float = Query(5.0, description="Ego pose spacing (m)"),
        mode: str = Query("simple", description="'simple' = flat list, 'full' = LiDAR-filtered sequence"),
        flip: bool = Query(False, description="Reverse driving direction (data augmentation)"),
    ):
        tracks = get_tracks()
        t = tracks.get(name) or GENERATED.get(name)
        if t is None:
            return JSONResponse({"error": f"Track '{name}' not found"}, 404)

        if mode == "simple":
            # Flat list: all cones with noise, no LiDAR filtering (for visualization)
            rng = np.random.RandomState(42)
            perception = []
            # flip: swap labels so yellow→Left, blue→Right from reversed ego's perspective
            left_cones, right_cones = (t.right, t.left) if flip else (t.left, t.right)
            for side, cones in [("Left", left_cones), ("Right", right_cones)]:
                for row in cones:
                    x, y, z = float(row[0]), float(row[1]), float(row[2])
                    if rng.rand() < drop_rate:
                        continue
                    x += rng.randn() * noise_xy
                    y += rng.randn() * noise_xy
                    perception.append({
                        "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                        "side": side, "score": round(0.85 + rng.rand() * 0.15, 3),
                        "source": "gaussian_noise" if noise_xy > 0 else "ground_truth",
                    })

            all_x = [c["x"] for c in perception] if perception else [0.0]
            all_y = [c["y"] for c in perception] if perception else [0.0]
            n_fp = int(len(perception) * fp_rate)
            for _ in range(n_fp):
                perception.append({
                    "x": round(rng.uniform(min(all_x) - 5, max(all_x) + 5), 3),
                    "y": round(rng.uniform(min(all_y) - 5, max(all_y) + 5), 3),
                    "z": -1.0,
                    "side": "Left" if rng.rand() < 0.5 else "Right",
                    "score": round(0.3 + rng.rand() * 0.3, 3),
                    "source": "false_positive",
                })

            return {
                "track": name,
                "mode": "simple",
                "params": {"noise_xy": noise_xy, "drop_rate": drop_rate, "fp_rate": fp_rate},
                "num_cones": len(perception),
                "cones": perception,
            }

        # Full pipeline mode
        config = PerceptionConfig(
            ego_spacing=ego_spacing,
            position_noise_xy=noise_xy,
            drop_rate=drop_rate,
            fp_rate=fp_rate,
            lidar_range_max=lidar_range,
            lidar_fov_deg=lidar_fov,
            seed=42,
        )
        pipeline = PerceptionPipeline(config)
        sequence = pipeline.run(t, flip=flip)

        frames_out = []
        for frame in sequence.frames:
            frames_out.append({
                "frame_id": frame.frame_id,
                "ego_pose": [float(v) for v in frame.ego_pose],
                "cones": [c.to_list() for c in frame.cones],
            })

        return {
            "track": name,
            "mode": "full",
            "params": {
                "noise_xy": noise_xy, "drop_rate": drop_rate, "fp_rate": fp_rate,
                "lidar_range": lidar_range, "lidar_fov": lidar_fov, "ego_spacing": ego_spacing,
            },
            "num_frames": len(frames_out),
            "frames": frames_out,
            "stats": sequence.stats,
        }

    # ── Export ──────────────────────────────────────────────────────────

    @app.post("/api/export")
    async def export_labels(request: dict):
        """Export perception cones in SideNet training format."""
        cones = request.get("cones", [])
        lines = []
        for c in cones:
            side = c.get("side", "")
            source = c.get("source", "ground_truth")
            if source == "false_positive":
                continue
            cls = "Cone_Left" if "Left" in side else "Cone_Right"
            lines.append(
                f"{c['x']:.4f} {c['y']:.4f} {c['z']:.4f} "
                f"0.200 0.200 0.300 0.000 1.0 {cls}"
            )
        text = "\n".join(lines)
        return {"text": text, "num_lines": len(lines)}

    @app.post("/api/export_batch")
    async def export_batch(request: dict):
        """Export approved frames as a zip of SideNet cloud_N.txt files."""
        import zipfile, io
        frames = request.get("frames", [])
        track_name = request.get("track_name", "track")
        buf = io.BytesIO()
        idx = 0
        with zipfile.ZipFile(buf, "w") as zf:
            for frame in frames:
                if frame.get("status") != "approved":
                    continue
                lines = []
                for c in frame.get("cones", []):
                    side = c.get("side", "")
                    source = c.get("source", "ground_truth")
                    if source == "false_positive":
                        continue
                    cls = "Cone_Left" if "Left" in side else "Cone_Right"
                    lines.append(
                        f"{c['x']:.4f} {c['y']:.4f} {c['z']:.4f} "
                        f"0.200 0.200 0.300 0.000 1.0 {cls}"
                    )
                zf.writestr(f"{track_name}/cloud_{idx}.txt", "\n".join(lines))
                idx += 1
        buf.seek(0)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={track_name}_sidenet.zip"},
        )

    # ── Track generation ───────────────────────────────────────────────

    @app.get("/api/presets")
    async def list_presets():
        data = load_presets()
        result = {}
        for name, preset in data.get("presets", {}).items():
            result[name] = {
                "description": preset.get("description", ""),
                "track_width": preset.get("track_width", data["defaults"].get("track_width", 4.0)),
                "num_segments": len(preset.get("segments", [])),
            }
        return {"presets": result, "defaults": data.get("defaults", {})}

    @app.post("/api/generate")
    async def generate_track(request: dict = Body(...)):
        """Generate a track from a preset. Body: {"preset": "simple_oval", "seed": 42}."""
        preset_name = request.get("preset", "simple_oval")
        seed = request.get("seed", 42)

        data = load_presets()
        defaults = data.get("defaults", {})
        preset_dict = data.get("presets", {}).get(preset_name)
        if preset_dict is None:
            return JSONResponse({"error": f"Preset '{preset_name}' not found"}, 404)

        preset = TrackPreset(
            name=preset_name,
            segments=preset_dict["segments"],
            track_width=preset_dict.get("track_width", defaults.get("track_width", 4.0)),
            cone_spacing_min=preset_dict.get("cone_spacing_min", defaults.get("cone_spacing_min", 3.0)),
            cone_spacing_max=preset_dict.get("cone_spacing_max", defaults.get("cone_spacing_max", 5.0)),
        )
        gen = TrackGenerator(preset, seed=seed)
        track_dict = gen.generate()

        # Store generated track as TrackData for API compatibility
        name = f"{preset_name}_s{seed}"
        track_data = TrackData(name, track_dict)
        GENERATED[name] = track_data

        return JSONResponse({
            "name": name,
            "stats": track_data.stats(),
        })

    # ── Static frontend ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(HTML_PAGE)

    return app


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BITFSD Track Generator & Visualizer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: monospace; background: #0a0a0f; color: #ccc; display: flex; height: 100vh; }
#sidebar { width: 320px; padding: 16px; background: #111118; overflow-y: auto; border-right: 1px solid #222; }
#sidebar h2 { font-size: 15px; margin-bottom: 6px; color: #6af; }
#sidebar h3 { font-size: 12px; margin: 12px 0 4px; color: #888; border-top: 1px solid #222; padding-top: 8px; }
#sidebar select, #sidebar input, #sidebar button { width: 100%; margin: 4px 0; padding: 6px; background: #1a1a28; color: #ccc; border: 1px solid #333; border-radius: 4px; }
#sidebar button { cursor: pointer; background: #224; margin-top: 6px; }
#sidebar button:hover { background: #336; }
#sidebar button.primary { background: #242; }
#sidebar button.primary:hover { background: #363; }
#sidebar button.btn-approve { background: #252; }
#sidebar button.btn-approve:hover { background: #373; }
#sidebar button.btn-reject { background: #422; }
#sidebar button.btn-reject:hover { background: #633; }
#sidebar button.btn-reset { background: #333; }
#sidebar button.btn-reset:hover { background: #444; }
#sidebar .slider-group { margin: 8px 0; }
#sidebar label { font-size: 11px; color: #888; display: block; margin-top: 4px; }
#sidebar .slider-group span { float: right; color: #6af; }
#sidebar .qc-row { display: flex; gap: 4px; margin-top: 4px; }
#sidebar .qc-row button { margin-top: 0; }
#preview { background: #0d0d14; border: 1px solid #222; border-radius: 4px; padding: 6px; margin-top: 6px; font-size: 10px; color: #8a8; max-height: 180px; overflow-y: auto; white-space: pre; line-height: 1.4; }
#main { flex: 1; position: relative; }
canvas { display: block; }
#info { position: absolute; bottom: 8px; left: 8px; color: #888; font-size: 11px; }
</style>
</head>
<body>
<div id="sidebar">
  <h2>B��ITFSD Generator</h2>

  <h3>Load / Generate Track</h3>
  <select id="trackSelect"><option>-- Loading --</option></select>
  <button onclick="loadTrack()">Load Track</button>

  <div style="display:flex;gap:4px;">
    <select id="presetSelect" style="flex:1;"></select>
    <button onclick="generateTrack()" class="primary" style="flex:1;">Generate</button>
  </div>
  <div style="display:flex;gap:4px;margin-top:2px;">
    <label style="font-size:10px;">Seed:</label>
    <input id="genSeed" type="number" value="42" style="width:60px;padding:2px 4px;font-size:11px;">
  </div>

  <h3>Tools</h3>
  <div style="display:flex;gap:4px;">
    <button id="btnMeasure" onclick="toggleMeasure()" style="flex:1;font-size:11px;">Measure</button>
    <button id="btnGrid" onclick="toggleGrid()" style="flex:1;font-size:11px;">Grid: ON</button>
  </div>
  <div style="display:flex;gap:4px;margin-top:4px;align-items:center;">
    <label style="font-size:10px;white-space:nowrap;">Cell:</label>
    <input id="gridSize" type="number" value="5" min="1" max="50" step="1" onchange="gridSize=parseFloat(this.value)||5;draw();" style="width:50px;padding:2px 4px;font-size:11px;">
    <span style="font-size:10px;color:#666;">m</span>
    <span id="measureInfo" style="font-size:10px;color:#ff0;margin-left:8px;display:none;"></span>
  </div>

  <h3>Perception Simulation</h3>
  <div class="slider-group">
    <label>Mode</label>
    <select id="percMode" style="margin:2px 0;" onchange="simPerception()">
      <option value="simple">Simple (all cones, no LiDAR)</option>
      <option value="full">Full (LiDAR filtered, multi-frame)</option>
    </select>
    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;">
      <input type="checkbox" id="flipDir" style="width:auto;margin:0;" onchange="simPerception()">
      <label style="margin:0;font-size:11px;color:#a86;">Reverse direction (flip)</label>
    </div>
    <label>Position Noise &sigma; <span id="noiseVal">0.10</span> m</label>
    <input id="noiseXY" type="range" min="0" max="0.5" step="0.01" value="0.10" oninput="document.getElementById('noiseVal').textContent=this.value;simPerception()">
    <label>Drop Rate <span id="dropVal">0.10</span></label>
    <input id="dropRate" type="range" min="0" max="0.5" step="0.01" value="0.10" oninput="document.getElementById('dropVal').textContent=this.value;simPerception()">
    <label>False Positive Rate <span id="fpVal">0.05</span></label>
    <input id="fpRate" type="range" min="0" max="0.3" step="0.01" value="0.05" oninput="document.getElementById('fpVal').textContent=this.value;simPerception()">
    <label>LiDAR Range <span id="rangeVal">30</span> m</label>
    <input id="lidarRange" type="range" min="5" max="80" step="5" value="30" oninput="document.getElementById('rangeVal').textContent=this.value;simPerception()">
    <label>LiDAR FoV <span id="fovVal">360</span>&deg;</label>
    <input id="lidarFov" type="range" min="30" max="360" step="10" value="360" oninput="document.getElementById('fovVal').textContent=this.value;simPerception()">
  </div>

  <h3>Frame QC</h3>
  <div id="frameInfo" style="font-size:10px;color:#666;margin-bottom:4px;">No frames loaded</div>
  <div class="qc-row">
    <button class="btn-approve" onclick="setQC('approved')" style="flex:1;font-size:11px;">Approve</button>
    <button class="btn-reject" onclick="setQC('rejected')" style="flex:1;font-size:11px;">Reject</button>
    <button class="btn-reset" onclick="setQC(null)" style="flex:1;font-size:11px;">Reset</button>
  </div>
  <div id="qcSummary" style="font-size:10px;color:#666;margin-top:4px;"></div>

  <h3>Export</h3>
  <button onclick="exportCurrent()" style="font-size:11px;">Export Current Frame</button>
  <button onclick="exportApproved()" class="primary" style="font-size:11px;">Export All Approved</button>
  <button onclick="clearPerception()" style="background:#422;font-size:11px;margin-top:4px;">Clear Overlay</button>

  <h3>SideNet Preview</h3>
  <div id="preview">No frame selected</div>

  <div id="stats" style="font-size:10px;color:#666;margin-top:8px;"></div>
</div>
<div id="main">
  <canvas id="c"></canvas>
  <div id="info">Click & drag to pan, scroll to zoom | Arrow keys to navigate frames</div>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

let offsetX = 0, offsetY = 0, scale = 2.5;
let dragging = false, lastX, lastY;
let trackData = null, perceptionResult = null, currentFrame = 0;
let measureMode = false, measurePts = [];
let showGrid = true, gridSize = 5;
let qcStatus = {};  // frame_id -> "approved" | "rejected" | undefined

function resize() {
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  draw();
}
window.addEventListener('resize', resize);

function canvasPos(e) {
  let rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

function screenToWorld(sx, sy) {
  return [(sx - canvas.width/2) / scale - offsetX, -(sy - canvas.height/2) / scale + offsetY];
}

canvas.addEventListener('mousedown', e => {
  if (measureMode) {
    let [cx, cy] = canvasPos(e);
    let [wx, wy] = screenToWorld(cx, cy);
    if (measurePts.length >= 2) measurePts = [];
    measurePts.push({x: wx, y: wy});
    if (measurePts.length === 2) {
      let dx = measurePts[1].x - measurePts[0].x;
      let dy = measurePts[1].y - measurePts[0].y;
      let dist = Math.sqrt(dx*dx + dy*dy);
      document.getElementById('measureInfo').style.display = 'inline';
      document.getElementById('measureInfo').textContent = 'dist: ' + dist.toFixed(2) + 'm';
    }
    draw();
    return;
  }
  dragging = true; lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('mouseup', () => dragging = false);
canvas.addEventListener('mousemove', e => {
  if (dragging && !measureMode) {
    offsetX -= (e.clientX - lastX) / scale;
    offsetY -= (e.clientY - lastY) / scale;
    lastX = e.clientX; lastY = e.clientY; draw();
  }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  let [cx, cy] = canvasPos(e);
  let [wx, wy] = screenToWorld(cx, cy);
  scale *= e.deltaY < 0 ? 1.1 : 0.9;
  offsetX = (cx - canvas.width/2) / scale - wx;
  offsetY = -(cy - canvas.height/2) / scale - wy;
  draw();
});

function worldToScreen(x, y) {
  return [(x + offsetX) * scale + canvas.width/2, (-y + offsetY) * scale + canvas.height/2];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grid
  if (showGrid) {
    let [wl, wt] = screenToWorld(0, 0);
    let [wr, wb] = screenToWorld(canvas.width, canvas.height);
    let minX = Math.min(wl, wr), maxX = Math.max(wl, wr);
    let minY = Math.min(wt, wb), maxY = Math.max(wt, wb);
    let startIx = Math.floor(minX / gridSize);
    let endIx = Math.ceil(maxX / gridSize);
    let startIy = Math.floor(minY / gridSize);
    let endIy = Math.ceil(maxY / gridSize);

    for (let ix = startIx; ix <= endIx; ix++) {
      let isMajor = (ix % 5 === 0);
      ctx.strokeStyle = isMajor ? '#222240' : '#151525';
      ctx.lineWidth = isMajor ? 1.0 : 0.3;
      let x = ix * gridSize;
      let [sx] = worldToScreen(x, 0);
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, canvas.height); ctx.stroke();
    }
    for (let iy = startIy; iy <= endIy; iy++) {
      let isMajor = (iy % 5 === 0);
      ctx.strokeStyle = isMajor ? '#222240' : '#151525';
      ctx.lineWidth = isMajor ? 1.0 : 0.3;
      let y = iy * gridSize;
      let [, sy] = worldToScreen(0, y);
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(canvas.width, sy); ctx.stroke();
    }
  }

  if (!trackData) return;

  function drawCones(cones, color, size, alpha) {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    for (let c of cones) {
      let [sx, sy] = worldToScreen(c.x, c.y);
      ctx.beginPath(); ctx.arc(sx, sy, size, 0, Math.PI*2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // GT cones
  drawCones(trackData.left, '#4488cc', 3, 0.5);
  drawCones(trackData.right, '#ccaa00', 3, 0.5);

  // Trajectory path (all ego poses connected)
  if (perceptionResult && perceptionResult.frames && perceptionResult.frames.length > 1) {
    ctx.strokeStyle = '#555';
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < perceptionResult.frames.length; i++) {
      let ep = perceptionResult.frames[i].ego_pose;
      let [sx, sy] = worldToScreen(ep[0], ep[1]);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    // Close the loop
    let ep0 = perceptionResult.frames[0].ego_pose;
    let [sx0, sy0] = worldToScreen(ep0[0], ep0[1]);
    ctx.lineTo(sx0, sy0);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Perception overlay
  if (perceptionResult) {
    let cones = [];
    if (perceptionResult.frames && perceptionResult.frames.length > 0) {
      let frame = perceptionResult.frames[Math.min(currentFrame, perceptionResult.frames.length-1)];
      cones = frame.cones;
    } else if (perceptionResult.cones) {
      cones = perceptionResult.cones;
    }

    for (let c of cones) {
      let [sx, sy] = worldToScreen(c[0], c[1]);
      let side = c[3] || c.side;
      let source = c[4] || c.source || 'ground_truth';
      if (source === 'false_positive') {
        ctx.fillStyle = '#ffaa00';
      } else if (source === 'gaussian_noise') {
        ctx.fillStyle = side === 'Left' ? '#44ccff' : '#ddcc00';
      } else {
        ctx.fillStyle = side === 'Left' ? '#2288aa' : '#aa8800';
      }
      ctx.beginPath(); ctx.arc(sx, sy, source === 'false_positive' ? 4 : 5, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = side === 'Left' ? '#2288cc' : '#cc9900';
      ctx.lineWidth = 1;
      if (source !== 'false_positive') ctx.stroke();
    }
  }

  // Start marker
  if (trackData.start && trackData.start.length >= 2) {
    let [sx, sy] = worldToScreen(trackData.start[0], trackData.start[1]);
    ctx.fillStyle = '#00ff00';
    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI*2); ctx.fill();
  }

  // Ego pose + LiDAR FoV arc + highlight ring
  if (perceptionResult && perceptionResult.frames && perceptionResult.frames.length > 0) {
    let frame = perceptionResult.frames[Math.min(currentFrame, perceptionResult.frames.length-1)];
    if (frame.ego_pose) {
      let egoX = frame.ego_pose[0], egoY = frame.ego_pose[1];
      let yaw = frame.ego_pose[3] || 0;
      let [ex, ey] = worldToScreen(egoX, egoY);

      // LiDAR FoV arc
      let lidarRange = parseFloat(document.getElementById('lidarRange').value) || 30;
      let lidarFov = parseFloat(document.getElementById('lidarFov').value) || 360;
      let rScreen = lidarRange * scale;
      let halfFovRad = (lidarFov / 2.0) * Math.PI / 180.0;
      let startAngle = -(yaw + halfFovRad);
      let endAngle = -(yaw - halfFovRad);
      ctx.fillStyle = 'rgba(100,200,255,0.06)';
      ctx.strokeStyle = 'rgba(100,200,255,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.arc(ex, ey, rScreen, startAngle, endAngle);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Highlight ring (current position)
      ctx.fillStyle = 'rgba(255,255,0,0.15)';
      ctx.beginPath(); ctx.arc(ex, ey, 12, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(ex, ey, 12, 0, Math.PI*2); ctx.stroke();

      // Ego dot + heading arrow
      ctx.fillStyle = '#ffff00';
      ctx.beginPath(); ctx.arc(ex, ey, 5, 0, Math.PI*2); ctx.fill();
      let ax = ex + Math.cos(yaw) * 15;
      let ay = ey - Math.sin(yaw) * 15;
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ax, ay); ctx.stroke();
    }
  }

  // Measure line
  if (measureMode && measurePts.length > 0) {
    for (let i = 0; i < measurePts.length; i++) {
      let [sx, sy] = worldToScreen(measurePts[i].x, measurePts[i].y);
      ctx.fillStyle = i === 0 ? '#00ff00' : '#ff4444';
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = '12px monospace';
      ctx.fillText('P' + (i+1), sx + 8, sy - 8);
    }
    if (measurePts.length === 2) {
      let [sx1, sy1] = worldToScreen(measurePts[0].x, measurePts[0].y);
      let [sx2, sy2] = worldToScreen(measurePts[1].x, measurePts[1].y);
      ctx.strokeStyle = '#ffff00'; ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2); ctx.stroke();
      ctx.setLineDash([]);
      let dx = measurePts[1].x - measurePts[0].x;
      let dy = measurePts[1].y - measurePts[0].y;
      let dist = Math.sqrt(dx*dx + dy*dy);
      let [mx, my] = worldToScreen((measurePts[0].x+measurePts[1].x)/2, (measurePts[0].y+measurePts[1].y)/2);
      ctx.fillStyle = '#ffff00'; ctx.font = 'bold 13px monospace';
      ctx.fillText(dist.toFixed(2) + ' m', mx - 30, my - 12);
    }
  }

  // Legend + QC status
  ctx.font = '11px monospace';
  ctx.fillStyle = '#aaa';
  ctx.fillText('GT Left(Blue)  GT Right(Yellow)  Perc  FalsePos(Orange)', 10, 20);
  let ly = 36;
  if (perceptionResult && perceptionResult.frames) {
    let fid = perceptionResult.frames[Math.min(currentFrame, perceptionResult.frames.length-1)].frame_id;
    let st = qcStatus[fid];
    let qcLabel = st === 'approved' ? ' [APPROVED]' : st === 'rejected' ? ' [REJECTED]' : ' [?]';
    ctx.fillStyle = st === 'approved' ? '#4f4' : st === 'rejected' ? '#f44' : '#ff0';
    ctx.fillText('Frame: ' + (currentFrame+1) + '/' + perceptionResult.frames.length + '  Ego' + qcLabel, 10, ly);
    ly += 16;
  }
  if (document.getElementById('flipDir').checked) {
    ctx.fillStyle = '#a86';
    ctx.fillText('FLIPPED (reverse direction)', 10, ly);
  }
}

// ── Frame info & QC ──────────────────────────────────────────────

function updateFrameInfo() {
  let el = document.getElementById('frameInfo');
  let qcEl = document.getElementById('qcSummary');
  if (!perceptionResult || !perceptionResult.frames) {
    el.textContent = 'No frames loaded';
    qcEl.textContent = '';
    updatePreview(null);
    return;
  }
  let frame = perceptionResult.frames[Math.min(currentFrame, perceptionResult.frames.length-1)];
  let cones = frame.cones || [];
  let leftC = 0, rightC = 0, fpC = 0, gtC = 0, gnC = 0;
  for (let c of cones) {
    let side = c[3] || c.side;
    let source = c[4] || c.source || 'ground_truth';
    if (side === 'Left') leftC++;
    else if (side === 'Right') rightC++;
    if (source === 'false_positive') fpC++;
    else if (source === 'gaussian_noise') gnC++;
    else gtC++;
  }

  let warn = '';
  if (leftC === 0) warn += ' <span style="color:#f44;">WARNING: No Left cones!</span>';
  if (rightC === 0) warn += ' <span style="color:#f44;">WARNING: No Right cones!</span>';

  el.innerHTML = 'Frame ' + (currentFrame+1) + '/' + perceptionResult.frames.length +
    '<br>Left: <span style="color:#4af;">' + leftC + '</span>  Right: <span style="color:#ca0;">' + rightC + '</span>  FP: <span style="color:#fa0;">' + fpC + '</span>' +
    '<br>GT: ' + gtC + '  Noisy: ' + gnC + warn;

  // QC summary
  let approved = 0, rejected = 0, pending = 0;
  for (let f of perceptionResult.frames) {
    let s = qcStatus[f.frame_id];
    if (s === 'approved') approved++;
    else if (s === 'rejected') rejected++;
    else pending++;
  }
  qcEl.innerHTML = '<span style="color:#4f4;">Approved: ' + approved + '</span> / <span style="color:#f44;">Rejected: ' + rejected + '</span> / Pending: ' + pending;

  // QC button highlight
  let fid = frame.frame_id;
  let st = qcStatus[fid];
  document.querySelector('.btn-approve').style.background = st === 'approved' ? '#4a4' : '#252';
  document.querySelector('.btn-reject').style.background = st === 'rejected' ? '#a44' : '#422';

  updatePreview(frame);
}

function updatePreview(frame) {
  let el = document.getElementById('preview');
  if (!frame || !frame.cones) { el.textContent = 'No frame selected'; return; }
  let lines = [];
  for (let c of frame.cones) {
    let side = c[3] || c.side;
    let source = c[4] || c.source || 'ground_truth';
    if (source === 'false_positive') continue;
    let cls = side === 'Left' ? 'Cone_Left' : 'Cone_Right';
    lines.push(c[0].toFixed(4) + ' ' + c[1].toFixed(4) + ' ' + c[2].toFixed(4) + ' 0.200 0.200 0.300 0.000 1.0 ' + cls);
  }
  el.textContent = lines.length > 0 ? lines.join('\\n') : '(empty frame)';
}

function setQC(status) {
  if (!perceptionResult || !perceptionResult.frames) return;
  let frame = perceptionResult.frames[Math.min(currentFrame, perceptionResult.frames.length-1)];
  if (status === null) {
    delete qcStatus[frame.frame_id];
  } else {
    qcStatus[frame.frame_id] = status;
  }
  draw();
  updateFrameInfo();
}

// ── Track loading ────────────────────────────────────────────────

async function loadTracks() {
  let resp = await fetch('/api/tracks');
  let tracks = await resp.json();
  let sel = document.getElementById('trackSelect');
  sel.innerHTML = '<option value="">-- Select --</option>';
  for (let t of tracks) {
    sel.innerHTML += '<option value="' + t.name + '">' + t.name + ' (' + t.track_length_est_m + 'm, ' + t.left_cones + 'L/' + t.right_cones + 'R)</option>';
  }
  let presetResp = await fetch('/api/presets');
  let presetData = await presetResp.json();
  let psel = document.getElementById('presetSelect');
  psel.innerHTML = '<option value="">-- Preset --</option>';
  for (let [name, info] of Object.entries(presetData.presets)) {
    psel.innerHTML += '<option value="' + name + '">' + name + ': ' + info.description + '</option>';
  }
}
loadTracks();

async function loadTrack() {
  let name = document.getElementById('trackSelect').value;
  if (!name) return;
  let resp = await fetch('/api/tracks/' + name);
  trackData = await resp.json();
  perceptionResult = null; currentFrame = 0; qcStatus = {};
  document.getElementById('stats').innerHTML = trackData.stats.track_length_est_m + 'm, ' + trackData.left.length + 'L/' + trackData.right.length + 'R, closed=' + trackData.closed_loop;
  fitView(); draw(); simPerception();
}

async function generateTrack() {
  let preset = document.getElementById('presetSelect').value;
  if (!preset) return;
  let seed = parseInt(document.getElementById('genSeed').value) || 42;
  let resp = await fetch('/api/generate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({preset: preset, seed: seed})
  });
  let data = await resp.json();
  trackData = null; perceptionResult = null; currentFrame = 0; qcStatus = {};
  let tResp = await fetch('/api/tracks/' + data.name);
  trackData = await tResp.json();
  document.getElementById('trackSelect').innerHTML += '<option value="' + data.name + '" selected>' + data.name + '</option>';
  document.getElementById('stats').innerHTML = data.stats.track_length_est_m + 'm, ' + data.stats.left_cones + 'L/' + data.stats.right_cones + 'R, closed=true (generated)';
  fitView(); draw(); simPerception();
}

function fitView() {
  if (!trackData) return;
  let xs = [], ys = [];
  for (let c of [...trackData.left, ...trackData.right]) { xs.push(c.x); ys.push(c.y); }
  if (xs.length === 0) return;
  let minX = Math.min(...xs), maxX = Math.max(...xs);
  let minY = Math.min(...ys), maxY = Math.max(...ys);
  let pad = 10;
  scale = Math.min(canvas.width/(maxX-minX+pad*2), canvas.height/(maxY-minY+pad*2));
  offsetX = -(minX + maxX)/2;
  offsetY = -(minY + maxY)/2;
}

// ── Perception ───────────────────────────────────────────────────

async function simPerception() {
  if (!trackData) return;
  let name = trackData.name;
  let mode = document.getElementById('percMode').value;
  let noise = document.getElementById('noiseXY').value;
  let drop = document.getElementById('dropRate').value;
  let fp = document.getElementById('fpRate').value;
  let range = document.getElementById('lidarRange').value;
  let fov = document.getElementById('lidarFov').value;
  let flip = document.getElementById('flipDir').checked;
  let url = '/api/tracks/' + name + '/perceive?noise_xy=' + noise + '&drop_rate=' + drop + '&fp_rate=' + fp + '&lidar_range=' + range + '&lidar_fov=' + fov + '&mode=' + mode + '&ego_spacing=5&flip=' + flip;

  let resp = await fetch(url, {method:'POST'});
  perceptionResult = await resp.json();
  currentFrame = 0;
  qcStatus = {};

  if (perceptionResult.frames) {
    document.onkeydown = function(e) {
      if (!perceptionResult || !perceptionResult.frames) return;
      if (e.key === 'ArrowLeft') { currentFrame = Math.max(0, currentFrame-1); draw(); updateFrameInfo(); }
      if (e.key === 'ArrowRight') { currentFrame = Math.min(perceptionResult.frames.length-1, currentFrame+1); draw(); updateFrameInfo(); }
    };
    document.getElementById('stats').innerHTML = 'Full mode: ' + perceptionResult.num_frames + ' frames, ' + (perceptionResult.stats?.total_cones || '?') + ' cones';
  } else {
    document.getElementById('stats').innerHTML = 'Simple mode: ' + perceptionResult.num_cones + ' cones';
  }
  draw();
  updateFrameInfo();
}

// ── Export ────────────────────────────────────────────────────────

function conesFromFrame(frame) {
  return (frame.cones || []).map(c => ({x:c[0], y:c[1], z:c[2], side:c[3]||c.side, source:c[4]||c.source||'ground_truth'}));
}

async function exportCurrent() {
  if (!perceptionResult) return;
  let cones;
  if (perceptionResult.frames) {
    let frame = perceptionResult.frames[currentFrame];
    cones = conesFromFrame(frame);
  } else {
    cones = perceptionResult.cones;
  }
  let resp = await fetch('/api/export', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({cones: cones})
  });
  let data = await resp.json();
  let blob = new Blob([data.text], {type:'text/plain'});
  let a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'cloud_' + currentFrame + '.txt'; a.click();
}

async function exportApproved() {
  if (!perceptionResult || !perceptionResult.frames) return;
  let approvedFrames = [];
  for (let f of perceptionResult.frames) {
    if (qcStatus[f.frame_id] === 'approved') {
      approvedFrames.push({frame_id: f.frame_id, cones: conesFromFrame(f), status: 'approved'});
    }
  }
  if (approvedFrames.length === 0) { alert('No approved frames to export'); return; }
  let resp = await fetch('/api/export_batch', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({frames: approvedFrames, track_name: trackData.name})
  });
  let blob = await resp.blob();
  let a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = trackData.name + '_sidenet.zip'; a.click();
}

function clearPerception() {
  perceptionResult = null; currentFrame = 0; qcStatus = {};
  document.getElementById('stats').innerHTML = trackData ? trackData.stats.track_length_est_m + 'm, ' + trackData.left.length + 'L/' + trackData.right.length + 'R' : '';
  document.getElementById('frameInfo').textContent = 'No frames loaded';
  document.getElementById('qcSummary').textContent = '';
  document.getElementById('preview').textContent = 'No frame selected';
  draw();
}

function toggleMeasure() {
  measureMode = !measureMode; measurePts = [];
  document.getElementById('measureInfo').style.display = 'none';
  document.getElementById('btnMeasure').style.background = measureMode ? '#464' : '#224';
  document.getElementById('btnMeasure').textContent = measureMode ? 'Measure: ON' : 'Measure';
  draw();
}

function toggleGrid() {
  showGrid = !showGrid;
  document.getElementById('btnGrid').textContent = showGrid ? 'Grid: ON' : 'Grid: OFF';
  document.getElementById('btnGrid').style.background = showGrid ? '#224' : '#422';
  draw();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && measureMode) toggleMeasure();
});

resize();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    import uvicorn
    app = create_app()
    print(f"Track Generator & Visualizer -> http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

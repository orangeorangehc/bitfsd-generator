# BITFSD Generator

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue.svg">
  <img alt="FastAPI" src="https://img.shields.io/badge/Web-FastAPI-009688.svg">
  <img alt="OpenPCDet" src="https://img.shields.io/badge/Output-OpenPCDet%20compatible-orange.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

**FSD（大学生无人方程式 / Formula Student Driverless）赛道与感知数据合成平台。** 依据 FSD 高速循迹赛道规则生成赛道、模拟 LiDAR 感知、导出锥桶级别（位置 + 左/右标签）数据，用于「感知 → 规划」闭环测试与下游模型训练。输出格式与 [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) `CustomDataset` 兼容，可直接喂给配套的左右分类器 [SideNet](https://github.com/Functionhx/SideNet)。

> 📖 本文档**以中文为主**，English version is available [below ↓](#english)

---

## ✨ 功能特性

- **赛道合成**：用「直道 + 弯道」分段描述生成闭环赛道中心线，沿法向偏移得到左右边界，按**曲率自适应**布置锥桶（弯道处自动加密）。
- **真实赛道加载**：内置 15 条真实 FSD 赛道（FSE / FSG / FSCZ / FSI / FSO / FSS 系列），与合成赛道走**同一套**下游管线。
- **感知仿真**：采样 ego 位姿 → LiDAR 视场 / 距离过滤 → 噪声注入（漏检 / 坐标抖动 / 虚警），逐帧还原真实检测器的输出分布。
- **Web 可视化**：FastAPI + Canvas 2D 单页前端，**免构建**；滑块实时预览感知噪声，方向键逐帧浏览。
- **多种导出格式**：OpenPCDet 标注格式 / 带来源标签（GT / 噪声 / 虚警）的感知格式。
- **下游打通**：导出数据可直接训练 [SideNet](https://github.com/Functionhx/SideNet) 锥桶左右分类器。

---

## 🏗️ 系统架构

真实赛道与合成赛道汇入**同一个** `TrackData` 对象，因此所有下游代码（感知、导出、Web、可视化）都与数据来源无关：

```
config/track_presets.yaml ──┐
                            ├──► TrackData ──► PerceptionPipeline ──► export ──► .txt 帧序列
data/*.yaml（真实 FSD）   ──┘    (track.py)     (perception.py)      (export.py)
```

**赛道生成（`src/track_generator.py`）** 是分段式而非曲线拟合：

1. 遍历 `{type: straight, length}` / `{type: curve, radius, angle}` 段，按 `centerline_resolution`（默认 0.2 m）生成稠密中心线折线；
2. 线性插值自动闭环；
3. 沿法向偏移 ±`track_width/2` 得到左右边界；
4. 在每条边界上按**曲率自适应间距**布桩：逐点计算三点曲率 → 5 点盒式平滑 → 线性映射到 `[cone_spacing_min, cone_spacing_max]`，曲率越大间距越小。

**感知管线（`src/perception.py`）** 每条赛道跑三个阶段：

1. **位姿生成** —— 沿弧长参数化的中心线按 `ego_spacing` 采样，由局部切线求 yaw，对 (x, y, yaw) 加高斯噪声；
2. **LiDAR 过滤** —— 把 GT 锥桶变换到 ego 坐标系，只保留 `[range_min, range_max]` 且在 `±fov/2` 内的；
3. **噪声注入** —— 逐锥桶 dropout、XY/Z 高斯抖动，再按 `fp_rate` 注入虚警（虚警同样过 LiDAR 距离过滤，保证「看起来合理」）。

每个 `PerceptionCone` 都带 `source` 标签（`ground_truth` / `gaussian_noise` / `false_positive`），便于可视化上色和训练时区分。

**锥桶颜色 → OpenPCDet 左右标签映射（`src/track.py`）：**

| YAML `class`                          | side          |
| ------------------------------------- | ------------- |
| `blue`                                | `Cone_Left`   |
| `yellow`                              | `Cone_Right`  |
| `big-orange` / `small-orange` / `unknown` | `Cone`    |
| `invisible`                           | 丢弃          |

---

## 🚀 安装

推荐使用独立环境配置脚本（Linux / WSL，Python 3.12/3.13）：

```bash
bash setup_env.sh          # 创建/复用本项目 .venv，安装依赖并检查
bash setup_env.sh --check  # 只检测，不安装或更新依赖
```

脚本从 `pyproject.toml` 读取依赖，覆盖数据生成、Web 服务和可视化；不安装可选推理所需的 PyTorch。
有 [`uv`](https://github.com/astral-sh/uv) 时可以自动下载 Python，否则使用本机 Python + venv/pip。
可用 `--venv .venv-new --python 3.12` 建立新环境，相对路径以本项目为基准。
已有环境会复用，满足要求的依赖会保留；损坏的环境会报错，不会自动删除。
该脚本按 `pyproject.toml` 的版本范围安装；若需要严格使用 `uv.lock`，继续使用下面的 `uv sync`。

使用脚本配置后，无需激活环境即可运行：

```bash
.venv/bin/python collect_multiseed.py --config config/perceive_mixed.yaml
.venv/bin/python main.py serve
```

也可以沿用 **Python 3.12 + uv** 的安装方式（`start.sh` 仍需要 uv）：

```bash
git clone git@github.com:Functionhx/bitfsd-generator.git
cd bitfsd-generator
uv sync            # 安装依赖（如需代理：export http_proxy=http://127.0.0.1:7890）
```

依赖：`fastapi`、`uvicorn`、`numpy`、`pyyaml`、`matplotlib`（详见 `pyproject.toml`）。

---

## 📖 快速开始

所有功能通过 `main.py` 子命令暴露（也可用 `uv run python main.py <cmd>`）：

```bash
# 1) 启动 Web 可视化（默认 http://localhost:8001）
./start.sh                    # 等价于 uv run python main.py serve
./start.sh -p 8080            # 换端口

# 2) 合成一条赛道（预设见 config/track_presets.yaml，如 simple_oval / hairpin / s_curve）
uv run python main.py generate --preset simple_oval --seed 42 --output output/

# 3) 对某条赛道跑感知仿真，逐帧导出标签文件
uv run python main.py perceive --track data/FSE22.yaml --output output/

# 4) 按 YAML 配置批量采集 SideNet 训练数据（默认 ego/LiDAR 坐标）
uv run python main.py collect --config config/perceive.yaml

# 5) 用训练好的检查点做左右分类推理 + 对比 GT
uv run python main.py infer --track data/FSCZ24.yaml --ckpt output/sidenet_ckpt/sidenet.pth
```

`perceive` 常用参数：`--ego-spacing`、`--noise-xy`、`--drop-rate`、`--fp-rate`、`--lidar-range`、`--lidar-fov`、`--seed`。

### 批量生成 SideNet 训练数据

通过专用 YAML 批量生成：

```bash
bash generate_data.sh                                      # config/generate_data.yaml
bash generate_data.sh --config config/generate_data.yaml --dry-run
```

修改 `config/generate_data.yaml` 中的地图、预设、seed、传感器参数和输出目录即可。
默认使用本项目 `.venv`，输出到 `output/sidenet_data_pipeline_ego`；已有输出目录会报错。
脚本从自身位置解析项目路径，可从其他目录调用。
随后在 SideNet/sidenet 项目中执行 `bash split_data.sh` 和 `bash start_training.sh`，
分别读取 `configs/split_data.yaml` 和 `configs/train.yaml`。
完整说明见 SideNet 的 `docs/WORKFLOW_SCRIPTS.md`。

首次使用请先阅读配套的 [数据生成、划分与训练操作指南](https://github.com/orangeorangehc/sidenet/blob/main/docs/TRAINING_GUIDE.md)，
其中包含两个仓库的克隆、共享环境安装、训练/验证/测试划分及独立测试命令。

若要一次生成虚拟 + 真实赛道的多个 seed，使用专用入口：

```bash
# 从 bitfsd-generator 目录执行，使用已经安装依赖的 SideNet 环境
../SideNet/.venv/bin/python collect_multiseed.py --config config/perceive_mixed.yaml
```

该配置使用 seed 42/43/44 和四个虚拟预设，输出到 `output/sidenet_data_mixed_ego/`；
真实赛道也会得到 `_s42` 等目录后缀，各 seed 的帧由同一个 manifest 管理。
脚本拒绝覆盖已有数据目录。seed 只改变观测噪声，不改变预设几何。
划分与训练说明见相邻项目的 [MIXED_DATASET.md](../SideNet/docs/MIXED_DATASET.md)。

`collect` 会读取 `real_tracks` 中的真实赛道，并读取 `synthetic_tracks` 中的
`config/track_presets.yaml` 预设。默认配置处理 14 条真实赛道，输出到：

```text
output/sidenet_data_ego/
├── dataset_manifest.yaml
├── FSE22/
│   ├── metadata.yaml
│   ├── cloud_0.txt
│   └── ...
└── ...
```

每个 `cloud_N.txt` 使用 SideNet v1 格式：

```text
x y z dx dy dz heading score class_name
```

默认 `x y z` 已转换到当前车辆 noisy pose 的 ego/LiDAR 坐标系（`+x` 向前、`+y`
向左），`class_name` 是 `Cone_Left` 或 `Cone_Right`。误检不会写入训练文件，因为
二分类 SideNet 没有误检的真实 side 标签。

`dataset_manifest.yaml` 和每个赛道的 `metadata.yaml` 会声明
`coordinate_frame: ego`、`side_semantics: track_global` 以及帧文件列表，SideNet
默认 loader 可以据此校验数据来源和坐标契约。

要加入合成赛道，编辑配置：

```yaml
synthetic_tracks: [simple_oval, hairpin, s_curve]
```

需要生成多个随机版本时，使用不同输出目录，避免覆盖已有数据：

```bash
for seed in 42 43 44; do
  uv run python main.py collect --config config/perceive.yaml \
    --seed "$seed" --output "output/sidenet_data_ego_s${seed}"
done
```

默认不生成 reverse `_flip` 数据。SideNet 的训练增强会在 ego frame 中进行 Y 反射并同步
交换 Left/Right 标签；只有明确需要反向行驶观测时，才将 `augmentation.flip` 改为 `true`。

---

## 📂 输出格式

`src/export.py` 产出两种**不要混淆**的文本格式：

```text
# OpenPCDet 格式（固定 dx/dy/dz/heading）
x y z 0.200 0.200 0.300 0.000 {Cone_Left|Cone_Right|Cone}

# 感知格式（额外带来源标签）
x y z {Left|Right} {ground_truth|gaussian_noise|false_positive} [orig_x orig_y orig_z]
```

只有 `gaussian_noise` 行才会附带末尾的 `orig_xyz`（噪声前的原始 GT 位置）。

---

## 🗺️ 真实赛道数据

`data/` 收录了 15 条真实 FSD 闭环赛道：`FSE22/23/24`、`FSG19/21/23/24`、`FSCZ24`、`FSI24`、`FSO20`、`FSS19`、`FSS22_V1/V2`。
另有 3 个非闭环项目场地：`skidpad`（8 字绕环）、`acceleration`（直线加速）、`gripMap`。后者也能被 `load_all_tracks` 读取，但闭环相关统计对它们无意义。

---

## 📁 项目结构

```
bitfsd-generator/
├── main.py                  # CLI 入口：generate / perceive / collect / serve / infer
├── start.sh                 # 启动 Web 服务（默认 8001 端口）
├── src/
│   ├── track_generator.py   # 分段式赛道生成 + 曲率自适应布桩
│   ├── track.py             # TrackData：统一真实/合成赛道的数据结构
│   ├── perception.py        # 三阶段感知仿真管线
│   ├── export.py            # 导出 OpenPCDet / 感知文本格式
│   ├── server.py            # FastAPI Web 服务（前端内嵌为单个 HTML 字符串）
│   └── visualize.py         # matplotlib 离线可视化
├── config/
│   ├── track_presets.yaml   # 合成赛道预设
│   └── perceive.yaml        # 批量数据采集配置
├── data/                    # 15 条真实 FSD 赛道 YAML
├── output/sidenet_ckpt/     # 预训练检查点（pointnet / sidenet / transformer .pth）
├── docs/                    # 通信协议等设计文档
├── FSD_RULE.md              # FSD 高速循迹赛道规则摘要（几何约束来源）
└── pyproject.toml
```

> 几何约束（闭环 200–500 m、最小宽度 3 m、发夹弯外径 ≥ 9 m、直道 ≤ 80 m 等）来自 [`FSD_RULE.md`](FSD_RULE.md)。

---

## 🔗 相关项目

- **[SideNet](https://github.com/Functionhx/SideNet)** —— 消费本平台导出的数据，训练锥桶左右分类器。
- **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)** —— 导出格式直接对接其 `CustomDataset`。

---
<a name="english"></a>

## English

**A track + perception data synthesis platform for Formula Student Driverless (FSD).** It generates closed-loop tracks from FSD high-speed-tracking rules, simulates LiDAR perception, and exports cone-level data (positions + Left/Right labels) for perception-to-planning closed-loop testing and downstream model training. Output is compatible with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) `CustomDataset` and feeds the companion classifier [SideNet](https://github.com/Functionhx/SideNet).

### Features

- **Track synthesis** — build a closed centerline from `straight + curve` segments, offset to left/right boundaries, and place cones with **curvature-adaptive spacing** (tighter on curves).
- **Real tracks** — 15 bundled real FSD tracks (FSE / FSG / FSCZ / FSI / FSO / FSS) load into the **same** pipeline as synthetic ones.
- **Perception simulation** — ego-pose sampling → LiDAR range/FoV filter → noise injection (dropout / jitter / false positives).
- **Web visualizer** — FastAPI + Canvas 2D single-page UI, no build step; live slider preview and arrow-key frame stepping.
- **Export formats** — OpenPCDet label format, plus a perception format tagged with provenance (GT / noise / false-positive).

### Architecture

Real and synthetic tracks both load into one `TrackData` object, so all downstream code is source-agnostic:

```
config/track_presets.yaml ──┐
                            ├──► TrackData ──► PerceptionPipeline ──► export ──► .txt frames
data/*.yaml (real FSD)   ──┘    (track.py)     (perception.py)      (export.py)
```

`TrackGenerator` is segment-based (not spline fitting): walk segments → dense centerline → auto-close → normal-offset boundaries → curvature-adaptive cone placement. `PerceptionPipeline` runs three stages per track: pose generation, LiDAR filtering, and noise injection; each output cone carries a `source` tag.

### Install & run

Requires **Python 3.12** and [`uv`](https://github.com/astral-sh/uv):

```bash
git clone git@github.com:Functionhx/bitfsd-generator.git
cd bitfsd-generator && uv sync

./start.sh                                                                  # web UI at :8001
uv run python main.py generate --preset simple_oval --seed 42              # synth a track
uv run python main.py perceive --track data/FSE22.yaml --output output/    # run perception
uv run python main.py collect  --config config/perceive.yaml               # batch collect
uv run python main.py infer    --track data/FSCZ24.yaml --ckpt output/sidenet_ckpt/sidenet.pth
```

### Output format

```text
# OpenPCDet:   x y z 0.200 0.200 0.300 0.000 {Cone_Left|Cone_Right|Cone}
# Perception:  x y z {Left|Right} {ground_truth|gaussian_noise|false_positive} [orig_x orig_y orig_z]
```

The trailing `orig_xyz` appears only on `gaussian_noise` rows. See the Chinese sections above for the full module map and cone-class mapping table.

### Related

- **[SideNet](https://github.com/Functionhx/SideNet)** — trains a Left/Right cone classifier on data exported here.
- **[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)** — export format plugs straight into its `CustomDataset`.

---

## 📄 License

[MIT](LICENSE) © Yuchen Fan

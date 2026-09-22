#!/usr/bin/env bash
# Linux / WSL. This file is independent of the sibling repository.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_REQUEST="3.12"
CHECK_ONLY=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
log() { printf '[环境] %s\n' "$*"; }
require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 需要一个参数"
}

usage() {
  cat <<'HELP'
用法：bash setup_env.sh [--check] [--python 3.12|解释器路径] [--venv 路径]
  默认       创建/复用本项目 .venv，按 pyproject.toml 安装依赖并验证。
  --check    只检查现有环境，不安装或更新依赖；失败返回非零。
  --python   新建环境时使用的 Python，默认 3.12；支持 3.12/3.13。
  --venv     环境路径，相对路径以本项目为基准，默认 .venv。
  -h, --help 显示帮助。

有 uv 时可自动下载 Python；否则使用本机 Python + venv/pip。
安装覆盖数据生成、Web 服务和可视化，不包含可选的 PyTorch 推理。
HELP
}
while (( $# )); do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --python) require_value "$@"; PYTHON_REQUEST="$2"; shift 2 ;;
    --venv) require_value "$@"; VENV_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1（使用 --help 查看用法）" ;;
  esac
done

# Resolve relative environment paths against this repository, not the caller's cwd.
[[ "$VENV_DIR" = /* ]] || VENV_DIR="$PROJECT_DIR/$VENV_DIR"
VENV_DIR="${VENV_DIR%/}"
[[ -n "$VENV_DIR" ]] || die "不能使用文件系统根目录作为虚拟环境"
ENV_PY="$VENV_DIR/bin/python"

python_supported() {
  "$1" -I -B -c 'import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] <= (3, 13)))' >/dev/null 2>&1
}

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
  [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$ENV_PY" ]] ||
    die "$VENV_DIR 不是完整的虚拟环境。请用 --venv 指定新目录；脚本不会删除旧目录。"
else
  (( CHECK_ONLY == 0 )) || die "虚拟环境不存在：$VENV_DIR；去掉 --check 可创建。"
  if command -v uv >/dev/null 2>&1; then
    log "使用 uv 创建环境（必要时下载 Python）：$VENV_DIR"
    uv venv --python "$PYTHON_REQUEST" "$VENV_DIR"
  else
    PYTHON_BIN=""
    for candidate in "$PYTHON_REQUEST" "python$PYTHON_REQUEST"; do
      if command -v "$candidate" >/dev/null 2>&1 && python_supported "$candidate"; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
    # Only the default request may fall back to another supported local Python.
    if [[ -z "$PYTHON_BIN" && "$PYTHON_REQUEST" == "3.12" ]]; then
      for candidate in python3.12 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_supported "$candidate"; then
          PYTHON_BIN="$candidate"
          break
        fi
      done
    fi
    [[ -n "$PYTHON_BIN" ]] ||
      die "找不到 Python 3.12/3.13 或 uv。请安装 uv（https://docs.astral.sh/uv/getting-started/installation/），或用 --python 指定解释器。"
    log "使用 $PYTHON_BIN 创建环境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" ||
      die "创建失败；Debian/Ubuntu 请检查对应的 python3.x-venv 包，然后使用新的 --venv 目录重试。"
  fi
fi

python_supported "$ENV_PY" || die "环境需要 Python 3.12 或 3.13；请用 --venv 指定新目录。"
"$ENV_PY" -I -B - "$VENV_DIR" <<'PY'
import pathlib
import sys
expected = pathlib.Path(sys.argv[1]).resolve()
if sys.prefix == sys.base_prefix or pathlib.Path(sys.prefix).resolve() != expected:
    sys.exit("错误：解释器没有指向指定虚拟环境，停止安装。")
print(f"[环境] Python {sys.version.split()[0]}: {sys.executable}")
PY

install_packages() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ENV_PY" "$@"
  else
    if ! "$ENV_PY" -I -B -m pip --version >/dev/null 2>&1; then
      "$ENV_PY" -I -B -m ensurepip --upgrade ||
        die "环境缺少 pip/ensurepip，请安装 uv 后重试。"
    fi
    "$ENV_PY" -I -B -m pip --isolated install --disable-pip-version-check "$@"
  fi
}

check_dependencies() {
  if command -v uv >/dev/null 2>&1; then
    uv pip check --python "$ENV_PY"
  elif "$ENV_PY" -I -B -m pip --version >/dev/null 2>&1; then
    "$ENV_PY" -I -B -m pip --isolated check
  fi
}

if (( CHECK_ONLY == 0 )); then
  # Read the canonical dependency list instead of duplicating it in this script.
  REQUIREMENTS_TEXT="$("$ENV_PY" -I -B - "$PROJECT_DIR/pyproject.toml" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as handle:
    print("\n".join(tomllib.load(handle)["project"]["dependencies"]))
PY
)"
  mapfile -t REQUIREMENTS <<< "$REQUIREMENTS_TEXT"
  log "安装/检查生成器依赖"
  install_packages "${REQUIREMENTS[@]}"
fi

"$ENV_PY" -I -B - "$PROJECT_DIR" <<'PY'
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib

root = Path(sys.argv[1])
with (root / "pyproject.toml").open("rb") as handle:
    requirements = tomllib.load(handle)["project"]["dependencies"]
try:
    for requirement in requirements:
        # The project's dependencies are simple >= release requirements.
        match = re.fullmatch(r"([\w-]+)>=(\d+(?:\.\d+)*)", requirement)
        if not match:
            raise RuntimeError(f"检查器需要更新以支持依赖表达式：{requirement}")
        name, minimum = match.groups()
        version = importlib.metadata.version(name)
        release = version.split("+")[0]
        if not re.fullmatch(r"\d+(?:\.\d+)*", release):
            raise RuntimeError(f"{name} {version} 不是稳定发行版")
        if tuple(map(int, release.split("."))) < tuple(map(int, minimum.split("."))):
            raise RuntimeError(f"{name} {version} 不满足 {requirement}")
        print(f"[依赖] {name} {version}")

    # Keep matplotlib caches temporary, including in --check mode.
    with tempfile.TemporaryDirectory(prefix="generator-env-check-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import fastapi
        import uvicorn
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure = plt.figure()
        plt.close(figure)
        sys.path[:0] = [str(root), str(root / "src")]
        for module in ("track", "track_generator", "perception", "export", "collect_multiseed", "server"):
            importlib.import_module(module)
except Exception as exc:
    sys.exit(f"环境检查失败：{exc}\n请运行不带 --check 的 setup_env.sh 修复依赖。")
print("[通过] 数据生成、Web 服务、无界面绘图模块可用。")
PY
check_dependencies
printf '\n完成。批量生成命令：\n  cd %q\n  %q collect_multiseed.py --config config/perceive_mixed.yaml\n' "$PROJECT_DIR" "$ENV_PY"
printf '启动 Web：\n  %q main.py serve\n' "$ENV_PY"

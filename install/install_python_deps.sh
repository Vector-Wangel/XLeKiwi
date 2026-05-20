#!/usr/bin/env bash
# Build the .venv for the XLeKiwi stack.
# Run as the regular user (NOT sudo): bash install/install_python_deps.sh
#
# Strategy:
#   - Create venv in repo root (.venv/) inheriting system site-packages so ROS 2
#     Python (rclpy, cv_bridge from /opt/ros/humble) is reachable after sourcing ROS.
#   - Install the Panthera motor whl matching $(python3 --version) and $(uname -m).
#   - Install everything else from requirements (panthera + demo deps).
#
# Assumes: install_system_deps.sh has been run first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"
echo "[info] repo root: $REPO_ROOT"

if [ "$(id -u)" -eq 0 ]; then
    echo "[err] do NOT run this with sudo — venv must be owned by your user." >&2
    exit 1
fi

# Pick a Python interpreter. Default to system python3, but allow override.
PY_BIN="${PY_BIN:-/usr/bin/python3}"
if [ ! -x "$PY_BIN" ]; then
    echo "[err] python interpreter not found: $PY_BIN" >&2
    exit 1
fi

PY_VER_MAJORMINOR="$($PY_BIN -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
PY_VER_DOTTED="$($PY_BIN -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
ARCH="$(uname -m)"
echo "[info] python: $PY_BIN ($PY_VER_DOTTED), arch: $ARCH"

if [[ "$PY_VER_MAJORMINOR" != "310" && "$PY_VER_MAJORMINOR" != "311" && "$PY_VER_MAJORMINOR" != "312" ]]; then
    echo "[warn] only Python 3.10/3.11/3.12 are tested. Got $PY_VER_DOTTED."
    read -r -p "Continue? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

echo "[1/4] create .venv (--system-site-packages so ROS 2 rclpy is reachable)"
if [ -f .venv/bin/activate ]; then
    echo "[note] .venv looks intact; reusing. Delete .venv/ manually if you want a fresh start."
elif [ -d .venv ]; then
    echo "[note] .venv exists but is broken (no bin/activate). Removing and recreating."
    rm -rf .venv
    "$PY_BIN" -m venv --system-site-packages .venv
else
    "$PY_BIN" -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "[2/4] install panthera motor whl"
PANTHERA_ARCH="$ARCH"
[ "$ARCH" = "aarch64" ] && PANTHERA_ARCH="aarch64"
[ "$ARCH" = "x86_64" ]  && PANTHERA_ARCH="x86_64"
WHL="$REPO_ROOT/panthera_python/motor_whl/hightorque_robot-1.0.0-cp${PY_VER_MAJORMINOR}-cp${PY_VER_MAJORMINOR}-linux_${PANTHERA_ARCH}.whl"
if [ ! -f "$WHL" ]; then
    # try the 1.2.0 series (x86_64 only in current motor_whl/)
    WHL="$REPO_ROOT/panthera_python/motor_whl/hightorque_robot-1.2.0-cp${PY_VER_MAJORMINOR}-cp${PY_VER_MAJORMINOR}-linux_${PANTHERA_ARCH}.whl"
fi
if [ ! -f "$WHL" ]; then
    echo "[err] no matching panthera whl for cp${PY_VER_MAJORMINOR}/${PANTHERA_ARCH}" >&2
    echo "[err] available files:"
    ls panthera_python/motor_whl/ | sed 's/^/  /' >&2
    exit 1
fi
echo "[info] using $WHL"
pip install "$WHL"

echo "[3/4] install panthera python deps"
pip install -r panthera_python/requirements.txt

echo "[4/4] install demo / camera deps not in panthera"
pip install \
    fastapi uvicorn websockets \
    viser \
    pyrealsense2 \
    odrive

echo
echo "=== python deps installed ==="
echo
echo "smoke test:"
python -c "import rclpy" 2>/dev/null || echo "[warn] rclpy not importable yet — source ROS first: 'source /opt/ros/humble/setup.bash' then re-test"
python -c "import cv2, numpy, fastapi, viser, odrive" \
    && echo "[ok] cv2/numpy/fastapi/viser/odrive importable in .venv"
python -c "import pinocchio; print('pinocchio:', pinocchio.__version__)"
python -c "import hightorque_robot; print('hightorque_robot ok')" 2>/dev/null || \
    echo "[warn] hightorque_robot import failed (often libyaml-cpp.so.0.6 missing — see install_system_deps.sh tail note)"

echo
echo "next: bash install/setup_runtime_env.sh"

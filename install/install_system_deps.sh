#!/usr/bin/env bash
# System packages for the XLeKiwi stack on a fresh machine.
# Run with: sudo bash install/install_system_deps.sh
#
# Auto-detects the Ubuntu codename and installs the matching ROS 2 distro:
#   Ubuntu 22.04 jammy → ROS 2 Humble  (Jetson Nano, JetPack 6.x)
#   Ubuntu 24.04 noble → ROS 2 Jazzy   (Jetson Thor, JetPack 7.x)
#
# Installs:
#   - apt prerequisites + universe
#   - ROS 2 (ros-base + cv-bridge + image-transport + rosbag2-mcap + rmw-fastrtps)
#   - libserialport-dev (the Panthera Python wheel needs libserialport.so.0)
#   - libyaml-cpp / libeigen3 dev headers
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "[err] must run as root: sudo bash $0" >&2
    exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
CODENAME="$(lsb_release -cs)"
ARCH="$(dpkg --print-architecture)"
echo "[info] user=$REAL_USER home=$REAL_HOME codename=$CODENAME arch=$ARCH"

# ── Auto-pick ROS distro from Ubuntu codename ──────────────────────────
case "$CODENAME" in
    jammy)
        ROS_DISTRO=humble
        EXTRA_APT=()
        ;;
    noble)
        ROS_DISTRO=jazzy
        # noble does not include python3-venv by default; install_python_deps.sh
        # needs it to create the .venv.
        EXTRA_APT=(python3-pip python3.12-venv python3.12-dev)
        ;;
    *)
        echo "[warn] codename=$CODENAME is not jammy (22.04) or noble (24.04)."
        echo "[warn] This script only supports those two pairs:"
        echo "         jammy → Humble    noble → Jazzy"
        read -r -p "Continue anyway and try ROS Jazzy? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || exit 1
        ROS_DISTRO=jazzy
        EXTRA_APT=()
        ;;
esac
echo "[info] will install ROS 2 $ROS_DISTRO"

echo "[1/6] apt prerequisites"
apt-get update
apt-get install -y curl gnupg lsb-release git net-tools software-properties-common build-essential "${EXTRA_APT[@]}"

echo "[2/6] enable Ubuntu universe"
add-apt-repository -y universe

echo "[3/6] add ROS 2 apt repo + key"
KEYRING=/usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "$KEYRING"
echo "deb [arch=$ARCH signed-by=$KEYRING] http://packages.ros.org/ros2/ubuntu $CODENAME main" \
    > /etc/apt/sources.list.d/ros2.list
apt-get update

echo "[4/6] install ROS 2 $ROS_DISTRO base + bridge deps"
apt-get install -y \
    "ros-${ROS_DISTRO}-ros-base" \
    "ros-${ROS_DISTRO}-rmw-fastrtps-cpp" \
    "ros-${ROS_DISTRO}-cv-bridge" \
    "ros-${ROS_DISTRO}-image-transport" \
    "ros-${ROS_DISTRO}-rosbag2" \
    "ros-${ROS_DISTRO}-rosbag2-storage-mcap" \
    ros-dev-tools

echo "[5/6] panthera CAN/serial deps"
apt-get install -y libserialport-dev libyaml-cpp-dev libeigen3-dev

# The Panthera wheel is built against yaml-cpp 0.6.x. jammy ships 0.7, noble ships 0.8.
# Don't auto-build — detect and tell the user to rebuild from source if needed.
if ! ldconfig -p | grep -q "libyaml-cpp.so.0.6"; then
    echo "[note] libyaml-cpp.so.0.6 not present (system has $(dpkg -s libyaml-cpp-dev | awk -F'[: -]' '/^Version/{print $2}' | head -1))."
    echo "[note] If 'import hightorque_robot' later complains about libyaml-cpp.so.0.6,"
    echo "[note] follow panthera_python/README.md tail section to build 0.6.1 from source."
fi

echo "[6/6] verify"
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && ros2 --help >/dev/null" \
        && echo "[ok] ros2 CLI works ($ROS_DISTRO)"
else
    echo "[err] /opt/ros/${ROS_DISTRO}/setup.bash missing" >&2
    exit 1
fi

echo
echo "=== system deps installed (ROS 2 $ROS_DISTRO) ==="
echo "next:"
echo "  1. bash install/install_python_deps.sh        # as $REAL_USER, NOT sudo"
echo "  2. bash install/setup_runtime_env.sh          # bashrc + fastdds profile"

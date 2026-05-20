#!/usr/bin/env bash
# Append ROS 2 + Insight 9 lines to ~/.bashrc and place the Fast DDS
# profile at ~/fastdds_usb.xml.
# Idempotent: re-running only adds missing lines.
#
# Run as user (NOT sudo): bash install/setup_runtime_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
BASHRC="${HOME}/.bashrc"
DDS_PROFILE="${HOME}/fastdds_usb.xml"
TEMPLATE="${REPO_ROOT}/install/fastdds_usb.xml.template"

echo "[info] repo: $REPO_ROOT"
echo "[info] bashrc: $BASHRC"

# 1. Place fastdds profile if missing
if [ ! -f "$DDS_PROFILE" ]; then
    if [ ! -f "$TEMPLATE" ]; then
        echo "[err] template not found: $TEMPLATE" >&2
        exit 1
    fi
    cp "$TEMPLATE" "$DDS_PROFILE"
    echo "[ok] copied template to $DDS_PROFILE"
    echo "[note] EDIT $DDS_PROFILE — replace 169.254.10.2 with this host's IP on the cdc_ncm interface."
    echo "[note] (find it with: ip -br addr | grep 169.254)"
else
    echo "[ok] $DDS_PROFILE already exists, leaving as-is"
fi

# 2. Bashrc lines (idempotent, marked block)
MARK="# --- XLeKiwi ROS 2 + Insight 9 (managed by install/setup_runtime_env.sh) ---"
END_MARK="# --- end XLeKiwi block ---"

if grep -qF "$MARK" "$BASHRC" 2>/dev/null; then
    echo "[ok] bashrc block already present, leaving as-is"
    echo "[hint] If you upgraded ROS distro or moved hosts, delete the block manually and re-run."
else
    # Use literal heredoc ('EOF' quoted) so the for-loop variables aren't expanded at install time.
    cat >> "$BASHRC" << 'EOF'

# --- XLeKiwi ROS 2 + Insight 9 (managed by install/setup_runtime_env.sh) ---
for _distro in jazzy humble; do
    if [ -f "/opt/ros/${_distro}/setup.bash" ]; then
        source "/opt/ros/${_distro}/setup.bash"
        break
    fi
done
unset _distro
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ROS_DOMAIN_ID
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_usb.xml
# --- end XLeKiwi block ---
EOF
    echo "[ok] appended XLeKiwi block to $BASHRC"
    echo "[note] block prefers Jazzy over Humble if both are installed"
fi

echo
echo "=== runtime env set up ==="
echo
echo "next:"
echo "  1. open new terminal (or 'source ~/.bashrc')"
echo "  2. plug Insight 9, find its host-side IP: ip -br addr | grep 169.254"
echo "  3. EDIT $DDS_PROFILE — set <address> to that IP"
echo "  4. (multi-NIC hosts only) sudo bash install/add_dds_route.sh"
echo "  5. ros2 topic list  # should show /insight/vio_100hz etc."

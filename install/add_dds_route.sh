#!/usr/bin/env bash
# Force DDS multicast (239.0.0.0/8) onto the Insight 9 USB-CDC interface so
# discovery + traffic stay on the USB link and don't leak onto WiFi / other NICs.
# Run: sudo bash install/add_dds_route.sh
set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "[err] must run as root: sudo bash $0" >&2
  exit 1
fi

# Auto-detect the USB-CDC interface (link-local 169.254.x.x with state UP).
# Override IFACE and SRC env vars to skip auto-detection.
if [ -z "${IFACE:-}" ] || [ -z "${SRC:-}" ]; then
  detected_line="$(ip -br addr | awk '$2=="UP" && $3 ~ /^169\.254\./ {print $1, $3; exit}')"
  if [ -z "$detected_line" ]; then
    echo "[err] no UP interface with a 169.254.x.x address found" >&2
    echo "[hint] plug Insight 9 first, or override with IFACE=... SRC=... $0" >&2
    exit 1
  fi
  IFACE="${IFACE:-$(echo "$detected_line" | awk '{print $1}')}"
  SRC="${SRC:-$(echo "$detected_line" | awk '{print $2}' | cut -d/ -f1)}"
fi
NET=239.0.0.0/8

echo "[info] using IFACE=$IFACE SRC=$SRC"

if ! ip -br addr show "$IFACE" | grep -q UP; then
  echo "[err] $IFACE is not UP" >&2
  exit 1
fi

# Remove any pre-existing same-prefix route, then add ours via the USB iface.
ip route del $NET 2>/dev/null || true
ip route add $NET dev $IFACE src $SRC

echo "[ok] route added:"
ip route show $NET
echo
echo "[verify] kernel will now choose $IFACE for any 239.x.x.x multicast"
ip route get 239.255.0.1

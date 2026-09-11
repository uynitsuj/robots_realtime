#!/usr/bin/env bash
# Bring up the CAN bus for the *third* YAM arm plugged into us07 — the one
# under test, not part of the bimanual us07 station — under a stable name.
#
# Surveyed 2026-09-10: three CANable adapters are attached. Two are the known
# station arms (already pinned by scripts/setup_can_yam_bimanual_box.sh):
#   208237984546500A          -> can_left    (left  follower)
#   207B34A158455017          -> can_right   (right follower)
# The new adapter has a 24-hex-digit serial (the box script only matches 16):
#   0065004F594E501820313332  -> can_test    (arm under test)
#
# robot_configs/yam/test_arm.yaml expects `can_test`.
#
# Usage:
#   ./scripts/setup_can_yam_test_arm.sh            # rename + bring up (needs sudo)
#   ./scripts/setup_can_yam_test_arm.sh --check    # print mapping only, no changes
#
# If you plug in a *different* new adapter, run --check, copy its serial into
# TEST_SERIAL below, and re-run.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

BITRATE=1000000
TEST_SERIAL="0065004F594E501820313332"
TARGET="can_test"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

serial_of() {
    udevadm info -a -p "/sys/class/net/$1" 2>/dev/null \
        | grep -m1 'ATTRS{serial}=="[0-9A-Fa-f]\{16,32\}"' \
        | sed 's/.*=="\(.*\)"/\1/' || true
}

echo "CAN adapters currently attached:"
TEST_IF=""
for iface in $(ls /sys/class/net); do
    [[ -d "/sys/class/net/$iface/device" ]] || continue
    s=$(serial_of "$iface")
    [[ -n "$s" ]] || continue
    printf "  %-12s serial %s\n" "$iface" "$s"
    [[ "$s" == "$TEST_SERIAL" ]] && TEST_IF="$iface"
done
echo ""
echo "test arm: serial $TEST_SERIAL -> ${TEST_IF:-<not found>}"

if [[ -z "$TEST_IF" ]]; then
    echo ""
    echo "ERROR: the test-arm CANable ($TEST_SERIAL) is not attached."
    echo "       If a different adapter is the new one, put its serial (listed"
    echo "       above) into TEST_SERIAL in this script."
    exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    ip -brief link show | grep can || true
    exit 0
fi

current_bitrate() {
    ip -details link show "$1" 2>/dev/null | grep -oP 'bitrate \K[0-9]+' | head -1
}

if [[ "$TEST_IF" != "$TARGET" ]]; then
    if ip link show "$TARGET" &>/dev/null; then
        sudo ip link set "$TARGET" down
        sudo ip link set "$TARGET" name "${TARGET}_old"
    fi
    sudo ip link set "$TEST_IF" down
    sudo ip link set "$TEST_IF" name "$TARGET"
    sleep 0.5
fi

if [[ "$(current_bitrate "$TARGET")" != "$BITRATE" ]]; then
    sudo ip link set "$TARGET" down
    sudo ip link set "$TARGET" type can bitrate "$BITRATE"
fi
if ! ip link show "$TARGET" | grep -q "state UP"; then
    sudo ip link set "$TARGET" up
fi

got=$(current_bitrate "$TARGET")
if [[ "$got" != "$BITRATE" ]] || ! ip link show "$TARGET" | grep -q "state UP"; then
    echo "  ✗ $TARGET is not up @ ${BITRATE} bit/s (bitrate=${got:-unset})"
    exit 1
fi
echo "  ✓ $TARGET up @ ${BITRATE} bit/s"
echo ""
ip -brief link show | grep can
echo ""
echo "Ready. Launch the joint-slider test with:"
echo "  uv run rr-session configs/yam/yam_single_joint_slider_test_us07.yaml"

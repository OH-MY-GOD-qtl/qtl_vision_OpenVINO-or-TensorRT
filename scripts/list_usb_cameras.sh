#!/usr/bin/env bash
# 列出 USB 相机的稳定路径。优先使用 by-path：同一物理 USB 口重插后路径不变。
set -euo pipefail

shopt -s nullglob

echo "== /dev/video* =="
for node in /dev/video*; do
    printf '%-14s -> %s\n' "$node" "$(readlink -f "$node" 2>/dev/null || true)"
done

echo
echo "== by-id（有唯一序列号时最稳） =="
for node in /dev/v4l/by-id/*; do
    printf '%-70s -> %s\n' "$node" "$(readlink -f "$node")"
done

echo
echo "== by-path（按物理口，推荐） =="
for node in /dev/v4l/by-path/*; do
    printf '%-70s -> %s\n' "$node" "$(readlink -f "$node")"
done

echo
echo "== 建议填入 omni_usb_cameras 的 device（只取 capture 节点 index0） =="
for node in /dev/v4l/by-path/*-video-index0; do
    printf '  - {device: "%s", yaw_offset_from_hik: 0}\n' "$node"
done

echo
echo "注意："
echo "  1. 只使用 *-video-index0；index1 通常是 UVC metadata 节点。"
echo "  2. by-path 绑定物理 USB 口；换口后路径会变，需要同步改配置。"
echo "  3. 若相机有唯一序列号，优先用 /dev/v4l/by-id。"

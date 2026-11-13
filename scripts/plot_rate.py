#!/usr/bin/env python3
"""统计 Plotter 发到 UDP 的帧率与字段完整性，用来验证控制线程是否真的跑在 10 ms。

用法：
    scripts/plot_rate.py                    # 监听 127.0.0.1:9870
    scripts/plot_rate.py --port 9871 --seconds 5

注意：PlotJuggler 也绑 9870，两者不能同时收。验证帧率时先关掉 PlotJuggler。
（Plotter 的端口是代码里的默认值，要换端口得改 app 的 Plotter 构造。）

输出两件事：
  1. 每秒一行的包速率，以及相邻两拍 `t` 的时间差（ms）；
  2. 结束时按字段统计「这一拍没发这个键」的次数 —— 缺键会让 PlotJuggler
     把前后两点直接连成直线，看起来像数据断点，是稀疏折线的常见来源。
"""

import argparse
import json
import socket
import statistics
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9870)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 表示一直跑到 Ctrl+C")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)

    all_keys: set[str] = set()
    missing: dict[str, int] = {}
    gaps_ms: list[float] = []
    prev_t: float | None = None
    total = 0
    bad = 0

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    bucket_start = time.monotonic()
    bucket_count = 0
    last_gap: float | None = None
    last_keys = 0

    print(f"listening on {args.host}:{args.port}  (Ctrl+C 结束)", flush=True)

    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(data, dict) or "t" not in data:
                bad += 1
                continue

            total += 1
            bucket_count += 1
            last_keys = len(data)

            keys = set(data.keys())
            if not all_keys:
                all_keys = keys
            for key in all_keys - keys:
                missing[key] = missing.get(key, 0) + 1

            t = float(data["t"])
            if prev_t is not None and t > prev_t:
                last_gap = (t - prev_t) * 1e3
                gaps_ms.append(last_gap)
            prev_t = t

            now = time.monotonic()
            if now - bucket_start >= 1.0:
                gap_text = f"{last_gap:6.1f}ms" if last_gap is not None else "   n/a  "
                print(
                    f"t={t:8.3f}s  hz={bucket_count / (now - bucket_start):6.1f}  "
                    f"dt={gap_text}  keys={last_keys}",
                    flush=True,
                )
                bucket_start = now
                bucket_count = 0
    except KeyboardInterrupt:
        pass

    print(f"\n总包数 {total}，解析失败 {bad}")
    if total == 0:
        print("没收到任何数据：确认 app 在跑、Plotter 的端口一致，且 PlotJuggler 没有占着同一个端口")
        return 1
    if len(gaps_ms) >= 2:
        print(
            "时间戳间隔 ms: "
            f"min={min(gaps_ms):.1f}  median={statistics.median(gaps_ms):.1f}  "
            f"p95={statistics.quantiles(gaps_ms, n=20)[-1]:.1f}  max={max(gaps_ms):.1f}"
        )
    if missing:
        print("缺键统计（缺键会造成折线断点）:")
        for key, count in sorted(missing.items(), key=lambda kv: -kv[1]):
            print(f"  {key:12s} {count}")
    else:
        print("缺键统计: 无，所有字段每拍都在发")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Requests per second against server.py, at several concurrency levels.

    bench/benchlock.sh -- .venv/bin/python bench/bench_rps.py --concurrency 1 2 4 8

Starts nothing: point it at a running server. Each client loops over the
20-sentence set and posts one request at a time; a level runs for
``--seconds`` after a short warm-up. Reported per level:

  RPS          completed requests / wall seconds
  x realtime   generated audio seconds / wall seconds (what a TTS service
               actually cares about: how many streams it can feed)
  p50 / p95    end-to-end latency seen by the client, queueing included
  batch        mean batch size the server merged those requests into
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import ASR_SET  # noqa: E402

TEXTS = list(ASR_SET.values())


def post(url: str, text: str) -> tuple[float, float, int, int]:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        wav = r.read()
        batch = int(r.headers.get("X-Batch-Size", 1))
    lat = time.perf_counter() - t0
    audio_s = max(len(wav) - 44, 0) / 2 / 24000          # 16-bit mono 24 kHz
    return lat, audio_s, batch, len(wav)


def run_level(url: str, n_clients: int, seconds: float) -> dict:
    stop = threading.Event()
    lat: list[float] = []
    audio: list[float] = []
    batches: list[int] = []
    lock = threading.Lock()

    def client(i: int):
        k = i
        while not stop.is_set():
            try:
                l, a, b, _ = post(url, TEXTS[k % len(TEXTS)])
            except Exception as e:                        # noqa: BLE001
                print(f"  client {i}: {e}", flush=True)
                return
            k += n_clients
            with lock:
                lat.append(l); audio.append(a); batches.append(b)

    threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(n_clients)]
    post(url, "预热。")
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=120)
    wall = time.perf_counter() - t0
    if not lat:
        return {"clients": n_clients, "n": 0}
    s = sorted(lat)
    return {"clients": n_clients, "n": len(lat), "wall": wall,
            "rps": len(lat) / wall, "xrt": sum(audio) / wall,
            "p50": s[len(s) // 2], "p95": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
            "mean_batch": statistics.mean(batches)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/tts")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    print(f"{'clients':>7} {'req':>5} {'RPS':>6} {'x realtime':>11} {'p50 ms':>8} {'p95 ms':>8} {'batch':>6}")
    rows = []
    for c in args.concurrency:
        r = run_level(args.url, c, args.seconds)
        rows.append(r)
        if not r["n"]:
            print(f"{c:>7}      failed"); continue
        print(f"{c:>7} {r['n']:>5} {r['rps']:>6.2f} {r['xrt']:>11.1f} "
              f"{r['p50'] * 1000:>8.0f} {r['p95'] * 1000:>8.0f} {r['mean_batch']:>6.2f}", flush=True)
    Path("out").mkdir(exist_ok=True)
    Path("out/rps.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

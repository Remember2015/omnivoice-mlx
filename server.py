"""A single-process HTTP server for the port, with dynamic batching.

    OMNIVOICE_REF_WAV=my.wav OMNIVOICE_REF_TEXT="…" .venv/bin/python server.py --model models/mlx-q8-fp16

    curl -s localhost:8080/tts -d '{"text":"今天天气不错。"}' -o out.wav

One worker thread owns the model and the GPU; the HTTP handler threads only
queue work and wait. That is not a scaling choice, it is the only correct one:
MLX has one GPU stream and an array built lazily on one thread cannot be
evaluated on another. The worker drains whatever is already queued (up to
``--max-batch``) and runs those through ``generate_batch``, so concurrency
turns into batching instead of contention.

Endpoints: POST /tts {"text": …} → audio/wav；GET /health → JSON。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent


@dataclass
class Job:
    text: str
    done: threading.Event = field(default_factory=threading.Event)
    audio: np.ndarray | None = None
    error: str | None = None
    queued_at: float = field(default_factory=time.perf_counter)
    synth_ms: float = 0.0
    batch_size: int = 1


class Engine(threading.Thread):
    """Owns the model. Everything MLX happens here."""

    def __init__(self, model: str, ref_wav: str, ref_text: str, max_batch: int, steps: int | None):
        super().__init__(daemon=True)
        self.model, self.ref_wav, self.ref_text = model, ref_wav, ref_text
        self.max_batch, self.steps = max_batch, steps
        self.q: queue.Queue[Job | None] = queue.Queue()
        self.ready = threading.Event()
        self.stats = {"requests": 0, "batches": 0, "batched_items": 0, "audio_s": 0.0}

    def run(self):
        from omnivoice_mlx import OmniVoiceTTS, SamplerConfig
        t0 = time.perf_counter()
        self.tts = OmniVoiceTTS(self.model)
        self.voice = self.tts.make_prompt(self.ref_wav, self.ref_text)
        self.cfg = SamplerConfig(num_steps=self.steps) if self.steps else SamplerConfig()
        self.tts.generate("你好。", self.voice, sampler=self.cfg, postprocess=False)   # warm up
        print(f"model ready in {time.perf_counter() - t0:.1f}s, max_batch={self.max_batch}", flush=True)
        self.ready.set()

        while True:
            job = self.q.get()
            if job is None:
                return
            batch = [job]
            while len(batch) < self.max_batch:          # 已经排着的一起做，没排着的不等
                try:
                    nxt = self.q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self.q.put(None)
                    break
                batch.append(nxt)
            self._run(batch)

    def _run(self, batch: list[Job]):
        t0 = time.perf_counter()
        try:
            if len(batch) == 1:
                results = [self.tts.generate(batch[0].text, self.voice, sampler=self.cfg)]
            else:
                results = self.tts.generate_batch([j.text for j in batch], self.voice,
                                                  sampler=self.cfg, max_batch=self.max_batch)
        except Exception as e:                           # noqa: BLE001 — 一个请求不该拖垮服务
            for j in batch:
                j.error = f"{type(e).__name__}: {e}"
                j.done.set()
            return
        ms = (time.perf_counter() - t0) * 1000
        self.stats["requests"] += len(batch)
        self.stats["batches"] += 1
        self.stats["batched_items"] += len(batch) if len(batch) > 1 else 0
        for j, r in zip(batch, results):
            j.audio, j.synth_ms, j.batch_size = r.audio, ms, len(batch)
            self.stats["audio_s"] += len(r.audio) / 24000
            j.done.set()

    def submit(self, text: str) -> Job:
        job = Job(text)
        self.q.put(job)
        return job


def wav_bytes(audio: np.ndarray, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # 默认会把每条请求打到 stderr，压测时是噪声
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, b"not found\n", "text/plain")
            s = dict(engine.stats, queued=engine.q.qsize())
            self._send(200, json.dumps(s).encode(), "application/json")

        def do_POST(self):
            if self.path != "/tts":
                return self._send(404, b"not found\n", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            try:
                text = json.loads(self.rfile.read(n))["text"]
            except Exception:                             # noqa: BLE001
                return self._send(400, json.dumps({"error": 'body must be {"text": "..."}'}).encode(),
                                  "application/json")
            job = engine.submit(text)
            job.done.wait()
            if job.error:
                return self._send(500, json.dumps({"error": job.error}).encode(), "application/json")
            body = wav_bytes(job.audio)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Synth-Ms", f"{job.synth_ms:.0f}")
            self.send_header("X-Batch-Size", str(job.batch_size))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-batch", type=int, default=4, help="1 关掉攒批")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 SamplerConfig 的 num_steps")
    args = ap.parse_args()

    ref_wav, ref_text = os.environ.get("OMNIVOICE_REF_WAV"), os.environ.get("OMNIVOICE_REF_TEXT")
    if not ref_wav or not ref_text:
        raise SystemExit("set OMNIVOICE_REF_WAV and OMNIVOICE_REF_TEXT")

    engine = Engine(args.model, ref_wav, ref_text, args.max_batch, args.steps)
    engine.start()
    engine.ready.wait()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

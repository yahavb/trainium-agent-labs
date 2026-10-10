"""Resident restyling daemon for the live demo. Runs on the pod and listens on 127.0.0.1:9000.

Four worker processes, one per NeuronCore, reuse the compiled tier1 artifacts. The policy is
real time: a worker never holds more than one frame, at most one frame waits, and when a
newer frame arrives the waiting one is dropped. Frames that barely differ from the last
processed frame are not run at all (the client reuses the last output).
"""
import os

# The pod's cgroup allows 11 CPUs while 192 are visible; keep every process to a few threads.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "2"

import argparse
import json
import multiprocessing as mp
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque

import cv2
import numpy as np

from common import SIZE, SMOOTH, STYLE_STRENGTH, STYLES
from proto import pack, read_msg

HOST, PORT = "127.0.0.1", 9000
CORES = (0, 1, 2, 3)
# neuron-monitor numbers this pod's cores 2 ahead of NEURON_RT_VISIBLE_CORES (wrapping at 4):
# load on visible core 1 shows up under monitor index 3.
MONITOR_INDEX_OFFSET = 2
FPS_WINDOW_S = 2.0
RATE_WINDOW_S = 5.0


def frame_change(a, b):
    """Largest mean abs difference (0..255) over an 8x8 grid of blocks of two reduced frames.

    A whole-image mean hides local motion: on a webcam a moving face changes it by well under
    1 while the static background dominates, so the filter keys on the most-changed block.
    """
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2).astype(np.float32)
    return float(cv2.resize(diff, (8, 8), interpolation=cv2.INTER_AREA).max())


def worker_main(idx, core, in_q, out_q, jpeg_quality, smooth):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    from pipeline_neuron import NeuronTurbo

    pipe = NeuronTurbo(smooth=smooth, strength=STYLE_STRENGTH["anime"])
    pipe.warmup(10)
    out_q.put(("ready", idx))
    while True:
        item = in_q.get()
        if item is None:
            break
        fid, gen, style, jpeg, prev_x0 = item
        t0 = time.perf_counter()
        out_jpeg, x0, error = None, None, None
        try:
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("could not decode jpeg")
            if bgr.shape[:2] != (SIZE, SIZE):
                bgr = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            if style != pipe.style:
                pipe.set_style(style)
                pipe.set_strength(STYLE_STRENGTH[style])
            # The previous output latent comes from the server: with four workers the frame
            # before this one was handled by another process.
            pipe.prev_x0 = torch.from_numpy(prev_x0) if prev_x0 is not None else None
            out = pipe(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            x0 = pipe.prev_x0.numpy()
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise ValueError("could not encode jpeg")
            out_jpeg = enc.tobytes()
        except Exception as e:  # noqa: BLE001  (a bad frame must not kill the worker)
            error = repr(e)
        out_q.put(("frame", idx, fid, gen, out_jpeg, x0, style, (time.perf_counter() - t0) * 1e3, error))


class CoreMonitor(threading.Thread):
    """Background neuron-monitor reader; .util[i] is the utilization (%) of visible core i."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util = [0.0] * len(CORES)
        self.updated = 0.0
        self.proc = None

    def run(self):
        cfg = {"period": "1s", "system_metrics": [],
               "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"}]}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
        while True:
            try:
                self.proc = subprocess.Popen(["neuron-monitor", "-c", f.name], stdout=subprocess.PIPE,
                                             stderr=subprocess.DEVNULL, text=True)
                for line in self.proc.stdout:
                    self.parse(line)
            except Exception as e:  # noqa: BLE001
                print(f"neuron-monitor reader: {e!r}", flush=True)
            time.sleep(2)

    def parse(self, line):
        try:
            data = json.loads(line)
        except ValueError:
            return
        by_index = {}
        for runtime in data.get("neuron_runtime_data") or []:
            counters = (runtime.get("report") or {}).get("neuroncore_counters") or {}
            for index, core in (counters.get("neuroncores_in_use") or {}).items():
                by_index[int(index)] = max(by_index.get(int(index), 0.0), core["neuroncore_utilization"])
        n = len(CORES)
        self.util = [round(by_index.get((c + MONITOR_INDEX_OFFSET) % n, 0.0), 1) for c in CORES]
        self.updated = time.time()


class Server:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()        # guards all scheduling state below
        self.send_lock = threading.Lock()
        self.monitor = CoreMonitor()
        self.conn, self.gen = None, 0
        self.style = "anime"
        ctx = mp.get_context("spawn")
        self.out_q = ctx.Queue()
        self.in_qs = [ctx.Queue() for _ in CORES]
        self.procs = [ctx.Process(target=worker_main, args=(i, c, self.in_qs[i], self.out_q, args.jpeg_quality, args.smooth),
                                  daemon=True) for i, c in enumerate(CORES)]
        self.reset()

    def reset(self):
        """Per-connection state."""
        self.idle = deque(range(len(CORES)))
        self.busy = {}                       # worker idx -> frame id
        self.pending = None                  # (id, style, jpeg, small, t_recv): the one waiting frame
        self.t_recv = {}                     # frame id -> receive time, for frames inside workers
        self.last_small, self.last_style = None, None
        self.last_x0, self.last_x0_style = None, None   # newest finished output latent
        self.last_sent_id = -1
        self.counts = {"received": 0, "processed": 0, "skipped": 0, "dropped": 0, "stale": 0, "errors": 0}
        self.done_times = deque()            # (t, was_processed) for the rolling rates
        self.recv_times = deque()            # (t, was_skipped)

    # ---- sending -------------------------------------------------------------------------
    def send(self, gen, header, payload=b""):
        with self.send_lock:
            conn = self.conn
            if conn is None or gen != self.gen:
                return
            try:
                conn.sendall(pack(header, payload))
            except OSError as e:
                print(f"send failed, closing connection: {e!r}", flush=True)
                self.conn = None
                self.close_conn(conn)

    def close_conn(self, conn):
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()

    def stats(self, now, compute_ms=None, service_ms=None, worker=None):
        while self.done_times and now - self.done_times[0][0] > FPS_WINDOW_S:
            self.done_times.popleft()
        while self.recv_times and now - self.recv_times[0][0] > RATE_WINDOW_S:
            self.recv_times.popleft()
        processed = sum(1 for _, p in self.done_times if p)
        c = self.counts
        out = {
            "fps": round(processed / FPS_WINDOW_S, 1),
            "fps_delivered": round(len(self.done_times) / FPS_WINDOW_S, 1),
            "skip_rate": round(sum(1 for _, s in self.recv_times if s) / max(len(self.recv_times), 1), 3),
            "skip_rate_total": round(c["skipped"] / max(c["received"], 1), 3),
            "core_util": self.monitor.util, "busy_workers": len(self.busy), **c,
        }
        if compute_ms is not None:
            out["compute_ms"] = round(compute_ms, 1)
        if service_ms is not None:
            out["service_ms"] = round(service_ms, 1)
            out["core"] = CORES[worker]
        return out

    # ---- scheduling (call with self.lock held) ---------------------------------------------
    def dispatch(self, fid, style, jpeg, small, t_recv):
        w = self.idle.popleft()
        self.busy[w] = fid
        self.t_recv[fid] = t_recv
        self.last_small, self.last_style = small, style
        prev_x0 = self.last_x0 if self.last_x0_style == style else None
        self.in_qs[w].put((fid, self.gen, style, jpeg, prev_x0))

    def on_frame(self, gen, header, jpeg):
        t_recv = time.perf_counter()
        fid, style = int(header["id"]), header.get("style") or self.style
        if style not in STYLES:
            style = self.style
        small = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        replies = []
        with self.lock:
            if gen != self.gen:
                return
            self.counts["received"] += 1
            similar = (small is not None and self.last_small is not None and style == self.last_style
                       and small.shape == self.last_small.shape and self.args.skip_threshold > 0
                       and frame_change(small, self.last_small) < self.args.skip_threshold)
            status = None
            if small is None:
                self.counts["errors"] += 1
                status = "error"
            elif similar:
                self.counts["skipped"] += 1
                status = "skipped"
            if status is None or status == "skipped":
                # the waiting frame is older than this one: it must never be shown after it
                if self.pending is not None and (status == "skipped" or not self.idle):
                    self.counts["dropped"] += 1
                    replies.append((self.pending[0], "dropped"))
                    self.pending = None
            if status is None:
                if self.idle:
                    self.dispatch(fid, style, jpeg, small, t_recv)
                else:
                    self.pending = (fid, style, jpeg, small, t_recv)
            now = time.perf_counter()
            self.recv_times.append((now, status == "skipped"))
            if status == "skipped":
                self.done_times.append((now, False))
            if status is not None:
                replies.append((fid, status))
            stats = self.stats(now, compute_ms=(now - t_recv) * 1e3) if replies else None
        for rid, rstatus in replies:
            self.send(gen, {"type": "result", "id": rid, "status": rstatus, "stats": stats})

    def on_worker_result(self, w, fid, gen, jpeg, x0, style, service_ms, error):
        with self.lock:
            if gen != self.gen:
                # result for a connection that is gone; reset() already rebuilt the idle list,
                # so only put the worker back if it is still marked busy with this frame
                if self.busy.get(w) == fid:
                    del self.busy[w]
                    self.idle.append(w)
                return
            self.busy.pop(w, None)
            self.idle.append(w)
            t_recv = self.t_recv.pop(fid, None)
            if self.pending is not None:
                pending, self.pending = self.pending, None
                self.dispatch(*pending)
            now = time.perf_counter()
            if error is not None:
                self.counts["errors"] += 1
                status = "error"
                print(f"worker {w} frame {fid}: {error}", flush=True)
            elif fid < self.last_sent_id:
                self.counts["stale"] += 1
                status = "stale"
            else:
                self.counts["processed"] += 1
                self.done_times.append((now, True))
                self.last_sent_id = fid
                self.last_x0, self.last_x0_style = x0, style
                status = "ok"
            compute_ms = (now - t_recv) * 1e3 if t_recv is not None else None
            stats = self.stats(now, compute_ms, service_ms, w)
        self.send(gen, {"type": "result", "id": fid, "status": status, "stats": stats},
                  jpeg if status == "ok" else b"")

    # ---- threads -------------------------------------------------------------------------
    def collector(self):
        while True:
            _, w, fid, gen, jpeg, x0, style, service_ms, error = self.out_q.get()
            self.on_worker_result(w, fid, gen, jpeg, x0, style, service_ms, error)

    def reader(self, conn, gen):
        try:
            while True:
                msg = read_msg(conn.recv)
                if msg is None:
                    break
                header, payload = msg
                if header.get("type") == "frame":
                    self.on_frame(gen, header, payload)
                elif header.get("type") == "control":
                    if header.get("style") in STYLES:
                        self.style = header["style"]
                        print(f"style -> {self.style}", flush=True)
        except (OSError, ValueError) as e:
            print(f"connection {gen} reader: {e!r}", flush=True)
        with self.lock:
            counts = dict(self.counts) if gen == self.gen else None
        print(f"connection {gen} closed {json.dumps(counts) if counts else ''}", flush=True)
        with self.send_lock:
            if self.conn is conn:
                self.conn = None
        self.close_conn(conn)

    def serve(self):
        t0 = time.perf_counter()
        self.monitor.start()
        for p in self.procs:
            p.start()
        for _ in self.procs:
            kind, idx = self.out_q.get(timeout=900)
            assert kind == "ready", kind
            print(f"worker {idx} on core {CORES[idx]} ready", flush=True)
        threading.Thread(target=self.collector, daemon=True).start()

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((HOST, PORT))
        listener.listen(4)
        print(f"READY on {HOST}:{PORT} after {time.perf_counter() - t0:.1f}s "
              f"(skip_threshold={self.args.skip_threshold}, jpeg_quality={self.args.jpeg_quality}, "
              f"smooth={self.args.smooth}, strength={STYLE_STRENGTH})", flush=True)
        while True:
            conn, _ = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(None)
            # one client at a time: a new connection (e.g. after a dropped exec stream) replaces the old
            with self.send_lock:
                old, self.conn = self.conn, None
            if old is not None:
                self.close_conn(old)
            with self.lock:
                self.gen += 1
                gen = self.gen
                busy = dict(self.busy)
                self.reset()
                # workers still running a frame of the old connection rejoin when they report back
                self.busy = busy
                self.idle = deque(i for i in range(len(CORES)) if i not in busy)
            with self.send_lock:
                self.conn = conn
            print(f"connection {gen} opened", flush=True)
            self.send(gen, {"type": "hello", "styles": list(STYLES), "style": self.style, "size": SIZE})
            threading.Thread(target=self.reader, args=(conn, gen), daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-threshold", type=float, default=2.0,
                    help="a frame reuses the last output when no block of it differs from the last processed "
                         "frame by more than this mean abs pixel difference (0..255); 0 disables the filter")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--smooth", type=float, default=SMOOTH,
                    help="blend of each predicted latent with the newest finished output latent (0 = off)")
    args = ap.parse_args()
    cv2.setNumThreads(1)
    server = Server(args)

    def shutdown(*_):
        # workers hold the NeuronCores, so they must not outlive the daemon
        for p in server.procs:
            if p.is_alive():
                p.terminate()
        if server.monitor.proc is not None:
            server.monitor.proc.terminate()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    server.serve()


if __name__ == "__main__":
    main()

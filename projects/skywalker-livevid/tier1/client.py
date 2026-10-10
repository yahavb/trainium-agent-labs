"""Live webcam client (runs on the Mac). Sends JPEG frames to tier1/server.py on the pod through
`kubectl exec -i ... relay.py` and shows input | styled side by side.

Keys: 1-4 switch style, q quits.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from proto import pack, read_msg

SIZE = 512
STYLES = ["anime", "claymation", "oil_painting", "pixel_art"]
MAX_IN_FLIGHT = 16
IN_FLIGHT_TIMEOUT_S = 3.0
FPS_WINDOW_S = 2.0
RELAY_CMD = ["kubectl", "exec", "-i", "seat-233", "-c", "app", "--",
             "/workspace/venvs/tnx/bin/python", "-u", "/workspace/livevid/tier1/relay.py"]
OUT_DIR = Path(__file__).resolve().parent / "out"


def percentile(values, q):
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 1) if values else None


class State:
    def __init__(self, style):
        self.lock = threading.Lock()
        self.running = True
        self.style = style
        self.capture = None            # (seq, frame_bgr, t_capture): newest camera frame
        self.capture_error = None
        self.connected = False
        self.link = None               # current relay process
        self.reconnects = 0
        self.in_flight = {}            # frame id -> t_capture
        self.styled = None             # (frame id, styled_bgr, t_capture), newest result
        self.server = {}               # last stats block from the server
        self.sent = 0
        self.results = {"ok": 0, "skipped": 0, "dropped": 0, "stale": 0, "error": 0, "timed_out": 0}
        self.delivered_times = deque()  # arrival times of ok + skipped results
        self.compute_ms, self.service_ms, self.core_util = [], [], []
        self.bytes_up = self.bytes_down = 0


def open_capture(args):
    """Must run on the main thread: macOS can only show the camera permission prompt from there."""
    source = args.source if args.source is not None else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        hint = "" if args.source is not None else (
            " (macOS: allow camera access for the app this was launched from, under "
            "System Settings > Privacy & Security > Camera, then run again)")
        raise SystemExit(f"error: cannot open video source {source!r}{hint}")
    return cap


def capture_loop(state, args, cap):
    """Keeps only the newest frame, center-cropped to a square and resized to 512."""
    from_file = args.source is not None
    period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if from_file else 0.0
    seq, next_t = 0, time.perf_counter()
    while state.running:
        ok, frame = cap.read()
        t_capture = time.perf_counter()
        if not ok:
            if from_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            state.capture_error = "camera read failed"
            state.running = False
            break
        h, w = frame.shape[:2]
        side = min(h, w)
        y, x = (h - side) // 2, (w - side) // 2
        frame = cv2.resize(frame[y:y + side, x:x + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if args.mirror:
            frame = cv2.flip(frame, 1)
        seq += 1
        with state.lock:
            state.capture = (seq, frame, t_capture)
        if from_file:
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))
    cap.release()


def sender_loop(state, args):
    """Sends the newest capture whenever fewer than --max-in-flight frames are unanswered."""
    last_seq, last_style = 0, None
    encode = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    while state.running:
        now = time.perf_counter()
        with state.lock:
            for fid in [f for f, t in state.in_flight.items() if now - t > IN_FLIGHT_TIMEOUT_S]:
                del state.in_flight[fid]
                state.results["timed_out"] += 1
            cap, link, style = state.capture, state.link, state.style
            ready = (state.connected and cap is not None and cap[0] != last_seq
                     and len(state.in_flight) < args.max_in_flight)
        if not ready:
            time.sleep(0.002)
            continue
        seq, frame, t_capture = cap
        last_seq = seq
        ok, jpeg = cv2.imencode(".jpg", frame, encode)
        if not ok:
            continue
        data = b""
        if style != last_style:
            data += pack({"type": "control", "style": style})
            last_style = style
        data += pack({"type": "frame", "id": seq, "style": style}, jpeg.tobytes())
        with state.lock:
            state.in_flight[seq] = t_capture
        try:
            link.stdin.write(data)
            link.stdin.flush()
            with state.lock:
                state.sent += 1
                state.bytes_up += len(data)
        except (OSError, ValueError):
            # the stream died; link_loop notices the EOF and respawns
            with state.lock:
                state.in_flight.pop(seq, None)
            last_style = None
            time.sleep(0.05)


def handle_result(state, header, payload):
    stats = header.get("stats") or {}
    status = header.get("status")
    styled = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR) if payload else None
    now = time.perf_counter()
    with state.lock:
        t_capture = state.in_flight.pop(header["id"], None)
        state.bytes_down += len(payload)
        if status in state.results:
            state.results[status] += 1
        if stats:
            state.server = stats
            if stats.get("core_util"):
                state.core_util.append(stats["core_util"])
        if status in ("ok", "skipped"):
            state.delivered_times.append(now)
        if status == "ok" and styled is not None and t_capture is not None:
            if state.styled is None or header["id"] > state.styled[0]:
                state.styled = (header["id"], styled, t_capture)
            if "compute_ms" in stats:
                state.compute_ms.append(stats["compute_ms"])
            if "service_ms" in stats:
                state.service_ms.append(stats["service_ms"])


def link_loop(state, args):
    """Owns the kubectl exec relay: reads results, and respawns the stream whenever it dies."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    relay_log = open(OUT_DIR / "client_relay.log", "ab")
    first = True
    while state.running:
        started = time.perf_counter()
        link = subprocess.Popen(RELAY_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=relay_log, bufsize=0)
        with state.lock:
            state.link = link
        try:
            while state.running:
                msg = read_msg(link.stdout.read)
                if msg is None:
                    break
                header, payload = msg
                if header.get("type") == "hello":
                    with state.lock:
                        state.connected = True
                elif header.get("type") == "result":
                    handle_result(state, header, payload)
        except (OSError, ValueError) as e:
            relay_log.write(f"client: stream error {e!r}\n".encode())
        with state.lock:
            was_connected, state.connected = state.connected, False
            state.in_flight.clear()
            if state.running and (was_connected or not first):
                state.reconnects += 1
        first = False
        try:
            link.kill()
            link.wait(timeout=2)
        except Exception:  # noqa: BLE001
            pass
        if state.running:
            relay_log.write(f"client: stream ended rc={link.returncode}, respawning\n".encode())
            relay_log.flush()
            # respawn about 1 s after the previous spawn at the latest, without spinning
            time.sleep(max(0.2, min(1.0, 1.0 - (time.perf_counter() - started))))
    with state.lock:
        link = state.link
    if link is not None and link.poll() is None:
        try:
            link.stdin.close()
            link.wait(timeout=2)
        except Exception:  # noqa: BLE001
            link.kill()


def text(img, s, org, scale=0.55, color=(255, 255, 255)):
    # A dark box rather than an outline: newer OpenCV builds render thick text wider than thin
    # text, so an outlined label does not line up with itself.
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    box = img[max(y - h - 4, 0):y + base + 2, max(x - 4, 0):x + w + 4]
    box[:] = box // 4
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_core_bars(canvas, util):
    x0, y0, bar_w, bar_h, gap = SIZE * 2 - 4 * 26 - 8, SIZE - 12, 18, 60, 8
    for i in range(4):
        u = float(util[i]) if i < len(util) else 0.0
        x = x0 + i * (bar_w + gap)
        cv2.rectangle(canvas, (x, y0 - bar_h), (x + bar_w, y0), (40, 40, 40), -1)
        filled = int(bar_h * min(max(u, 0.0), 100.0) / 100.0)
        cv2.rectangle(canvas, (x, y0 - filled), (x + bar_w, y0), (80, 220, 120), -1)
        cv2.rectangle(canvas, (x, y0 - bar_h), (x + bar_w, y0), (230, 230, 230), 1)
        text(canvas, str(i), (x + 4, y0 - bar_h - 5), 0.4)
    text(canvas, "core util", (x0 - 78, y0 - 4), 0.42)


class Recorder:
    """Writes BGR frames to an mp4 through ffmpeg (libx264). OpenCV's own writer on macOS uses a
    low bitrate that smears the overlay text over the fast-changing styled half."""

    def __init__(self, path, fps=30.0):
        self.proc, self.writer = None, None
        size = f"{SIZE * 2}x{SIZE}"
        try:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", size,
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
            self.kind = "ffmpeg libx264"
        except FileNotFoundError:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (SIZE * 2, SIZE))
            if not self.writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {path}")
            self.kind = "opencv avc1"

    def write(self, frame):
        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except OSError:
                pass
        else:
            self.writer.write(frame)

    def release(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        else:
            self.writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", default=None, help="video file to use instead of the webcam (looped)")
    ap.add_argument("--style", default="anime", choices=STYLES)
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument("--max-in-flight", type=int, default=MAX_IN_FLIGHT,
                    help="frames sent but not yet answered; the round trip through kubectl exec is long, "
                         "so this caps the frame rate at max_in_flight / round_trip")
    ap.add_argument("--record", nargs="?", const="", default=None, metavar="PATH",
                    help="save the side-by-side view to an mp4 (default tier1/out/live_<time>.mp4)")
    ap.add_argument("--duration", type=float, default=0, help="stop after this many seconds (0 = until q)")
    ap.add_argument("--cycle-styles", type=float, default=0, metavar="SECONDS",
                    help="switch to the next style automatically every SECONDS (0 = keys only)")
    ap.add_argument("--no-mirror", dest="mirror", action="store_false", help="do not flip the image horizontally")
    ap.add_argument("--no-window", action="store_true", help="run without a window (for automated tests)")
    ap.add_argument("--stats-out", default=str(OUT_DIR / "live_stats.json"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = open_capture(args)
    state = State(args.style)
    threads = [threading.Thread(target=capture_loop, args=(state, args, cap), daemon=True)]
    threads += [threading.Thread(target=f, args=(state, args), daemon=True) for f in (sender_loop, link_loop)]
    for t in threads:
        t.start()

    writer, record_path = None, None
    if args.record is not None:
        record_path = Path(args.record or OUT_DIR / f"live_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        writer = Recorder(record_path)
        print(f"recording to {record_path} ({writer.kind})", flush=True)

    shown_id, styled_img = None, np.zeros((SIZE, SIZE, 3), np.uint8)
    shown_times, g2g_ms, fps_samples = deque(), [], []
    t_start = time.perf_counter()
    t_first_frame, next_record = None, t_start
    canvas = np.zeros((SIZE, SIZE * 2, 3), np.uint8)
    try:
        while state.running:
            now = time.perf_counter()
            if args.duration and t_first_frame is not None and now - t_first_frame >= args.duration:
                break
            if args.cycle_styles and t_first_frame is not None:
                with state.lock:
                    state.style = STYLES[(STYLES.index(args.style) + int((now - t_first_frame) / args.cycle_styles))
                                         % len(STYLES)]
            with state.lock:
                cap, styled, server = state.capture, state.styled, dict(state.server)
                connected, reconnects, style = state.connected, state.reconnects, state.style
                results = dict(state.results)
                last_compute = state.compute_ms[-1] if state.compute_ms else 0.0
                while state.delivered_times and now - state.delivered_times[0] > FPS_WINDOW_S:
                    state.delivered_times.popleft()
                delivered_fps = len(state.delivered_times) / FPS_WINDOW_S
            new_frame_t = None
            if styled is not None and styled[0] != shown_id:
                shown_id, styled_img, new_frame_t = styled

            canvas[:, :SIZE] = cap[1] if cap is not None else 0
            canvas[:, SIZE:] = styled_img
            while shown_times and now - shown_times[0] > FPS_WINDOW_S:
                shown_times.popleft()
            fps = len(shown_times) / FPS_WINDOW_S
            recent = g2g_ms[-30:]
            text(canvas, "input", (10, 24))
            text(canvas, f"{style}  [1-4 style, q quit]", (SIZE + 10, 24))
            text(canvas, f"{fps:4.1f} FPS styled   {delivered_fps:4.1f} incl. reused", (SIZE + 10, 48))
            text(canvas, f"glass-to-glass {np.median(recent):4.0f} ms" if recent else "glass-to-glass --",
                 (SIZE + 10, 70))
            text(canvas, f"server compute {last_compute:4.0f} ms   "
                         f"skip {100 * server.get('skip_rate', 0):3.0f}%", (SIZE + 10, 92))
            text(canvas, f"dropped {results['dropped']}  reconnects {reconnects}", (SIZE + 10, 114), 0.45)
            draw_core_bars(canvas, server.get("core_util") or [])
            if not connected:
                text(canvas, "reconnecting...", (SIZE + 150, SIZE // 2), 0.9, (60, 60, 255))

            if not args.no_window:
                cv2.imshow("livevid tier1: input | sd-turbo on 4 NeuronCores", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if ord("1") <= key <= ord("4"):
                    with state.lock:
                        state.style = STYLES[key - ord("1")]
            else:
                time.sleep(0.004)

            done = time.perf_counter()
            if new_frame_t is not None:
                # capture timestamp (right after the camera read returned) -> frame on screen
                g2g_ms.append((done - new_frame_t) * 1e3)
                shown_times.append(done)
                if t_first_frame is None:
                    t_first_frame = done
            if t_first_frame is not None and done - t_first_frame > FPS_WINDOW_S:
                fps_samples.append(fps)
            # nothing is recorded until the first styled frame is on screen
            if writer is not None and t_first_frame is not None and done >= next_record:
                writer.write(canvas)
                next_record = max(next_record + 1 / 30.0, done - 0.1)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        t_end = time.perf_counter()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        for t in threads:
            t.join(timeout=3)

    if state.capture_error:
        print(f"error: {state.capture_error}")
    active = (t_end - t_first_frame) if t_first_frame is not None else 0.0
    util = np.asarray(state.core_util, dtype=np.float64) if state.core_util else np.zeros((1, 4))
    summary = {
        "source": args.source or f"camera {args.camera}", "max_in_flight": args.max_in_flight, "duration_s": round(t_end - t_start, 1),
        "active_s": round(active, 1), "frames_sent": state.sent, "results": state.results,
        "styled_frames_shown": len(g2g_ms),
        "end_to_end_fps": round(len(g2g_ms) / active, 1) if active > 0 else 0.0,
        "end_to_end_fps_rolling_p50": percentile(fps_samples, 50),
        "glass_to_glass_ms": {"p50": percentile(g2g_ms, 50), "p99": percentile(g2g_ms, 99)},
        "server_compute_ms": {"p50": percentile(state.compute_ms, 50), "p99": percentile(state.compute_ms, 99)},
        "worker_service_ms": {"p50": percentile(state.service_ms, 50), "p99": percentile(state.service_ms, 99)},
        "core_util_mean_pct": [round(float(v), 1) for v in util.mean(axis=0)],
        "core_util_max_pct": [round(float(v), 1) for v in util.max(axis=0)],
        "skip_rate": state.server.get("skip_rate_total"), "server_counts": {
            k: state.server.get(k) for k in ("received", "processed", "skipped", "dropped", "stale", "errors")},
        "reconnects": state.reconnects,
        "upload_mb_s": round(state.bytes_up / 1e6 / max(active, 1e-6), 2),
        "download_mb_s": round(state.bytes_down / 1e6 / max(active, 1e-6), 2),
        "recording": str(record_path) if record_path else None,
    }
    Path(args.stats_out).write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
    if state.link is not None and state.link.poll() is None:
        state.link.kill()
    os._exit(0)  # the kubectl child and reader threads may still be blocked in I/O


if __name__ == "__main__":
    main()

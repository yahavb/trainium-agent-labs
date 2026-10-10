"""Wire format shared by server.py and client.py (stdlib only; relay.py just copies bytes).

Each message is: uint32 header length, uint32 payload length (big endian), a JSON header,
then the raw payload (JPEG bytes, possibly empty).

  client -> server  {"type": "frame", "id": int, "style": str} + jpeg
                    {"type": "control", "style": str}
  server -> client  {"type": "hello", "styles": [...]}
                    {"type": "result", "id": int, "status": str, "stats": {...}} + jpeg
                    status is "ok" (jpeg attached), "skipped" (similar to the last processed
                    frame: reuse the last output), "dropped" (superseded by a newer frame),
                    "stale" (finished after a newer result was already sent) or "error".
"""
import json
import struct

HEAD = struct.Struct(">II")
MAX_HEADER = 1 << 16
MAX_PAYLOAD = 1 << 24


def pack(header, payload=b""):
    h = json.dumps(header, separators=(",", ":")).encode()
    return HEAD.pack(len(h), len(payload)) + h + payload


def read_exact(read, n):
    """Reads exactly n bytes with read(k) -> up to k bytes; returns None on EOF."""
    chunks, got = [], 0
    while got < n:
        chunk = read(n - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_msg(read):
    """Returns (header dict, payload bytes), or None on EOF."""
    head = read_exact(read, HEAD.size)
    if head is None:
        return None
    hlen, plen = HEAD.unpack(head)
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise ValueError(f"corrupt stream: header {hlen} bytes, payload {plen} bytes")
    h = read_exact(read, hlen)
    if h is None:
        return None
    payload = read_exact(read, plen) if plen else b""
    if payload is None:
        return None
    return json.loads(h), payload

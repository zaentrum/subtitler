"""VobSub (DVD subpictures, `.idx` + `.sub`): what a track shows, when, as bitmaps.

The .idx is text: the video's size, a palette of 16 RGB colours, and per
subtitle its timestamp and the position in the .sub where it begins (after a
`delay:` line, shifted by it). The .sub is an MPEG program stream: packs of
private stream 1 whose payloads, for the track's substream (0x20 + index),
carry a subpicture unit (SPU) in one or more packets: its size, where its
control sequences begin, its pixels and the control sequences:

- 0x00 forced start, 0x01 start and 0x02 stop of the display, each after its
  sequence's delay (1024 ticks of 90 kHz a unit);
- 0x03 the four colours (palette indices) and 0x04 their contrast (alpha,
  0-15) of the pixel values 3, 2, 1, 0;
- 0x05 the display area, 0x06 where the top and bottom fields' pixels begin;
- 0xFF the end of a sequence.

The pixels are two bits each, run-length coded by nibbles, a line at a time,
the top field's lines (0, 2, 4, ...) apart from the bottom field's.

The pictures are pgs.Picture, so the OCR step reads either the same way.
Pure Python and numpy; nothing is written.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from .pgs import Picture, Placed

TICK = 1024 / 90_000.0  # a control sequence's delay unit, in seconds


def read_idx(text: str) -> dict:
    """The .idx: size, palette (16 x RGB), and the timestamps (seconds) and file positions."""
    size, palette, entries, delay = None, [], [], 0.0
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("size:"):
            m = re.search(r"(\d+)\s*x\s*(\d+)", s)
            size = (int(m.group(1)), int(m.group(2))) if m else None
        elif s.startswith("palette:"):
            palette = [tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for c in re.findall(r"[0-9a-fA-F]{6}", s)]
        elif s.startswith("delay:"):
            m = re.search(r"(-)?(\d+):(\d+):(\d+)[:.](\d+)", s)
            if m:
                d = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000
                delay += -d if m.group(1) else d
        elif s.startswith("timestamp:"):
            m = re.search(r"timestamp:\s*(-)?(\d+):(\d+):(\d+)[:.](\d+),\s*filepos:\s*([0-9a-fA-F]+)", s)
            if m:
                t = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) + int(m.group(5)) / 1000
                entries.append((delay + (-t if m.group(1) else t), int(m.group(6), 16)))
    while len(palette) < 16:
        palette.append((0, 0, 0))
    return {"size": size or (720, 576), "palette": palette[:16], "entries": entries}


def _packets(sub: bytes, pos: int):
    """The private-stream-1 payloads from pos on: (substream id, payload)."""
    n = len(sub)
    while pos + 6 <= n:
        if sub[pos:pos + 3] != b"\x00\x00\x01":
            nxt = sub.find(b"\x00\x00\x01", pos + 1)
            if nxt < 0:
                return
            pos = nxt
            continue
        code = sub[pos + 3]
        if code == 0xBA:  # pack header: MPEG-2 (14 bytes and its stuffing) or MPEG-1 (12)
            if pos + 14 <= n and sub[pos + 4] & 0xC0 == 0x40:
                pos += 14 + (sub[pos + 13] & 0x07)
            else:
                pos += 12
            continue
        if code == 0xB9:  # program end
            return
        length = struct.unpack_from(">H", sub, pos + 4)[0]
        body = sub[pos + 6:pos + 6 + length]
        pos += 6 + length
        if code != 0xBD or len(body) < 3:
            continue
        if body[0] & 0xC0 == 0x80:  # an MPEG-2 PES header
            start = 3 + body[2]
        else:  # MPEG-1: stuffing, an optional buffer size, then its time stamps
            k = 0
            while k < len(body) and body[k] == 0xFF:
                k += 1
            if k < len(body) and body[k] & 0xC0 == 0x40:
                k += 2
            k += {0x20: 5, 0x30: 10}.get(body[k] & 0xF0, 1) if k < len(body) else 0
            start = k
        if start < len(body):
            yield body[start], body[start + 1:]


def spu_at(sub: bytes, pos: int, stream: int | None = None) -> bytes | None:
    """The subpicture unit that begins at pos (its packets joined), of the substream given or the
    first one met."""
    data = bytearray()
    want = None
    sid_wanted = 0x20 + stream if stream is not None else None
    for sid, payload in _packets(sub, pos):
        if not 0x20 <= sid <= 0x3F or (sid_wanted is not None and sid != sid_wanted):
            continue
        sid_wanted = sid
        if want is None:
            if len(payload) < 2:
                return None
            want = struct.unpack_from(">H", payload, 0)[0]
        data += payload
        if len(data) >= want:
            return bytes(data[:want])
    return bytes(data) if want and len(data) >= 4 else None


def _rle_field(spu: bytes, offset: int, width: int, lines: int, out: np.ndarray, first: int) -> None:
    """One field's lines (first, first + 2, ...) from the nibbles at offset."""
    nib = offset * 2
    end = len(spu) * 2

    def get() -> int:
        nonlocal nib
        if nib >= end:
            return 0
        b = spu[nib >> 1]
        v = (b >> 4) if not (nib & 1) else (b & 0x0F)
        nib += 1
        return v

    y = first
    while y < lines and nib < end:
        x = 0
        while x < width and nib < end:
            v = get()
            if v < 0x4:
                v = (v << 4) | get()
                if v < 0x10:
                    v = (v << 4) | get()
                    if v < 0x40:
                        v = (v << 4) | get()
            run, colour = v >> 2, v & 3
            if run == 0:  # to the end of the line
                run = width - x
            k = min(run, width - x)
            if colour:
                out[y, x:x + k] = colour
            x += k
        if nib & 1:  # each line ends on a byte
            nib += 1
        y += 2


def decode_spu(spu: bytes) -> dict | None:
    """A subpicture unit: its start and stop (seconds after its timestamp), whether forced, its
    area, its colours and contrasts, and its pixels (values 0-3)."""
    if len(spu) < 4:
        return None
    ctrl = struct.unpack_from(">H", spu, 2)[0]
    start = stop = None
    forced = False
    colours, contrast = [0, 1, 2, 3], [0, 15, 15, 15]
    area = offsets = None
    pos = ctrl
    seen = set()
    while pos + 4 <= len(spu) and pos not in seen:
        seen.add(pos)
        delay, nxt = struct.unpack_from(">HH", spu, pos)
        t = delay * TICK
        k = pos + 4
        while k < len(spu):
            cmd = spu[k]
            k += 1
            if cmd == 0x00:
                forced, start = True, t if start is None else start
            elif cmd == 0x01:
                start = t if start is None else start
            elif cmd == 0x02:
                stop = t
            elif cmd == 0x03 and k + 2 <= len(spu):
                b0, b1 = spu[k], spu[k + 1]
                colours = [b1 & 0x0F, b1 >> 4, b0 & 0x0F, b0 >> 4]  # for pixel values 0, 1, 2, 3
                k += 2
            elif cmd == 0x04 and k + 2 <= len(spu):
                b0, b1 = spu[k], spu[k + 1]
                contrast = [b1 & 0x0F, b1 >> 4, b0 & 0x0F, b0 >> 4]
                k += 2
            elif cmd == 0x05 and k + 6 <= len(spu):
                b = spu[k:k + 6]
                x1, x2 = (b[0] << 4) | (b[1] >> 4), ((b[1] & 0x0F) << 8) | b[2]
                y1, y2 = (b[3] << 4) | (b[4] >> 4), ((b[4] & 0x0F) << 8) | b[5]
                area = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
                k += 6
            elif cmd == 0x06 and k + 4 <= len(spu):
                offsets = struct.unpack_from(">HH", spu, k)
                k += 4
            elif cmd == 0x07 and k + 2 <= len(spu):
                k += struct.unpack_from(">H", spu, k)[0]
            else:  # 0xFF, or a command it does not know: the sequence ends
                break
        if nxt == pos:
            break
        pos = nxt
    if area is None or offsets is None or area[2] <= 0 or area[3] <= 0:
        return None
    pixels = np.zeros((area[3], area[2]), dtype=np.uint8)
    _rle_field(spu, offsets[0], area[2], area[3], pixels, 0)
    _rle_field(spu, offsets[1], area[2], area[3], pixels, 1)
    return {"start": start or 0.0, "stop": stop, "forced": forced, "area": area, "colours": colours,
            "contrast": contrast, "pixels": pixels}


def pictures(idx_text: str, sub: bytes, min_alpha: int = 16) -> list[Picture]:
    """What the track shows, in order: a Picture a subpicture with anything visible in it, from its
    start to its stop, or to the next one's start where it has no stop (or a later one)."""
    idx = read_idx(idx_text)
    rgb = np.array(idx["palette"], dtype=np.float32)
    luma = np.clip(0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2], 0, 255).astype(np.uint8)
    out: list[Picture] = []
    for t, pos in sorted(idx["entries"]):
        spu = spu_at(sub, pos)
        s = decode_spu(spu) if spu else None
        if s is None:
            continue
        lum = luma[np.array(s["colours"])][s["pixels"]]
        alpha = (np.array(s["contrast"], dtype=np.uint8) * 17)[s["pixels"]]
        if not (alpha >= min_alpha).any():
            continue
        x, y, _, _ = s["area"]
        start = t + s["start"]
        end = t + s["stop"] if s["stop"] is not None and s["stop"] > s["start"] else None
        out.append(Picture(start, end, [Placed(x, y, lum, alpha, s["forced"])], idx["size"], forced=s["forced"]))
    out.sort(key=lambda p: p.start)
    for a, b in zip(out, out[1:]):
        if a.end is None or a.end > b.start:
            a.end = b.start
    return out


def read(idx_path: str | Path) -> list[Picture]:
    """A VobSub track from its .idx and the .sub beside it."""
    p = Path(idx_path)
    return pictures(p.read_text(encoding="latin-1"), p.with_suffix(".sub").read_bytes())

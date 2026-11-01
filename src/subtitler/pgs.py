"""PGS (Blu-ray presentation graphics, `.sup`): what a track shows, when, as bitmaps.

A .sup file is a run of segments, each `PG`, a PTS and a DTS (90 kHz), a type
and a size:

- PCS (0x16), the presentation composition: the video's size, the composition's
  number and state (normal, acquisition point, epoch start), whether only the
  palette changes, the palette it uses, and its composition objects (the
  object, the window, the flags: 0x80 cropped, 0x40 forced; its position and
  the crop);
- WDS (0x17) the windows; PDS (0x14) a palette: entries of Y, Cr, Cb and alpha;
- ODS (0x15) an object: its id and version, first and last in its sequence
  (an object larger than a segment comes in fragments), its size and its
  pixels, run-length coded;
- END (0x80) the end of a display set.

A display set (PCS to END) shows its composition objects from its PTS until
the next display set's; one with no objects clears the screen and so ends the
one before. Palettes and objects hold within an epoch (from an epoch start
on), so a later set may show an object an earlier one defined. Sets that
show the same thing again (acquisition points, a palette-only update of a
fade) are one picture; a set whose pixels are all transparent shows nothing.

Pure Python and numpy; nothing is written.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PDS, ODS, PCS, WDS, END = 0x14, 0x15, 0x16, 0x17, 0x80
EPOCH_START, ACQUISITION_POINT = 0x80, 0x40
CLOCK = 90_000.0


class PgsError(ValueError):
    """The bytes are not a PGS stream (or end in the middle of a segment)."""


@dataclass
class Segment:
    kind: int
    pts: float  # seconds
    data: bytes


def segments(data: bytes) -> list[Segment]:
    out, i, n = [], 0, len(data)
    while i + 13 <= n:
        if data[i:i + 2] != b"PG":
            raise PgsError(f"no segment header at byte {i}")
        pts, _dts, kind, size = struct.unpack_from(">IIBH", data, i + 2)
        end = i + 13 + size
        if end > n:
            raise PgsError(f"the segment at byte {i} runs past the end of the file")
        out.append(Segment(kind, pts / CLOCK, data[i + 13:end]))
        i = end
    if i != n and data[i:].strip(b"\x00"):
        raise PgsError(f"{n - i} bytes after the last segment")
    return out


def decode_rle(data: bytes, width: int, height: int) -> np.ndarray:
    """An object's pixels (palette indices, height x width) from its run-length code. The bytes up
    to the next 0 are single pixels of their colour, copied at once; a 0 begins a run or a line's
    end."""
    buf = bytearray(width * height)
    i, n, y, x = 0, len(data), 0, 0
    find = data.find
    while i < n and y < height:
        j = find(b"\x00", i)
        if j < 0:
            j = n
        if j > i:  # pixels, one byte each
            k = min(j - i, width - x)
            if k > 0:
                at = y * width + x
                buf[at:at + k] = data[i:i + k]
            x += j - i
            i = j
            if i >= n:
                break
        i += 1  # the 0
        if i >= n:
            break
        b = data[i]
        i += 1
        if b == 0:  # end of the line
            y, x = y + 1, 0
            continue
        flag, run = b & 0xC0, b & 0x3F
        colour = 0
        if flag & 0x40:  # a 14-bit length
            run = (run << 8) | (data[i] if i < n else 0)
            i += 1
        if flag & 0x80:  # a colour other than 0
            colour = data[i] if i < n else 0
            i += 1
        if colour and x < width:
            at = y * width + x
            k = min(run, width - x)
            buf[at:at + k] = bytes((colour,)) * k
        x += run
    return np.frombuffer(bytes(buf), dtype=np.uint8).reshape(height, width)


@dataclass
class Placed:
    """An object on the screen: its pixels' luma and alpha (0-255), at x, y."""

    x: int
    y: int
    luma: np.ndarray  # uint8, Y scaled to full range
    alpha: np.ndarray  # uint8
    forced: bool = False

    @property
    def height(self) -> int:
        return self.luma.shape[0]

    @property
    def width(self) -> int:
        return self.luma.shape[1]


@dataclass
class Picture:
    """What a track shows from `start` to `end` (seconds): one or more objects."""

    start: float
    end: float | None
    objects: list[Placed]
    size: tuple[int, int]  # the video's width and height
    forced: bool = False  # every object flagged forced
    sets: int = 1  # display sets it spans (acquisition points, palette updates)
    key: tuple = field(default=(), repr=False)

    @property
    def duration(self) -> float:
        return (self.end or self.start) - self.start


@dataclass
class _Object:
    version: int
    width: int
    height: int
    data: bytearray
    complete: bool


def _palette_update(pal: np.ndarray, data: bytes) -> None:
    # pal: 256 x 4 (Y, Cr, Cb, A); entries a PDS leaves out keep their values.
    for k in range(2, len(data) - 4, 5):
        pal[data[k]] = (data[k + 1], data[k + 2], data[k + 3], data[k + 4])


def _full_range(y: np.ndarray) -> np.ndarray:
    # Blu-ray luma is limited range (16-235).
    return np.clip((y.astype(np.int32) - 16) * 255 // 219, 0, 255).astype(np.uint8)


def display_sets(data: bytes) -> Iterator[dict]:
    """The display sets of a stream, one after the other, each with the screen it leaves: its PTS,
    whether it is an epoch start, whether only its palette changes, and its objects decoded (their
    palette indices, their palette, their position)."""
    palettes: dict[int, np.ndarray] = {}
    objects: dict[int, _Object] = {}
    decoded: dict[tuple, np.ndarray] = {}
    cur: dict | None = None
    for seg in segments(data):
        d = seg.data
        if seg.kind == PCS:
            if len(d) < 11:
                raise PgsError("a composition segment too short")
            width, height, _rate, number, state, pal_update, pal_id, count = struct.unpack_from(">HHBHBBBB", d, 0)
            if state & EPOCH_START:
                palettes.clear()
                objects.clear()
                decoded.clear()
            comps, k = [], 11
            for _ in range(count):
                if k + 8 > len(d):
                    break
                oid, wid, flags, x, y = struct.unpack_from(">HBBHH", d, k)
                k += 8
                crop = None
                if flags & 0x80 and k + 8 <= len(d):
                    crop = struct.unpack_from(">HHHH", d, k)
                    k += 8
                comps.append({"object": oid, "window": wid, "x": x, "y": y, "crop": crop, "forced": bool(flags & 0x40)})
            cur = {"pts": seg.pts, "size": (width, height), "number": number, "state": state,
                   "palette_only": bool(pal_update & 0x80), "palette": pal_id, "compositions": comps}
        elif seg.kind == PDS and len(d) >= 2:
            pal = palettes.setdefault(d[0], np.zeros((256, 4), dtype=np.uint8))
            _palette_update(pal, d)
        elif seg.kind == ODS and len(d) >= 4:
            oid, version, seq = struct.unpack_from(">HBB", d, 0)
            if seq & 0x80:  # first in its sequence: the size, then the pixels
                if len(d) < 11:
                    raise PgsError("an object segment too short")
                w, h = struct.unpack_from(">HH", d, 7)
                objects[oid] = _Object(version, w, h, bytearray(d[11:]), bool(seq & 0x40))
            elif oid in objects:
                objects[oid].data += d[4:]
                objects[oid].complete = bool(seq & 0x40)
        elif seg.kind == END and cur is not None:
            screen = []
            pal = palettes.get(cur["palette"])
            for c in cur["compositions"]:
                ob = objects.get(c["object"])
                if ob is None or pal is None or not ob.width or not ob.height:
                    continue
                digest = hashlib.blake2b(bytes(ob.data), digest_size=12).digest()
                key = (c["object"], ob.version, digest, ob.width, ob.height)
                idx = decoded.get(key)
                if idx is None:
                    idx = decoded[key] = decode_rle(bytes(ob.data), ob.width, ob.height)
                x, y = c["x"], c["y"]
                if c["crop"]:
                    cx, cy, cw, ch = c["crop"]
                    idx = idx[cy:cy + ch, cx:cx + cw]
                    if not idx.size:
                        continue
                screen.append((key, c, x, y, idx))
            cur["screen"] = [(key, c, x, y, idx, pal.copy()) for key, c, x, y, idx in screen]
            yield cur
            cur = None


def pictures(data: bytes, min_alpha: int = 16) -> list[Picture]:
    """What the track shows, in order: one Picture from the display set that shows it to the one that
    clears or replaces it. Sets that show the same objects at the same places again are one picture
    (acquisition points; a palette update with anything still visible); a set whose pixels are all
    (nearly) transparent shows nothing."""
    out: list[Picture] = []
    open_: Picture | None = None
    opacity = 0
    for s in display_sets(data):
        shown, key = [], []
        for okey, c, x, y, idx, pal in s["screen"]:
            alpha = pal[:, 3][idx]
            if not (alpha >= min_alpha).any():
                continue
            shown.append(Placed(x, y, _full_range(pal[:, 0][idx]), alpha, c["forced"]))
            key.append((okey, x, y, c["crop"]))
        key = tuple(key)
        if open_ is not None and shown and key == open_.key:
            open_.sets += 1
            # A fade: the picture as it is most opaque is the one to read.
            most = max(int(p.alpha.max()) for p in shown)
            if most > opacity:
                open_.objects, opacity = shown, most
            continue
        if open_ is not None:
            open_.end = s["pts"]
            out.append(open_)
            open_ = None
        if shown:
            open_ = Picture(s["pts"], None, shown, s["size"], forced=all(p.forced for p in shown), key=key)
            opacity = max(int(p.alpha.max()) for p in shown)
    if open_ is not None:
        out.append(open_)
    return out


def read(path: str | Path) -> list[Picture]:
    return pictures(Path(path).read_bytes())


def _shows(d: bytes) -> bool:
    """A composition that shows objects and is no acquisition point or palette update."""
    return len(d) >= 11 and d[10] > 0 and not d[7] & ACQUISITION_POINT and not d[8] & 0x80


def count_bytes(data: bytes) -> int:
    """count() of a stream in memory."""
    return sum(1 for s in segments(data) if s.kind == PCS and _shows(s.data))


def count(path: str | Path) -> int:
    """How many pictures a track shows, from its composition segments alone (the objects' pixels
    skipped, not read): the sets that show objects and are no acquisition point or palette update."""
    n = 0
    with open(path, "rb") as f:
        while True:
            head = f.read(13)
            if len(head) < 13:
                if head.strip(b"\x00"):
                    raise PgsError("the file ends in the middle of a segment header")
                break
            if head[:2] != b"PG":
                raise PgsError("no segment header")
            kind, size = head[10], struct.unpack_from(">H", head, 11)[0]
            if kind != PCS:
                f.seek(size, 1)
                continue
            if _shows(f.read(size)):
                n += 1
    return n

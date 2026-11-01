"""Synthetic image subtitles for the tests: PGS display sets and VobSub subpictures, built here.

Nothing in them comes from a title: bitmaps are drawn by the tests (blocks, strokes, text rendered
with Pillow's own font where Pillow is installed).
"""

from __future__ import annotations

import struct

import numpy as np

PDS, ODS, PCS, WDS, END = 0x14, 0x15, 0x16, 0x17, 0x80


# --- PGS ------------------------------------------------------------------------

def rle(idx: np.ndarray) -> bytes:
    """PGS run-length code of a bitmap of palette indices, every form of run used."""
    out = bytearray()
    for row in idx:
        x, w = 0, len(row)
        while x < w:
            c = int(row[x])
            n = 1
            while x + n < w and row[x + n] == c and n < 16383:
                n += 1
            if c == 0:
                out += bytes([0, n]) if n < 64 else bytes([0, 0x40 | (n >> 8), n & 0xFF])
            elif n < 3:
                out += bytes([c]) * n
            elif n < 64:
                out += bytes([0, 0x80 | n, c])
            else:
                out += bytes([0, 0xC0 | (n >> 8), n & 0xFF, c])
            x += n
        out += b"\x00\x00"
    return bytes(out)


def segment(kind: int, pts: float, data: bytes) -> bytes:
    ticks = int(round(pts * 90_000))
    return b"PG" + struct.pack(">IIBH", ticks, 0, kind, len(data)) + data


def pcs(pts: float, comps: list[dict], number: int = 0, state: int = 0x80, palette_only: bool = False,
        size: tuple[int, int] = (1920, 1080)) -> bytes:
    d = struct.pack(">HHBHBBBB", size[0], size[1], 0x10, number, state, 0x80 if palette_only else 0, 0, len(comps))
    for c in comps:
        flags = (0x80 if c.get("crop") else 0) | (0x40 if c.get("forced") else 0)
        d += struct.pack(">HBBHH", c["id"], 0, flags, c["x"], c["y"])
        if c.get("crop"):
            d += struct.pack(">HHHH", *c["crop"])
    return segment(PCS, pts, d)


def wds(pts: float, windows: list[tuple[int, int, int, int]]) -> bytes:
    d = bytes([len(windows)]) + b"".join(struct.pack(">BHHHH", i, *w) for i, w in enumerate(windows))
    return segment(WDS, pts, d)


def pds(pts: float, entries: dict[int, tuple[int, int, int, int]], version: int = 0) -> bytes:
    d = bytes([0, version]) + b"".join(bytes([i, *e]) for i, e in sorted(entries.items()))
    return segment(PDS, pts, d)


def ods(pts: float, oid: int, idx: np.ndarray, version: int = 0, fragment: int = 0) -> bytes:
    """An object, in fragments of `fragment` bytes of pixels where given."""
    h, w = idx.shape
    data = rle(idx)
    size = len(data) + 4
    if not fragment or len(data) <= fragment:
        return segment(ODS, pts, struct.pack(">HBB", oid, version, 0xC0) + size.to_bytes(3, "big")
                       + struct.pack(">HH", w, h) + data)
    parts = [data[i:i + fragment] for i in range(0, len(data), fragment)]
    out = segment(ODS, pts, struct.pack(">HBB", oid, version, 0x80) + size.to_bytes(3, "big")
                  + struct.pack(">HH", w, h) + parts[0])
    for k, p in enumerate(parts[1:], 1):
        out += segment(ODS, pts, struct.pack(">HBB", oid, version, 0x40 if k == len(parts) - 1 else 0x00) + p)
    return out


# White fill on a black outline, the rest transparent: Y, Cr, Cb, alpha (limited range).
PALETTE = {0: (16, 128, 128, 0), 1: (235, 128, 128, 255), 2: (16, 128, 128, 255), 3: (126, 128, 128, 128)}


def show(pts: float, objects: list[tuple[int, int, int, np.ndarray]], forced: bool = False, palette=None,
         number: int = 0, state: int = 0x80, fragment: int = 0, crop=None) -> bytes:
    """A display set that shows objects (id, x, y, bitmap)."""
    comps = [{"id": oid, "x": x, "y": y, "forced": forced, "crop": crop} for oid, x, y, _ in objects]
    out = pcs(pts, comps, number, state)
    out += wds(pts, [(x, y, b.shape[1], b.shape[0]) for _, x, y, b in objects])
    out += pds(pts, palette or PALETTE)
    for oid, _, _, b in objects:
        out += ods(pts, oid, b, fragment=fragment)
    return out + segment(END, pts, b"")


def clear(pts: float, number: int = 0) -> bytes:
    """A display set with no object: it clears the screen."""
    return pcs(pts, [], number, state=0x00) + wds(pts, [(0, 0, 1, 1)]) + segment(END, pts, b"")


def again(pts: float, objects: list[tuple[int, int, int]], palette=None, palette_only: bool = False,
          number: int = 0) -> bytes:
    """A set that shows objects defined before again: an acquisition point, or a palette update."""
    comps = [{"id": oid, "x": x, "y": y} for oid, x, y in objects]
    out = pcs(pts, comps, number, state=0x00 if palette_only else 0x40, palette_only=palette_only)
    if palette:
        out += pds(pts, palette)
    return out + segment(END, pts, b"")


def outlined(mask: np.ndarray, border: int = 2) -> np.ndarray:
    """A mask of letters as a PGS bitmap: fill 1, a black outline 2 around it, the rest 0."""
    m = mask.astype(bool)
    grown = m.copy()
    for dy in range(-border, border + 1):
        for dx in range(-border, border + 1):
            grown |= np.roll(np.roll(m, dy, 0), dx, 1)
    out = np.zeros(m.shape, dtype=np.uint8)
    out[grown] = 2
    out[m] = 1
    return out


def text_mask(text: str, size: int = 40, italic: bool = False) -> np.ndarray | None:
    """Text drawn with Pillow's own font (None without Pillow): True where the letters are."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # an old Pillow: its bitmap font only
        return None
    lines = text.split("\n")
    w = max(int(font.getlength(line)) for line in lines) + 2 * size
    h = int(size * 1.5) * len(lines) + size
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        lw = font.getlength(line)
        d.text(((w - lw) / 2, size / 2 + k * size * 1.5), line, fill=255, font=font)
    a = np.array(im) > 127
    if italic:  # sheared 12 degrees to the right, as italics lean
        out = np.zeros_like(a)
        for y in range(h):
            s = int(round((h - 1 - y) * 0.21))
            out[y, s:] = a[y, :w - s] if s else a[y]
        a = out
    ys, xs = np.nonzero(a)
    return a[ys.min() - 4:ys.max() + 5, xs.min() - 4:xs.max() + 5]


# --- VobSub ----------------------------------------------------------------------

def rle2(idx: np.ndarray, top: bool) -> bytes:
    """VobSub's nibble run-length code of one field (the even lines, or the odd ones)."""
    nibbles: list[int] = []
    for y in range(0 if top else 1, idx.shape[0], 2):
        row = idx[y]
        x, w = 0, len(row)
        while x < w:
            c = int(row[x])
            n = 1
            while x + n < w and row[x + n] == c and n < 255:
                n += 1
            if x + n == w and n > 3:  # to the end of the line
                nibbles += [0, 0, 0, c]
            else:
                v = (n << 2) | c
                if n < 4:
                    nibbles += [v]
                elif n < 16:
                    nibbles += [v >> 4, v & 15]
                elif n < 64:
                    nibbles += [0, v >> 4, v & 15]
                else:
                    nibbles += [0, v >> 8, (v >> 4) & 15, v & 15]
            x += n
        if len(nibbles) % 2:
            nibbles.append(0)
    return bytes((a << 4) | b for a, b in zip(nibbles[::2], nibbles[1::2]))


def spu(idx: np.ndarray, x: int, y: int, start: float, stop: float, forced: bool = False,
        colours=(0, 1, 2, 3), contrast=(0, 15, 15, 15)) -> bytes:
    """A subpicture unit: its pixels (values 0-3), then a sequence that starts it (forced or not) and
    one that stops it, each after its delay."""
    h, w = idx.shape
    top, bottom = rle2(idx, True), rle2(idx, False)
    pix_top, pix_bottom = 4, 4 + len(top)
    ctrl = 4 + len(top) + len(bottom)
    area = bytes([x >> 4, ((x & 15) << 4) | ((x + w - 1) >> 8), (x + w - 1) & 0xFF,
                  y >> 4, ((y & 15) << 4) | ((y + h - 1) >> 8), (y + h - 1) & 0xFF])
    c, a = colours, contrast
    cmds = bytes([0x00 if forced else 0x01, 0x03, (c[3] << 4) | c[2], (c[1] << 4) | c[0],
                  0x04, (a[3] << 4) | a[2], (a[1] << 4) | a[0], 0x05]) + area \
        + bytes([0x06]) + struct.pack(">HH", pix_top, pix_bottom) + b"\xff"
    first = struct.pack(">HH", int(round(start * 90_000 / 1024)), 0) + cmds  # its next offset is set below
    second_at = ctrl + len(first)
    first = first[:2] + struct.pack(">H", second_at) + first[4:]
    second = struct.pack(">HH", int(round(stop * 90_000 / 1024)), second_at) + b"\x02\xff"
    body = top + bottom + first + second
    size = 4 + len(body)
    return struct.pack(">HH", size, ctrl) + body


def pes_packets(unit: bytes, stream: int = 0, chunk: int = 1000) -> bytes:
    """A unit in MPEG-2 program stream packs of private stream 1, `chunk` bytes of it each."""
    out = b""
    for k in range(0, len(unit), chunk):
        payload = bytes([0x20 + stream]) + unit[k:k + chunk]
        header = bytes([0x81, 0x80 if k == 0 else 0x00, 5 if k == 0 else 0]) + (b"\x21\x00\x01\x00\x01" if k == 0 else b"")
        pes = b"\x00\x00\x01\xbd" + struct.pack(">H", len(header) + len(payload)) + header + payload
        pack = b"\x00\x00\x01\xba" + bytes([0x44, 0, 4, 0, 4, 1, 0, 0, 3, 0xF8])
        out += pack + pes
    return out


def vobsub(units: list[tuple[float, bytes]], size=(720, 480)) -> tuple[str, bytes]:
    """An .idx and its .sub of subpicture units (timestamp, unit)."""
    sub = b""
    lines = ["# VobSub index file, v7 (do not modify this line!)", f"size: {size[0]}x{size[1]}",
             "palette: 000000, ffffff, 000000, 808080, " + ", ".join(["000000"] * 12), "id: en, index: 0"]
    for t, u in units:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        lines.append(f"timestamp: {int(h):02d}:{int(m):02d}:{int(s):02d}:{int(round((s % 1) * 1000)):03d}, "
                     f"filepos: {len(sub):09x}")
        sub += pes_packets(u)
    return "\n".join(lines) + "\n", sub

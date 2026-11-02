"""VobSub decoding: the .idx, subpicture units over several packets, both fields, times and colours."""

from __future__ import annotations

import numpy as np
import pytest
from synth import spu, vobsub

from subtitler import vobsub as vs


def bitmap(h: int = 31, w: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        x = 0
        while x < w:
            n = int(rng.choice([1, 2, 5, 20, 90, 300]))
            out[y, x:x + n] = int(rng.choice([0, 1, 2, 3]))
            x += n
    return out


def test_a_track_of_two_subpictures(tmp_path):
    a, b = bitmap(seed=1), bitmap(24, 410, 2)
    idx, sub = vobsub([(61.5, spu(a, 200, 400, 0.0, 2.5)), (65.25, spu(b, 150, 420, 0.1, 3.0, forced=True))])
    (tmp_path / "3.idx").write_text(idx, encoding="latin-1")
    (tmp_path / "3.sub").write_bytes(sub)
    pics = vs.read(tmp_path / "3.idx")
    assert len(pics) == 2 and pics[0].size == (720, 480)
    assert pics[0].start == pytest.approx(61.5) and pics[0].end == pytest.approx(61.5 + 2.5, abs=0.012)
    assert pics[1].start == pytest.approx(65.25 + 0.1, abs=0.012) and pics[1].forced and not pics[0].forced
    o = pics[0].objects[0]
    assert (o.x, o.y, o.luma.shape) == (200, 400, a.shape)
    # Palette: 0 black transparent, 1 white, 2 black, 3 grey; contrast 0, 15, 15, 15.
    assert (o.luma[a == 1] == 255).all() and (o.luma[a == 2] == 0).all() and (o.luma[a == 3] == 128).all()
    assert (o.alpha[a == 0] == 0).all() and (o.alpha[a != 0] == 255).all()
    assert np.array_equal(pics[1].objects[0].alpha > 0, b != 0)  # both fields, odd lines too


def test_the_idx_and_its_delay():
    idx = vs.read_idx("size: 720x576\npalette: ff0000, 00ff00\ndelay: 00:00:01:500\n"
                      "timestamp: 00:00:10:000, filepos: 000000800\n")
    assert idx["size"] == (720, 576) and idx["palette"][:2] == [(255, 0, 0), (0, 255, 0)] and len(idx["palette"]) == 16
    assert idx["entries"] == [(11.5, 0x800)]


def test_a_unit_without_a_stop_ends_at_the_next_one():
    a = bitmap(10, 50, 3)
    # A stop no later than its start is no stop: the next unit's start ends it.
    idx, sub = vobsub([(1.0, spu(a, 0, 0, 0.0, 0.0)), (4.0, spu(a, 0, 0, 0.0, 1.0))])
    pics = vs.pictures(idx, sub)
    assert pics[0].end == pytest.approx(4.0) and pics[1].end == pytest.approx(5.0, abs=0.012)


def test_a_subpicture_larger_than_its_video_is_left_out_and_an_idx_of_no_size_bounds_none():
    a = bitmap(10, 50, 3)
    units = [(1.0, spu(a, 0, 0, 0.0, 1.0)), (3.0, spu(a, 0, 0, 0.0, 1.0))]
    idx, sub = vobsub(units, size=(40, 480))  # wider than its video: no pixels decoded, a picture of nothing
    assert [(p.start, len(p.objects), p.skipped) for p in vs.pictures(idx, sub)] == [(1.0, 0, 1), (3.0, 0, 1)]
    idx, sub = vobsub(units, size=(50, 480))
    assert vs.pictures(idx, sub)[0].objects[0].width == 50
    # An .idx that names no size: no video to bound the area by (an HD track's areas are wider than a DVD's).
    big = bitmap(10, 1069, 4)
    idx, sub = vobsub([(1.0, spu(big, 0, 0, 0.0, 1.0))])
    idx = "\n".join(line for line in idx.splitlines() if not line.startswith("size:")) + "\n"
    pics = vs.pictures(idx, sub)
    assert vs.read_idx(idx)["size"] is None and pics[0].objects[0].width == 1069 and pics[0].size == (720, 576)

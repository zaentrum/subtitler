"""PGS decoding: run-length code, display sets, times, objects, palettes; synthetic streams built in the tests."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from synth import END, ODS, PALETTE, again, clear, ods, pcs, pds, rle, segment, show

from subtitler import pgs


def blocks(h: int = 30, w: int = 200, seed: int = 0) -> np.ndarray:
    """A bitmap with runs of every length: single pixels, short and long runs, colour 0 and others."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        x = 0
        while x < w:
            n = int(rng.choice([1, 2, 3, 7, 40, 70]))
            out[y, x:x + n] = int(rng.choice([0, 1, 2, 3]))
            x += n
    return out


def test_the_run_length_code_decodes_every_kind_of_run():
    for seed in range(5):
        b = blocks(seed=seed)
        assert np.array_equal(pgs.decode_rle(rle(b), b.shape[1], b.shape[0]), b)
    wide = np.zeros((2, 20000), dtype=np.uint8)
    wide[0, 100:17000] = 1  # a run longer than 16383 is coded as two
    assert np.array_equal(pgs.decode_rle(rle(wide), 20000, 2), wide)


def test_a_picture_lasts_from_its_set_to_the_set_that_clears_it():
    a, b = blocks(seed=1), blocks(seed=2)
    data = show(10.0, [(0, 100, 900, a)]) + clear(12.5) + show(13.0, [(0, 120, 900, b)]) + show(15.0, [(1, 50, 50, a)])
    data += clear(16.25)
    pics = pgs.pictures(data)
    assert [(p.start, p.end) for p in pics] == [(10.0, 12.5), (13.0, 15.0), (15.0, 16.25)]  # a new one ends the one before
    assert pics[0].size == (1920, 1080) and (pics[0].objects[0].x, pics[0].objects[0].y) == (100, 900)
    o = pics[0].objects[0]
    assert o.luma[a == 1].min() == 255 and o.luma[a == 2].max() == 0  # limited range to full range
    assert (o.alpha[a == 0] == 0).all() and (o.alpha[a == 3] == 128).all()


def test_objects_of_one_set_are_one_picture_and_a_fragmented_object_decodes_whole():
    top, bottom = blocks(20, 300, 3), blocks(20, 260, 4)
    big = blocks(120, 1500, 5)
    pics = pgs.pictures(show(1.0, [(0, 600, 860, top), (1, 620, 900, bottom)]) + clear(2.0)
                        + show(3.0, [(2, 10, 10, big)], fragment=4000) + clear(4.0))
    assert len(pics) == 2 and len(pics[0].objects) == 2
    assert [o.y for o in pics[0].objects] == [860, 900]
    lum = pics[1].objects[0].luma
    assert lum.shape == big.shape and (lum[big == 1] == 255).all()


def test_acquisition_points_and_palette_updates_are_the_same_picture():
    b = blocks(seed=6)
    fade = {k: (y, cr, cb, a // 2) for k, (y, cr, cb, a) in PALETTE.items()}
    gone = {k: (y, cr, cb, 0) for k, (y, cr, cb, _) in PALETTE.items()}
    data = (show(5.0, [(0, 0, 0, b)], palette=fade) + again(5.5, [(0, 0, 0)]) + again(6.0, [(0, 0, 0)], PALETTE, True)
            + again(7.0, [(0, 0, 0)], gone, True) + clear(7.5))
    pics = pgs.pictures(data)
    assert len(pics) == 1 and pics[0].sets == 3
    assert (pics[0].start, pics[0].end) == (5.0, 7.0)  # all transparent from 7.0 on
    assert pics[0].objects[0].alpha.max() == 255  # the picture as it is most opaque


def test_forced_flag_crop_and_epoch():
    b = blocks(40, 400, 7)
    pics = pgs.pictures(show(1.0, [(0, 0, 0, b)], forced=True, crop=(100, 10, 50, 20)) + clear(2.0))
    assert pics[0].forced and pics[0].objects[0].luma.shape == (20, 50)
    assert np.array_equal(pics[0].objects[0].alpha, np.array([[PALETTE[v][3] for v in r] for r in b[10:30, 100:150]]))
    # An epoch start forgets the objects before it: a set that names one shows nothing.
    later = show(1.0, [(0, 0, 0, b)]) + clear(2.0) + show(3.0, [(5, 0, 0, b)]) + again(4.0, [(0, 0, 0)])
    assert [p.start for p in pgs.pictures(later)] == [1.0, 3.0]


def test_not_a_pgs_stream():
    with pytest.raises(pgs.PgsError):
        pgs.pictures(b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n" * 3)


def test_count_reads_the_composition_segments_alone(tmp_path):
    b = blocks(seed=8)
    data = (show(1.0, [(0, 0, 0, b)]) + again(1.5, [(0, 0, 0)]) + clear(2.0) + show(3.0, [(0, 0, 0, b)]) + clear(4.0))
    f = tmp_path / "1.sup"
    f.write_bytes(data)
    assert pgs.count(f) == 2 == len(pgs.read(f))
    f.write_bytes(b"PG\x00\x00")
    with pytest.raises(pgs.PgsError):
        pgs.count(f)


def test_an_object_larger_than_its_video_is_left_out_before_it_is_decoded(monkeypatch):
    # A few bytes whose object claims 60000 x 60000 pixels (3.6 GB once decoded): left out unread, its
    # picture one of no objects, which the OCR step counts as read as nothing; the track goes on.
    claim = struct.pack(">HBB", 0, 0, 0xC0) + (7).to_bytes(3, "big") + struct.pack(">HH", 60000, 60000) + b"\x01\x00\x00"
    b = blocks(seed=3)
    data = (pcs(1.0, [{"id": 0, "x": 0, "y": 0}]) + pds(1.0, PALETTE) + segment(ODS, 1.0, claim)
            + segment(END, 1.0, b"") + again(1.5, [(0, 0, 0)]) + clear(2.0) + show(3.0, [(0, 10, 900, b)])
            + clear(4.0))
    real = pgs.decode_rle
    calls = []
    monkeypatch.setattr(pgs, "decode_rle", lambda d, w, h: calls.append((w, h)) or real(d, w, h))
    pics = pgs.pictures(data)
    assert [(p.start, p.end, len(p.objects), p.skipped) for p in pics] == [(1.0, 2.0, 0, 1), (3.0, 4.0, 1, 0)]
    assert calls == [(b.shape[1], b.shape[0])] and not pics[0].forced and pics[0].sets == 2
    # An object as wide as its video is one (a line across the screen); on a smaller video it is not.
    line = np.zeros((40, 1920), dtype=np.uint8)
    line[10:30, 100:1800] = 1
    assert pgs.pictures(show(1.0, [(0, 0, 900, line)]) + clear(2.0))[0].objects[0].width == 1920
    small = (pcs(1.0, [{"id": 0, "x": 0, "y": 0}], size=(720, 480)) + pds(1.0, PALETTE) + ods(1.0, 0, line)
             + segment(END, 1.0, b"") + clear(2.0))
    assert [(len(p.objects), p.skipped) for p in pgs.pictures(small)] == [(0, 1)]
    assert pgs.too_large(5000, 10, (0, 0)) and not pgs.too_large(4096, 4096, (0, 0))  # no size: 4096 a side

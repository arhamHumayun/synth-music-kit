"""Smoke test: the starter example renders, is the right length, and loops without a click."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from synthkit.musiclib import SR, master  # noqa: E402
from synthkit.theory import A_MINOR, diminute, off_scale  # noqa: E402

import hello_loop  # noqa: E402
import menu_short  # noqa: E402
import stage1_short  # noqa: E402


@pytest.mark.parametrize("track", [hello_loop, stage1_short, menu_short])
def test_example_renders_and_loops(track):
    mix, n, _song = track.build()
    assert n == track.BARS * 16 * (SR * 60 // (track.BPM * 4))   # a whole number of samples
    y = master(mix, n, -15.5)
    assert y.shape == (2, n) and np.isfinite(y).all() and np.abs(y).max() <= 1.0
    jump = np.abs(y[:, 0] - y[:, -1]).max()                                  # the last sample flows into the first
    assert jump < 3 * np.percentile(np.abs(np.diff(y, axis=1)), 99.9)


def test_theory_helpers():
    assert not off_scale(hello_loop.TUNE, A_MINOR)
    assert diminute([(0, 0, 60, 4)]) == [(0, 0.0, 60, 2.0)]


def test_menu_and_arcade_share_one_motif():
    assert stage1_short.MOTIF is menu_short.MOTIF
    assert stage1_short.BPM == 2 * menu_short.BPM

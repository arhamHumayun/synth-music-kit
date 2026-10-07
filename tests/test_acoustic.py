"""Measurements that pin down what the acoustic instruments are meant to do (we cannot hear them, so we check the physics)."""

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from synthkit.acoustic import acoustic_guitar, acoustic_kit, air_flute, cello, felt_piano, upright_bass  # noqa: E402
from synthkit.musiclib import SR  # noqa: E402

import porch_light  # noqa: E402


def _peak_hz(x, lo, hi, start=0.15, length=1.0):
    seg = x[int(start * SR):int((start + length) * SR)]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    freqs = np.fft.rfftfreq(8 * len(seg), 1 / SR)
    band = (freqs > lo) & (freqs < hi)
    return freqs[band][np.argmax(spectrum[band])]


def _rms_db(x, a, b):
    seg = x[int(a * SR):int(b * SR)]
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)


def _centroid(x):
    spectrum = np.abs(np.fft.rfft(x))
    return (spectrum * np.fft.rfftfreq(len(x), 1 / SR)).sum() / spectrum.sum()


def test_every_instrument_plays_the_note_it_is_asked_for():
    for fn, f in ((acoustic_guitar, 196.0), (upright_bass, 82.4), (felt_piano, 261.6), (air_flute, 440.0), (cello, 220.0)):
        x = fn(f, 1.5)
        assert np.isfinite(x).all() and np.abs(x).max() <= 1.0 + 1e-9
        cents = 1200 * np.log2(_peak_hz(x, f * 0.8, f * 1.25) / f)
        assert abs(cents) < 12, (fn.__name__, cents)


def test_plucked_and_struck_notes_decay_and_bowed_notes_sustain():
    for fn in (acoustic_guitar, upright_bass):
        x = fn(110.0, 2.0)
        assert _rms_db(x, 0.15, 0.3) - _rms_db(x, 0.9, 1.05) > 8
    piano = felt_piano(196.0, 3.0)
    assert _rms_db(piano, 0.15, 0.3) - _rms_db(piano, 1.8, 1.95) > 3
    bowed = cello(220.0, 2.0)
    assert abs(_rms_db(bowed, 0.5, 0.7) - _rms_db(bowed, 1.2, 1.4)) < 3


def test_steel_is_brighter_than_nylon():
    assert _centroid(acoustic_guitar(196.0, 1.5, steel=True)) > 1.3 * _centroid(acoustic_guitar(196.0, 1.5))


def test_piano_partials_are_stretched_sharp_of_whole_multiples():
    x = felt_piano(65.4, 2.0)
    seg = x[int(0.1 * SR):int(1.5 * SR)]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    freqs = np.fft.rfftfreq(8 * len(seg), 1 / SR)
    ratios = []
    for p in (6, 10, 12):                                # not 8: the hammer strikes 1/8 along, which cancels it
        band = (freqs > 65.4 * p * 0.97) & (freqs < 65.4 * p * 1.06)
        ratios.append(freqs[band][np.argmax(spectrum[band])] / (65.4 * p))
    assert all(r > 1.0 for r in ratios) and ratios[-1] > ratios[0]


def test_drums_live_in_the_right_part_of_the_spectrum():
    kit = acoustic_kit(np.random.default_rng(1))

    def share(x, lo, hi):
        power = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        return power[(freqs >= lo) & (freqs < hi)].sum() / power.sum()

    assert share(kit["kick"], 0, 150) > 0.9
    assert share(kit["hat0"], 5000, SR / 2) > 0.9
    assert share(kit["snare0"], 1500, SR / 2) > 0.5


def test_porch_light_renders_and_loops():
    mix, n, _song = porch_light.build()
    assert n == porch_light.BARS * 16 * (SR * 60 // (porch_light.BPM * 4))
    assert np.isfinite(mix).all()

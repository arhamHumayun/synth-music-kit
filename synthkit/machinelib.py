"""Cold, mechanical instruments: a klaxon, a tuned gong, bowed-metal drones, a tin clank and a bit-crusher.

Everything is synthesized; no samples. Functions take (frequency Hz, duration s) and return a mono array.
"""

from __future__ import annotations

import numpy as np

from synthkit.heavylib import drive
from synthkit.musiclib import SR, adsr, bp, harm_count, hp, lp, peak_norm, taxis

# Inharmonic partial ratios (amplitude, decay seconds) of a struck gong, loosely after a tam-tam.
GONG_PARTIALS = ((1.0, 1.0, 4.2), (1.47, 0.8, 3.4), (2.09, 0.7, 3.0), (2.56, 0.55, 2.4), (3.15, 0.45, 2.0),
                 (4.2, 0.3, 1.5), (5.4, 0.2, 1.1))
# Bowed metal: a plate rubbed with a bow, a slow swell of near-harmonic partials that beat against each other.
DRONE_PARTIALS = ((1.0, 1.0), (2.003, 0.55), (2.76, 0.38), (3.91, 0.26), (5.43, 0.16))
DRONE_BEAT_HZ = 0.35


def klaxon(f: float, dur: float, rel: float = 0.05, scoop: float = 0.035) -> np.ndarray:
    """An alarm voice: two detuned band-limited square waves with a quick pitch scoop, pushed into distortion."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    glide = 1.0 + scoop * np.exp(-t / 0.025)
    out = np.zeros(n)
    for cents in (-14.0, 14.0):
        ff = f * 2 ** (cents / 1200)
        ph = 2 * np.pi * np.cumsum(ff * glide) / SR
        for h in range(1, harm_count(ff, 24) + 1, 2):
            out += np.sin(h * ph) / h
    out = lp(out, 4200, 2)
    out = drive(out / (np.max(np.abs(out)) + 1e-9), 2.6)
    return hp(out, 90, 2) * adsr(n, dur, 0.004, rel)


def gong(f: float, ring: float = 5.0) -> np.ndarray:
    """A struck gong tuned to `f`: inharmonic partials, a slow bloom as the upper ones swell in, and a noise crash."""
    n = int(ring * SR)
    t = taxis(n)
    rng = np.random.default_rng(int(f * 10))
    x = np.zeros(n)
    for i, (ratio, amp, decay) in enumerate(GONG_PARTIALS):
        bloom = 1.0 - np.exp(-t / (0.02 + 0.08 * i))
        x += amp * np.sin(2 * np.pi * f * ratio * t + rng.random() * 6.28) * np.exp(-t / decay) * bloom
    crash = bp(rng.standard_normal(n), 900, 7000, 2) * np.exp(-t / 0.9) * np.minimum(1.0, t / 0.01)
    return peak_norm(x + 0.18 * crash) * np.minimum(1.0, t / 0.001)


def bowed_metal(f: float, dur: float, a: float = 1.2, rel: float = 1.5) -> np.ndarray:
    """A slow, cold swell of near-harmonic partials beating against each other, with a little bow noise."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    rng = np.random.default_rng(int(f * 7))
    x = np.zeros(n)
    for i, (ratio, amp) in enumerate(DRONE_PARTIALS):
        beat = 1.0 + 0.5 * np.sin(2 * np.pi * (DRONE_BEAT_HZ + 0.11 * i) * t + rng.random() * 6.28)
        x += amp * beat * np.sin(2 * np.pi * f * ratio * t + rng.random() * 6.28)
    bow = lp(rng.standard_normal(n), 2200, 2) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t))
    return peak_norm(x + 0.05 * bow) * adsr(n, dur, a, rel)


def bitcrush(x: np.ndarray, bits: int = 6, hold: int = 5) -> np.ndarray:
    """Quantise to `bits` and hold every `hold`th sample: a broken-speaker grit."""
    held = np.repeat(x[::hold], hold)[: len(x)]
    q = 2.0 ** (bits - 1)
    return np.round(held * q) / q


def clank(f: float, rng, ring: float = 0.5) -> np.ndarray:
    """A short pitched tin clank on `f`, handy for tracing a rhythm cell."""
    n = int(ring * SR)
    t = taxis(n)
    ratios = (1.0, 2.76, 5.4, 8.9)
    x = sum(a * np.sin(2 * np.pi * f * r * t + rng.random() * 6.28) * np.exp(-t / d)
            for r, a, d in zip(ratios, (1.0, 0.7, 0.45, 0.3), (0.16, 0.1, 0.06, 0.035)))
    x = x + 0.4 * bp(rng.standard_normal(n), 2500, 9000, 2) * np.exp(-t / 0.008)
    return peak_norm(x * np.minimum(1.0, t / 0.0006))

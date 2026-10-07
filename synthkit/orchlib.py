"""Extra voices for the story tracks: piano, organ, flute, harp, pizzicato, tremolo strings, timpani, glockenspiel.

Everything is synthesized (additive / subtractive / modal); see heavylib.py for strings, brass, choir and
guitars. Functions take (frequency Hz, duration s) and return a mono array unless noted.
"""

from __future__ import annotations

import numpy as np

from synthkit.heavylib import string_ensemble, subtractive_saw
from synthkit.musiclib import SR, adsr, bell_note, bp, harm_count, lp, peak_norm, taxis


def piano_note(f: float, dur: float, vel: float = 1.0) -> np.ndarray:
    """Hammered-string piano: slightly stretched partials that die faster the higher they are, a felt thump,
    and a damper when the key is released."""
    rel = 0.22
    n = int((dur + rel) * SR)
    t = taxis(n)
    b = 0.00022 * (f / 261.6) ** 2  # string stiffness: partials run a little sharp
    h = np.arange(1, harm_count(f, 18) + 1)[:, None]
    fh = f * h * np.sqrt(1 + b * h**2)
    base = float(np.clip(2.8 * (261.6 / f) ** 0.4, 0.6, 4.0))
    decay = base / (1 + 0.45 * (h - 1))
    amp = h ** -0.95 * np.exp(-t[None, :] / decay)
    x = (amp * np.sin(2 * np.pi * fh * t[None, :])).sum(0)
    x *= np.minimum(1.0, t / 0.0015)
    x *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.07))
    rng = np.random.default_rng(int(f * 7))
    thump = bp(rng.standard_normal(n), 120, 900, 1) * np.exp(-t / 0.012) * 0.25
    return peak_norm(x + thump) * 0.55 * vel


def organ_note(f: float, dur: float) -> np.ndarray:
    """Drawbar organ with a slow rotary-speaker shimmer and a short key click."""
    rel = 0.07
    n = int((dur + rel) * SR)
    t = taxis(n)
    x = np.zeros(n)
    for ratio, amp in ((1, 1.0), (2, 0.75), (3, 0.55), (4, 0.42), (6, 0.28), (8, 0.18)):
        if f * ratio < 0.45 * SR:
            x += amp * np.sin(2 * np.pi * f * ratio * t + 0.3 * np.sin(2 * np.pi * 6.1 * t))
    x *= 0.82 + 0.18 * np.sin(2 * np.pi * 6.1 * t)
    click = 0.35 * np.sin(2 * np.pi * f * 5 * t) * np.exp(-t / 0.012)
    return (x + click) * adsr(n, dur, 0.010, rel) * 0.16


def flute_note(f: float, dur: float, rel: float = 0.14) -> np.ndarray:
    """Airy flute: near-sine with a breath-noise onset and gentle late vibrato."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    ramp = np.clip((t - 0.18) / 0.35, 0, 1)
    ph = 2 * np.pi * f * t + (f * 0.0035 / 5.2) * ramp * np.sin(2 * np.pi * 5.2 * t)
    x = np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.07 * np.sin(3 * ph)
    rng = np.random.default_rng(int(f * 13))
    breath = bp(rng.standard_normal(n), max(500.0, f * 1.2), min(9000.0, f * 7.0), 1) * (0.07 + 0.12 * np.exp(-t / 0.08))
    return (x * 0.45 + breath) * adsr(n, dur, 0.05, rel)


def harp_note(f: float, ring: float = 2.0) -> np.ndarray:
    """Plucked harp string: bright pluck that rounds off quickly, long soft tail."""
    n = int(ring * SR)
    t = taxis(n)
    h = np.arange(1, harm_count(f, 8) + 1)[:, None]
    amp = h ** -1.2 * np.exp(-t[None, :] / (1.4 / h**0.55))
    x = (amp * np.sin(2 * np.pi * f * h * t[None, :])).sum(0)
    rng = np.random.default_rng(int(f * 3))
    x += 0.25 * bp(rng.standard_normal(n), 1500, 6000, 1) * np.exp(-t / 0.006)
    return peak_norm(x * np.minimum(1.0, t / 0.002)) * 0.5 * np.clip((ring - t) / 0.2, 0, 1)


def pizz_note(f: float, dur: float = 0.2) -> np.ndarray:
    """Pizzicato strings: a plucked saw ensemble that snaps shut."""
    return subtractive_saw(f, dur, 0.04, 2800.0, 450.0, 0.03, q=0.9, a=0.002, env_tau=0.11,
                           detunes=(-9.0, 9.0), hcap=30) * 0.5


def tremolo_strings(f: float, dur: float, bright: float = 3400.0) -> np.ndarray:
    """Sustained string ensemble bowed with a fast tremolo (the 'something is coming' texture). Stereo."""
    x = string_ensemble(f, dur, stacc=False, bright=bright)
    t = taxis(x.shape[1])
    return x * (0.5 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 13.0 * t)) ** 1.5)


def timpani_note(f: float) -> np.ndarray:
    """Tuned orchestral drum: pitch-bent body, felt thud, long boom."""
    n = int(1.6 * SR)
    t = taxis(n)
    ph = 2 * np.pi * np.cumsum(f * (1 + 0.22 * np.exp(-t / 0.05))) / SR
    body = np.sin(ph) * np.exp(-t / 0.55) + 0.35 * np.sin(2 * ph * 1.5) * np.exp(-t / 0.25)
    rng = np.random.default_rng(int(f))
    felt = lp(rng.standard_normal(n), 900, 2) * np.exp(-t / 0.02) * 0.5
    return peak_norm(np.tanh(1.3 * (body + felt)))


def glock_note(f: float) -> np.ndarray:
    """Glockenspiel / celesta sparkle."""
    return bell_note(f, ring=1.8, bright=1.3) * 0.9

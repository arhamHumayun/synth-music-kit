"""Acoustic instruments built from physical models: guitar, upright bass, piano, cello, flute and a small drum kit.

Where the other modules shape oscillators, these model how the real instrument makes its sound:

  * plucked strings (guitar, bass): a Karplus-Strong string, a delay line with a lossy loop, excited by a noise burst
    with the pick position cut out of it, then coloured by the instrument body's resonances
  * piano: modal synthesis. Stiff strings make partials sit slightly sharp of whole multiples (inharmonicity), three
    slightly detuned strings per note beat against each other, upper partials die faster, a felt hammer rounds the top
  * cello: a band-limited saw (bow-slip-stick) with a slow bow attack, delayed vibrato, rosin noise and body resonances
  * flute: few harmonics, breath noise, a pitch overshoot and a chiff at the attack
  * drums: pitched heads with a pitch drop, snare wires, stick and beater clicks, metallic cymbal partials

Voices take (frequency Hz, duration s) and return a mono array. `dur` is how long the note is held before it is damped:
a plucked or struck note keeps ringing at its natural rate until then. Everything is deterministic (noise is seeded from
the pitch), and every function says what it models so you can tell what to tune. Nothing here uses samples.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from synthkit.heavylib import subtractive_saw
from synthkit.musiclib import SR, adsr, bp, hp, lp, peak_norm, taxis

# Body resonances as (low Hz, high Hz, gain): the wooden box adds energy near its main modes.
GUITAR_BODY = ((85.0, 125.0, 0.55), (180.0, 250.0, 0.40), (370.0, 480.0, 0.22), (900.0, 1200.0, 0.10))
BASS_BODY = ((55.0, 85.0, 0.7), (120.0, 170.0, 0.35), (260.0, 340.0, 0.15))
CELLO_BODY = ((200.0, 300.0, 0.5), (380.0, 520.0, 0.4), (950.0, 1250.0, 0.25), (2200.0, 2900.0, 0.2))
PIANO_BODY = ((180.0, 330.0, 0.22), (550.0, 800.0, 0.08))
# The inharmonic partials of a cymbal or hat, in Hz.
CYMBAL_PARTIALS = (6200.0, 7900.0, 9400.0, 11300.0, 12900.0)


def _damp(n: int, dur: float, rel: float = 0.08) -> np.ndarray:
    """1 until `dur`, then a quick exponential fade: a finger or damper touching the string."""
    t = taxis(n)
    return np.where(t < dur, 1.0, np.exp(-(t - dur) / rel))


def _body(x: np.ndarray, modes: tuple, dry: float = 1.0) -> np.ndarray:
    """Add the instrument body's resonances: each mode is a band of the signal boosted back in."""
    out = dry * x
    for lo, hi, gain in modes:
        out = out + gain * bp(x, lo, hi, 2)
    return out


def _string(f: float, ring: float, t60: float, exc_hz: float, loop_lp: float, pick: float, seed: int) -> np.ndarray:
    """A Karplus-Strong string, run as one IIR filter. `t60` is the DC decay time, `exc_hz` how bright the pluck is,
    `loop_lp` how fast highs die (0 = never, 0.25 = very dull), `pick` where along the string it is plucked."""
    length = SR / f
    n_int = max(1, int(length - 1))
    frac = length - 1 - n_int                                   # the loop filter below adds ~1 sample of delay
    lowpass = np.array([loop_lp, 1 - 2 * loop_lp, loop_lp])
    h = np.convolve(lowpass, [1 - frac, frac])                  # linear interpolation tunes the fractional delay
    gain = 10 ** (-3.0 / (t60 * f))                             # per-period loss that gives the requested T60
    denom = np.zeros(n_int + len(h))
    denom[0] = 1.0
    denom[n_int:n_int + len(h)] -= gain * h
    rng = np.random.default_rng(seed * 7919 + int(f * 10))
    k = int(length)
    burst = lp(rng.standard_normal(max(k, 64)), exc_hz, 2)[:k]
    shift = max(1, int(pick * length))
    burst[shift:] = burst[shift:] - burst[:-shift]               # a pluck at `pick` cancels the harmonics with a node there
    burst = burst - burst.mean()
    x = np.zeros(int(ring * SR))
    x[:k] = burst / (np.max(np.abs(burst)) + 1e-9)
    return lfilter([1.0], denom, x)


def acoustic_guitar(f: float, dur: float, steel: bool = False, seed: int = 0) -> np.ndarray:
    """A fingerpicked guitar string: nylon (warm, dull) or steel (bright), with the body's boomy low modes and a thump."""
    t60 = float(np.clip(3.6 * (196.0 / f) ** 0.45, 0.9, 5.0))
    ring = min(dur + 0.4, t60 * 1.1) if dur > 0 else t60
    y = _string(f, max(ring, 0.3), t60, exc_hz=5200.0 if steel else 2300.0, loop_lp=0.07 if steel else 0.16,
                pick=0.14 if steel else 0.18, seed=seed)
    n = len(y)
    y = _body(y, GUITAR_BODY)
    t = taxis(n)
    rng = np.random.default_rng(seed + int(f))
    thump = lp(rng.standard_normal(n), 700, 2) * np.exp(-t / 0.012) * 0.25 * np.max(np.abs(y))
    return peak_norm((y + thump) * _damp(n, dur, 0.09))


def upright_bass(f: float, dur: float, seed: int = 0) -> np.ndarray:
    """A plucked double bass: a heavy, dull string with a woody body boom and a finger thump."""
    t60 = float(np.clip(1.5 * (60.0 / f) ** 0.3, 0.6, 2.0))
    y = _string(f, min(dur + 0.4, t60 * 1.1), t60, exc_hz=520.0, loop_lp=0.25, pick=0.22, seed=seed)
    n = len(y)
    y = lp(_body(y, BASS_BODY), 900, 2)
    t = taxis(n)
    rng = np.random.default_rng(seed + int(f))
    thump = lp(rng.standard_normal(n), 400, 2) * np.exp(-t / 0.015) * 0.3 * np.max(np.abs(y))
    return peak_norm((y + thump) * _damp(n, dur, 0.07))


def felt_piano(f: float, dur: float, softness: float = 7.0) -> np.ndarray:
    """A piano: stiff strings (partials slightly sharp), three detuned strings per note beating, a felt hammer, damper."""
    ring = min(dur + 0.6, 9.0)
    n = int(ring * SR)
    t = taxis(n)
    rng = np.random.default_rng(int(f * 13))
    stiffness = 0.00006 + 0.0003 * (f / 261.6) ** 1.6            # inharmonicity coefficient B, larger for short thick treble
    tau1 = float(np.clip(7.5 * (130.0 / f) ** 0.6, 1.2, 10.0))   # the fundamental's decay time, longer for bass notes
    detune_cents = (-2.2, 0.0, 1.8)
    x = np.zeros(n)
    max_partial = int(min(36, 0.45 * SR / f))
    for p in range(1, max_partial + 1):
        fp = f * p * np.sqrt(1 + stiffness * p * p)               # sharp of p * f: the signature piano stretch
        if fp > 0.45 * SR:
            break
        amp = abs(np.sin(np.pi * p * 0.125)) / p / (1 + (p / softness) ** 2)   # hammer strikes 1/8 along; felt rounds highs
        tau = tau1 / (1 + 0.55 * (p - 1) ** 1.05)
        env = np.exp(-t / tau)
        phase = rng.random() * 6.28                               # the hammer starts all three strings in phase
        for c in detune_cents:
            x += amp * env * np.sin(2 * np.pi * fp * 2 ** (c / 1200) * t + phase)
    attack = np.minimum(1.0, t / 0.0012)
    thump = lp(rng.standard_normal(n), 600, 2) * np.exp(-t / 0.007) * 0.25 * np.max(np.abs(x))
    x = _body(x * attack + thump, PIANO_BODY)
    return peak_norm(x * _damp(n, dur, 0.22))


def cello(f: float, dur: float, vib_depth: float = 0.0055) -> np.ndarray:
    """A bowed cello: a saw through a bow-pressure filter, slow bow attack, delayed vibrato, rosin noise, body resonances."""
    rel = 0.14
    x = subtractive_saw(f, dur, rel, fc0=1500.0, fc1=2400.0, tau=0.25, q=0.9, a=0.11, detunes=(0.0,), hcap=40,
                        tilt=1.0, vib_hz=5.2, vib=vib_depth)
    n = len(x)
    t = taxis(n)
    rng = np.random.default_rng(int(f * 5))
    rosin = bp(rng.standard_normal(n), 1500, 6500, 2) * (0.012 + 0.13 * np.exp(-t / 0.07))     # a scrape on the attack, a hiss after
    x = x / (np.max(np.abs(x)) + 1e-9)
    y = _body(x, CELLO_BODY) + rosin
    return peak_norm(y)


def air_flute(f: float, dur: float) -> np.ndarray:
    """A concert flute: a few soft harmonics, breath noise, a small pitch overshoot and a chiff at the attack, late vibrato."""
    rel = 0.1
    n = int((dur + rel) * SR)
    t = taxis(n)
    rng = np.random.default_rng(int(f * 3))
    vib = 1.0 + 0.0045 * np.clip((t - 0.25) / 0.3, 0, 1) * np.sin(2 * np.pi * 5.0 * t)
    ph = 2 * np.pi * np.cumsum(f * vib * (1 + 0.012 * np.exp(-t / 0.05))) / SR
    tone = sum(a * np.sin(h * ph) for h, a in zip(range(1, 6), (1.0, 0.45, 0.2, 0.08, 0.04)))
    breath = bp(rng.standard_normal(n), max(300.0, f * 1.5), min(9000.0, f * 5.0), 2)
    chiff = bp(rng.standard_normal(n), 2000, 6500, 2) * np.exp(-t / 0.035)
    env = adsr(n, dur, 0.07, rel)
    y = tone * env + 0.07 * breath * env + 0.2 * chiff
    return peak_norm(lp(y, 7000, 2))


# ---------------------------------------------------------------------------- drums


def _noise(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n)


def acoustic_kit(rng) -> dict[str, np.ndarray]:
    """A small, soft kit: pillow kick, snare with wires (three takes), cross-stick, closed and open hats, shaker, brush."""
    k: dict[str, np.ndarray] = {}
    # Kick: a head tuned to ~55 Hz with a quick pitch drop, a shell knock and a soft beater tick.
    n = int(0.6 * SR)
    t = taxis(n)
    phase = 2 * np.pi * np.cumsum(52 + 70 * np.exp(-t / 0.028)) / SR
    body = np.sin(phase) * np.exp(-t / 0.17)
    knock = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.05) * 0.3
    tick = bp(_noise(n, rng), 1800, 5000, 2) * np.exp(-t / 0.003) * 0.35
    k["kick"] = peak_norm(lp(body + knock + tick, 3500, 2) * np.minimum(1.0, t / 0.0008))
    # Snare: two shell tones with a small pitch drop, the wires' rattle, a stick click (three takes).
    n = int(0.5 * SR)
    t = taxis(n)
    for i in range(3):
        drop = 1 + 0.15 * np.exp(-t / 0.012)
        shell = np.sin(2 * np.pi * np.cumsum(185 * drop) / SR) * np.exp(-t / 0.07) * 0.6
        shell += np.sin(2 * np.pi * np.cumsum(330 * drop) / SR + i) * np.exp(-t / 0.045) * 0.35
        wires = bp(_noise(n, rng), 1500, 9500, 2) * np.exp(-t / (0.14 + 0.02 * i))
        click = bp(_noise(n, rng), 3000, 8000, 2) * np.exp(-t / 0.002) * 0.5
        k[f"snare{i}"] = peak_norm((shell + wires + click) * np.minimum(1.0, t / 0.0006))
    # Cross-stick: stick on the rim, a woody tick with a short ring (two takes).
    n = int(0.25 * SR)
    t = taxis(n)
    for i in range(2):
        tick = np.sin(2 * np.pi * (1700 + 90 * i) * t) * np.exp(-t / 0.006) + 0.6 * np.sin(2 * np.pi * (2450 + 60 * i) * t) * np.exp(-t / 0.004)
        wood = np.sin(2 * np.pi * 550 * t) * np.exp(-t / 0.02) * 0.7
        air = bp(_noise(n, rng), 2000, 6000, 2) * np.exp(-t / 0.004) * 0.5
        k[f"rim{i}"] = peak_norm((tick + wood + air) * np.minimum(1.0, t / 0.0004))
    # Hats: noise plus cymbal partials, high-passed; closed is short, open rings and fades.
    for i in range(3):
        n = int(0.12 * SR)
        t = taxis(n)
        metal = sum(np.sin(2 * np.pi * p * (1 + 0.01 * i) * t + rng.random() * 6.28) for p in CYMBAL_PARTIALS)
        k[f"hat{i}"] = peak_norm(hp(bp(_noise(n, rng), 6000, 15000, 2) + 0.3 * metal, 6500, 2) * np.exp(-t / 0.028))
    n = int(0.7 * SR)
    t = taxis(n)
    metal = sum(np.sin(2 * np.pi * p * t + rng.random() * 6.28) for p in CYMBAL_PARTIALS)
    k["open_hat"] = peak_norm(hp(bp(_noise(n, rng), 5500, 15000, 2) + 0.3 * metal, 6000, 2) * np.exp(-t / 0.2))
    # Shaker: two bursts of high noise, a soft "chk-a" (two takes).
    for i in range(2):
        n = int(0.14 * SR)
        t = taxis(n)
        env = np.minimum(1.0, t / 0.012) * np.exp(-t / (0.035 + 0.01 * i))
        k[f"shaker{i}"] = peak_norm(bp(_noise(n, rng), 5000, 12000, 2) * env)
    # Brush: a noise swell, the sweep of a wire brush on the snare head.
    n = int(0.4 * SR)
    t = taxis(n)
    env = np.minimum(1.0, t / 0.12) * np.exp(-t / 0.13)
    k["brush"] = peak_norm(bp(_noise(n, rng), 3000, 9500, 2) * env)
    return k

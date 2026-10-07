"""Shared synthesis toolkit for procedural music (see render.py).

Theory helpers, oscillators/filters/envelopes, instruments, drum kit, reverb/delay/sidechain, the
`Song` scaffold (sample-accurate bar/step grid, stems, loop-safe tail folding), the mix bus and
loudness mastering. Everything is deterministic (seeded) so renders are reproducible.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100
TAIL = 8 * SR  # overhang rendered past the loop point, then folded onto the start
PEAK_CEILING_DB = -1.5

# ---------------------------------------------------------------------------- theory

_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def midi(name: str) -> int:
    return 12 * (int(name[-1]) + 1) + _PC[name[:-1]]


def hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


CHORDS = {  # name -> (quality, root pitch class)
    "Dm": ("min", 2),
    "Bb": ("maj", 10),
    "F": ("maj", 5),
    "C": ("maj", 0),
    "Gm": ("min", 7),
    "A": ("maj", 9),  # dominant: leads back to Dm
}
TRIAD = {"min": (0, 3, 7), "maj": (0, 4, 7)}
D_MINOR = [2, 4, 5, 7, 9, 10, 0]
SCALE_NOTES = [m for m in range(24, 120) if m % 12 in D_MINOR]


def nearest_octave(pc: int, target: int) -> int:
    return pc + 12 * round((target - pc) / 12)


def chord_pcs(name: str) -> list[int]:
    q, root = CHORDS[name]
    return [(root + i) % 12 for i in TRIAD[q]]


def chord_tones(name: str, target: int) -> list[int]:
    q, root = CHORDS[name]
    r = nearest_octave(root, target)
    return [r + i for i in TRIAD[q]]


def arp_tones(name: str) -> list[int]:
    a, b, c = chord_tones(name, 62)
    return [a, b, c, a + 12, b + 12]


def bass_root(name: str) -> int:
    return nearest_octave(CHORDS[name][1], 38)


def voice_lead(prev: list[int] | None, name: str, lo: int = 48, hi: int = 65) -> list[int]:
    """Close-position triad of `name` that moves the least from `prev` (smooth pad changes)."""
    pcs = chord_pcs(name)
    opts = [[m for m in range(lo, hi + 1) if m % 12 == pc] for pc in pcs]
    best, best_cost = None, 1e9
    for a in opts[0]:
        for b in opts[1]:
            for c in opts[2]:
                v = sorted((a, b, c))
                if v[2] - v[0] > 12:
                    continue
                cost = sum(abs(x - y) for x, y in zip(v, sorted(prev))) if prev else abs(sum(v) / 3 - 56)
                if cost < best_cost:
                    best, best_cost = v, cost
    return best


def diatonic_shift(m: int, steps: int, scale_notes: list[int] | None = None) -> int:
    """Move `m` by `steps` scale degrees (default scale: D natural minor); off-scale notes go up a minor third."""
    notes = scale_notes or SCALE_NOTES
    if m not in notes:
        return m + 3
    return notes[notes.index(m) + steps]


# ---------------------------------------------------------------------------- dsp helpers


def taxis(n: int) -> np.ndarray:
    return np.arange(n) / SR


def _sos(kind: str, fc, order: int = 4):
    return signal.butter(order, fc, btype=kind, fs=SR, output="sos")


def lp(x, fc, order=4):
    return signal.sosfilt(_sos("low", fc, order), x, axis=-1)


def hp(x, fc, order=4):
    return signal.sosfilt(_sos("high", fc, order), x, axis=-1)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos("band", [lo, hi], order), x, axis=-1)


def adsr(n: int, dur: float, a: float, r: float, tau: float | None = None) -> np.ndarray:
    t = taxis(n)
    e = np.minimum(1.0, t / max(a, 1e-4))
    if tau:
        e = e * np.exp(-t / tau)
    rel = np.clip(1 - (t - dur) / max(r, 1e-4), 0, 1) ** 2
    return e * np.where(t < dur, 1.0, rel)


def peak_norm(x: np.ndarray) -> np.ndarray:
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def harm_count(f: float, cap: int) -> int:
    return int(max(1, min(cap, (0.45 * SR) // f)))


def additive_saw(f: float, t: np.ndarray, cap: int, decay_k: float = 0.0, tilt: float = 1.0) -> np.ndarray:
    """Band-limited saw; higher harmonics fade faster when decay_k > 0 (a cheap filter envelope)."""
    h = np.arange(1, harm_count(f, cap) + 1)[:, None]
    amp = h ** -tilt * np.exp(-decay_k * (h - 1) * t[None, :])
    return (amp * np.sin(2 * np.pi * f * h * t[None, :])).sum(0)


def pan_gains(p: float) -> tuple[float, float]:
    a = (p + 1) * np.pi / 4
    return float(np.cos(a)), float(np.sin(a))


def curve(length: int, keys: list[tuple[float, float]], sps: int) -> np.ndarray:
    """Piecewise-linear automation over time. keys = [(bar, value), ...]."""
    xs = [k[0] * 16 * sps for k in keys]
    ys = [k[1] for k in keys]
    return np.interp(np.arange(length), xs, ys)


# ---------------------------------------------------------------------------- instruments


def pluck(f, dur, k=6.0, tau=0.18, cap=28):
    n = int((dur + 0.06) * SR)
    t = taxis(n)
    return additive_saw(f, t, cap, decay_k=k) * adsr(n, dur, 0.002, 0.05, tau=tau) * 0.5


def bass_note(f, dur, k=3.5):
    n = int((dur + 0.04) * SR)
    t = taxis(n)
    saw = additive_saw(f, t, 20, decay_k=k)
    sub = np.sin(2 * np.pi * f * t)
    x = np.tanh(1.6 * (0.85 * sub + 0.45 * saw))
    return x * adsr(n, dur, 0.004, 0.035)


def sub_note(f, dur, a=0.03, r=0.25):
    n = int((dur + r) * SR)
    t = taxis(n)
    x = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return x * adsr(n, dur, a, r)


def lead_note(f, dur, rel=0.2, vib_hz=5.6, vib=0.0045, bright=4200.0):
    """Detuned band-limited saws with delayed vibrato: the run theme's hook voice."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    ramp = np.clip((t - 0.10) / 0.25, 0, 1)
    mod = (f * vib / vib_hz) * ramp * np.sin(2 * np.pi * vib_hz * t)
    h = np.arange(1, harm_count(f, 40) + 1)[:, None]
    lpg = 1 / np.sqrt(1 + (h * f / bright) ** 2)
    out = np.zeros(n)
    for det in (1.0, 2 ** (9 / 1200)):
        ph = 2 * np.pi * f * det * t + mod
        out += (lpg / h * np.sin(h * ph[None, :])).sum(0)
    return out * adsr(n, dur, 0.012, rel) * 0.35


def soft_lead_note(f, dur, rel=1.0, a=0.09):
    """Mellow flute/horn-ish voice for the menu: a few sine harmonics, slow vibrato, a little breath."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    ramp = np.clip((t - 0.2) / 0.5, 0, 1)
    ph = 2 * np.pi * f * t + (f * 0.004 / 5.0) * ramp * np.sin(2 * np.pi * 5.0 * t)
    x = np.sin(ph) + 0.38 * np.sin(2 * ph) + 0.14 * np.sin(3 * ph) + 0.05 * np.sin(4 * ph)
    rng = np.random.default_rng(int(f * 10))
    breath = lp(rng.standard_normal(n), 2400) * 0.02 * np.clip(t / 0.3, 0, 1)
    return (x * 0.45 + breath) * adsr(n, dur, a, rel)


def bell_note(f, ring=2.6, bright=2.2):
    """Glassy FM bell/mallet (harmonic 1:3 ratio so it can never go out of key)."""
    n = int(ring * SR)
    t = taxis(n)
    idx = bright * np.exp(-t / 0.35)
    x = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * 3 * f * t)) * np.exp(-t / 0.9)
    x += 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.45)
    x *= np.minimum(1.0, t / 0.002) * np.clip((ring - t) / 0.3, 0, 1)
    return x * 0.5


def pad_chord(freqs, dur, a, r, rng, detune=(-11, 0, 11), width=0.65):
    """Stereo detuned-saw pad (naive saws; the pad bus is low-passed, which hides the aliasing)."""
    n = int((dur + r) * SR)
    t = taxis(n)
    out = np.zeros((2, n))
    for f in freqs:
        for i, c in enumerate(detune):
            ff = f * 2 ** (c / 1200)
            saw = 2 * ((ff * t + rng.random()) % 1.0) - 1
            gl, gr = pan_gains((i - 1) * width)
            out[0] += saw * gl
            out[1] += saw * gr
    return out * adsr(n, dur, a, r) / (len(freqs) * len(detune)) * 1.4


def stab_chord(name: str, vel=1.0):
    tones = [t for t in chord_tones(name, 53)]
    return sum(pluck(hz(m), 0.14, k=9.0, tau=0.1, cap=18) for m in tones) * vel * 0.6


def make_drums(rng) -> dict[str, np.ndarray]:
    d: dict[str, np.ndarray] = {}
    # Kick: pitch-swept sine + tiny click, driven into a tanh.
    n = int(0.5 * SR)
    t = taxis(n)
    ph = 2 * np.pi * np.cumsum(46 + 130 * np.exp(-t / 0.028)) / SR
    click = lp(rng.standard_normal(n), 3500) * np.exp(-t / 0.004)
    d["kick"] = peak_norm(np.tanh(2.0 * (np.sin(ph) * np.exp(-t / 0.15) + 0.35 * click)))
    # Snare: noise body + two tonal partials.
    n = int(0.4 * SR)
    t = taxis(n)
    for i in range(3):
        noise = bp(rng.standard_normal(n), 1400, 9000) * np.exp(-t / 0.085)
        tone = np.sin(2 * np.pi * 187 * t) * np.exp(-t / 0.06) * 0.55 + np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03) * 0.25
        d[f"snare{i}"] = peak_norm(np.tanh(1.4 * (noise * 0.9 + tone)))
    # Clap: three flams + tail.
    n = int(0.35 * SR)
    t = taxis(n)
    noise = bp(rng.standard_normal(n), 1100, 4200)
    env = sum(np.exp(-np.clip(t - o, 0, None) / 0.012) * (t >= o) for o in (0.0, 0.011, 0.023))
    env = env + 0.5 * np.exp(-t / 0.09)
    d["clap"] = peak_norm(noise * env)
    # Hats.
    for i in range(3):
        n = int(0.12 * SR)
        t = taxis(n)
        d[f"hat{i}"] = peak_norm(hp(rng.standard_normal(n), 7500) * np.exp(-t / 0.028))
    n = int(0.55 * SR)
    t = taxis(n)
    d["open_hat"] = peak_norm(hp(rng.standard_normal(n), 6500) * np.exp(-t / 0.16) * np.clip((0.55 - t) / 0.1, 0, 1))
    # Crash (and its reverse for swells into a downbeat).
    n = int(3.2 * SR)
    t = taxis(n)
    crash = hp(rng.standard_normal(n), 3800) * np.exp(-t / 0.9) * np.minimum(1.0, t / 0.002)
    d["crash"] = peak_norm(crash)
    d["rev_crash"] = peak_norm(crash[::-1] * np.minimum(1.0, np.arange(n) / 4000.0))
    return d


def riser(dur: float, rng, f0=180.0, f1=2400.0):
    n = int(dur * SR)
    t = taxis(n)
    noise = hp(rng.standard_normal(n), 1800) * (t / dur) ** 2.2
    f = f0 * (f1 / f0) ** (t / dur)
    sweep = (2 * ((np.cumsum(f) / SR) % 1.0) - 1) * (t / dur) ** 2.6 * 0.5
    x = noise * 0.7 + lp(sweep, 4200)
    return peak_norm(x) * np.minimum(1.0, (dur - t) / 0.02)


def noise_swell(n: int, rng, lo=300, hi=2200):
    x = np.stack([bp(rng.standard_normal(n), lo, hi, order=2) for _ in range(2)])
    return peak_norm(x)


def make_ir(rt60: float, rng, predelay=0.018):
    n = int(rt60 * 1.15 * SR)
    t = taxis(n)
    irs = []
    for _ in range(2):
        noise = rng.standard_normal(n)
        body = lp(noise, 2600, 2) * np.exp(-6.9 * t / rt60)  # lows ring for the full RT60
        air = hp(noise, 2600, 2) * np.exp(-6.9 * t / (rt60 * 0.4))  # highs die much faster
        ir = body + 0.8 * air
        ir = np.concatenate([np.zeros(int(predelay * SR)), ir]) * np.minimum(1.0, np.arange(n + int(predelay * SR)) / 300.0)
        irs.append(ir / np.sqrt(np.sum(ir**2)))
    m = max(len(i) for i in irs)
    return np.stack([np.pad(i, (0, m - len(i))) for i in irs])


def reverb(send: np.ndarray, ir: np.ndarray) -> np.ndarray:
    return np.stack([signal.oaconvolve(send, ir[c])[: len(send)] for c in range(2)])


def pingpong(x: np.ndarray, delay: int, fb: float, taps: int, damp: float) -> np.ndarray:
    n = len(x)
    wet = np.zeros((2, n))
    for k in range(1, taps + 1):
        sh = k * delay
        if sh >= n:
            break
        wet[(k + 1) % 2, sh:] += (fb ** (k - 1)) * x[: n - sh]
    return lp(wet, damp, 2)


def sidechain(length: int, positions: list[int], depth: float, tau: float) -> np.ndarray:
    g = np.ones(length)
    w = int(tau * 6 * SR)
    t = np.arange(w) / SR
    dip = 1 - depth * np.exp(-t / tau)
    ramp = int(0.002 * SR)
    dip[:ramp] = np.linspace(1.0, dip[ramp], ramp)
    for p in positions:
        seg = g[p : p + w]
        seg *= dip[: len(seg)]
    return g


# ---------------------------------------------------------------------------- song scaffold


class Song:
    def __init__(self, bpm: int, bars: int, seed: int):
        assert (SR * 60) % (bpm * 4) == 0, "bpm must give a whole number of samples per 16th"
        self.bpm, self.bars = bpm, bars
        self.sps = SR * 60 // (bpm * 4)
        self.n = bars * 16 * self.sps
        self.len = self.n + TAIL
        self.rng = np.random.default_rng(seed)
        self.stems: dict[str, np.ndarray] = {}
        self.send = np.zeros(self.len)
        self.kicks: list[int] = []

    def at(self, bar: int, step: float = 0) -> int:
        return int(round((bar * 16 + step) * self.sps))

    def put(self, stem: str, x: np.ndarray, pos: int, gain=1.0, pan=0.0, send=0.0):
        if pos >= self.len:
            return
        buf = self.stems.setdefault(stem, np.zeros((2, self.len)))
        end = min(self.len, pos + x.shape[-1])
        seg = x[..., : end - pos]
        if x.ndim == 1:
            gl, gr = pan_gains(pan)
            buf[0, pos:end] += seg * gain * gl
            buf[1, pos:end] += seg * gain * gr
            mono = seg
        else:
            buf[:, pos:end] += seg * gain
            mono = seg.sum(0) * 0.5
        if send:
            self.send[pos:end] += mono * gain * send

    def drone(self, stem: str, x: np.ndarray, gain=1.0, send=0.0):
        """Continuous (loop-periodic) material of exactly one song length: not folded."""
        buf = self.stems.setdefault(stem, np.zeros((2, self.len)))
        buf[:, : self.n] += x[..., : self.n] * gain
        if send:
            self.send[: self.n] += (x.sum(0) if x.ndim == 2 else x)[: self.n] * gain * send * 0.5

    def loop_freq(self, f: float) -> float:
        """Nudge f (<0.01 Hz) so a sine completes whole cycles per loop and the seam stays click-free."""
        secs = self.n / SR
        return round(f * secs) / secs


def build_chord_map(sections):
    chord_at, sect_at = {}, {}
    for name, start, prog in sections:
        for i, c in enumerate(prog):
            chord_at[start + i] = c
            sect_at[start + i] = (name, i)
    return chord_at, sect_at


def chord_runs(chord_at, bars):
    runs, b = [], 0
    while b < bars:
        e = b
        while e + 1 < bars and chord_at[e + 1] == chord_at[b]:
            e += 1
        runs.append((b, e - b + 1, chord_at[b]))
        b = e + 1
    return runs


def place_motif(s: Song, stem: str, motif: list, start_bar: int, repeats: int, gain, send, note_fn, harmony=False, hgain=0.55, min_harm_len=4, octave=0):
    for r in range(repeats):
        base = (start_bar + 4 * r) * 16
        for step, name, ln in motif:
            m = midi(name) + octave
            pos = (base + step) * s.sps
            dur = ln * s.sps / SR * 0.96
            s.put(stem, note_fn(hz(m), dur), pos, gain=gain, pan=-0.12, send=send)
            if harmony and ln >= min_harm_len:
                s.put(stem, note_fn(hz(diatonic_shift(m, 2)), dur), pos, gain=gain * hgain, pan=0.18, send=send)


# ---------------------------------------------------------------------------- mastering


def measure(path: str) -> tuple[float, float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    summary = out[out.rfind("Summary:"):]
    lufs = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))
    return lufs, peak


def soft_limit(x: np.ndarray, ceil_db: float) -> np.ndarray:
    ceil = 10 ** (ceil_db / 20)
    knee = ceil * 0.72
    a = np.abs(x)
    over = a > knee
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (ceil - knee) * np.tanh((a[over] - knee) / (ceil - knee)))
    return y


def fold_tail(x: np.ndarray, n: int) -> np.ndarray:
    out = x[:, :n].copy()
    tail = x[:, n : n + TAIL]
    out[:, : tail.shape[1]] += tail
    return out


def master(mix: np.ndarray, n: int, target_lufs: float) -> np.ndarray:
    x = fold_tail(mix, n)
    # Circular high-pass (filter a 3x tiling and keep the middle) so the loop seam stays continuous.
    x = hp(np.concatenate([x, x, x], axis=1), 28, 2)[:, n : 2 * n]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.wav")
        gain = 1.0
        for _ in range(4):
            y = soft_limit(x * gain, PEAK_CEILING_DB)
            wavfile.write(path, SR, y.T.astype(np.float32))
            lufs, _ = measure(path)
            err = target_lufs - lufs
            if abs(err) < 0.15:
                break
            gain *= 10 ** (err / 20)
    return y


def to_ogg(wav_path: str, ogg_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path, "-c:a", "libvorbis", "-q:a", "5", ogg_path],
        check=True,
    )


def seam_report(y: np.ndarray, name: str):
    jump = np.max(np.abs(y[:, 0] - y[:, -1]))
    typical = np.percentile(np.abs(np.diff(y, axis=1)).max(0), 99.9)
    print(f"  [{name}] loop seam step {jump:.4f} vs 99.9th-pct sample step {typical:.4f}")


def stem_report(stems: dict[str, np.ndarray], n: int, gains: dict[str, float], targets: dict[str, float] | None = None):
    """Print each stem's level in the mix; with `targets` (RMS dBFS) also the gain that would hit them."""
    print("  stem levels in the mix (RMS dBFS over song / peak dBFS)" + (" -> gain that hits target:" if targets else ":"))
    for k, v in sorted(stems.items()):
        v = v * gains.get(k, 1.0)
        r = 20 * np.log10(np.sqrt(np.mean(v[:, :n] ** 2) + 1e-20) + 1e-12)
        line = f"    {k:10s} {r:7.1f}  {20 * np.log10(np.max(np.abs(v)) + 1e-12):7.1f}"
        if targets and k in targets:
            line += f"   target {targets[k]:6.1f} -> gain {gains.get(k, 1.0) * 10 ** ((targets[k] - r) / 20):.2f}"
        print(line)


# ---------------------------------------------------------------------------- mix bus


DEFAULT_DELAY_SOURCES = ("arp", 1.0), ("lead", 0.6), ("stab", 0.4), ("bells", 0.5)


def render_mix(s: Song, stem_gain: dict[str, float], cfg: dict) -> np.ndarray:
    stems = s.stems
    # brightness automation (dark <-> bright low-pass blend); "pad_filter" is the legacy single-stem form
    filters = dict(cfg.get("stem_filters", {}))
    if "pad_filter" in cfg:
        filters["pad"] = cfg["pad_filter"]
    for name, (lo, hi, keys) in filters.items():
        if name in stems:
            w = curve(s.len, keys, s.sps)
            stems[name] = lp(stems[name], lo, 2) * (1 - w) + lp(stems[name], hi, 2) * w
    # sidechain "pump" from the kick
    gains = {}
    for name, (depth, tau) in cfg["sc"].items():
        gains[name] = sidechain(s.len, s.kicks, depth, tau)
    send = s.send * gains.get("send", 1.0)
    for name in list(stems):
        if name in gains:
            stems[name] = stems[name] * gains[name]
    # ping-pong delay on the plucky / lead material
    delay = s.sps * cfg["delay_steps"]
    dsrc = np.zeros(s.len)
    for name, amt in cfg.get("delay_sources", DEFAULT_DELAY_SOURCES):
        if name in stems:
            dsrc += stems[name].sum(0) * 0.5 * amt * stem_gain.get(name, 1.0)
    wet = pingpong(dsrc, delay, cfg["delay_fb"], cfg["delay_taps"], 5500) * cfg["delay_gain"]
    # reverb send -> stereo return
    rev = reverb(send, make_ir(cfg["rt60"], np.random.default_rng(7)))
    mix = wet + rev * cfg["rev_gain"]
    for name, x in stems.items():
        mix = mix + x * stem_gain.get(name, 1.0)
    return mix

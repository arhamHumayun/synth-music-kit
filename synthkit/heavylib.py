"""Heavy / industrial / orchestral-synth instruments for the darker tracks.

Builds on musiclib.py. Nothing here uses samples: the "guitars" are tuned-down synth strings pushed
through a two-stage oversampled distortion and a speaker-cabinet filter (double-tracked left/right),
strings/brass/choir are subtractive synths, and the percussion kit is modelled (metal partials,
pitched sine bodies, filtered noise).

The workhorse is `subtractive_saw`: a band-limited saw whose spectrum is shaped each instant by a
2-pole resonant low-pass whose cutoff glides from `fc0` to `fc1`. Because it is evaluated on the
harmonics directly it needs no filter state, never aliases, and covers growl bass, strings, brass
and leads with one routine.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from synthkit.musiclib import SR, Song, adsr, bp, harm_count, hp, hz, lp, pan_gains, peak_norm, sub_note, taxis

# ---------------------------------------------------------------------------- distortion / cabinet


def drive(x: np.ndarray, pre: float, asym: float = 0.0, oversample: int = 4) -> np.ndarray:
    """tanh waveshaper at `oversample`x so the added harmonics fold back as little as possible."""
    up = signal.resample_poly(x, oversample, 1)
    y = np.tanh(pre * up + asym) - np.tanh(asym)
    return signal.resample_poly(y, 1, oversample)[: len(x)]


def cab(x: np.ndarray) -> np.ndarray:
    """Closed-back 4x12-ish: no sub, presence bump around 2 kHz, steep roll-off above ~4.6 kHz."""
    y = hp(x, 80, 2)
    y = lp(y, 4800, 4)
    return y + 0.85 * bp(y, 1300, 3200, 1) + 0.3 * bp(y, 2600, 4400, 1)


_GTR_CACHE: dict = {}


def gtr_note(f: float, dur: float, kind: str = "chug", seed: int = 0, pre=(10.0, 7.0), tight: float = 110.0,
             intervals=(0,), rms: float = 0.30) -> np.ndarray:
    """One distorted guitar note (or power chord via `intervals`), cached since riffs repeat constantly.

    kind="chug"   palm-muted: harmonics die within ~50 ms, the gate keeps the note short and tight.
    kind="sustain" ringing: slow decay, the distortion compresses it into a long singing sustain.
    """
    key = (round(f, 3), round(dur, 4), kind, seed, pre, tight, tuple(intervals), rms)
    if key in _GTR_CACHE:
        return _GTR_CACHE[key]
    rel = 0.025
    n = int((dur + rel) * SR)
    t = taxis(n)
    rng = np.random.default_rng(seed * 7919 + int(f * 10))
    k0, k1 = (24.0, 7.0) if kind == "chug" else (1.4, 0.6)
    src = np.zeros(n)
    for iv in intervals:
        ff = f * 2 ** (iv / 12) * 2 ** ((((seed % 2) * 2 - 1) * 3.0) / 1200)  # +-3 cents between takes
        h = np.arange(1, harm_count(ff, 110) + 1)[:, None]
        amp = h ** -0.8 * np.exp(-(k0 + k1 * (h - 1) / 10.0) * t[None, :])
        src += (amp * np.sin(2 * np.pi * ff * h * t[None, :] + rng.random() * 6.28)).sum(0)
    src = src / (np.max(np.abs(src)) + 1e-9)
    src += 0.25 * bp(rng.standard_normal(n), 1800, 5500, 1) * np.exp(-t / 0.004)  # pick attack
    y = drive(hp(src, tight, 2), pre[0])
    y = drive(hp(lp(y, 7000, 2), 230, 1), pre[1])
    y = cab(y) * adsr(n, dur, 0.001, rel)
    body = y[: max(1, int(dur * SR))]
    y = y * (rms / (np.sqrt(np.mean(body**2)) + 1e-9))
    _GTR_CACHE[key] = y
    return y


def place_gtr(s: Song, bar: int, step: float, note: int, steps: float, kind: str = "chug", gain: float = 1.0,
              send: float = 0.0, intervals=(0,), stem: str = "gtr", pre=(10.0, 7.0), tight: float = 110.0):
    """Double-tracked guitar: two takes (different detune/phase), 3.5 ms apart, hard left and right."""
    f = hz(note)
    dur = max(0.03, steps * s.sps / SR - (0.012 if kind == "chug" else 0.0))
    pos = s.at(bar, step)
    for take, (pan, off) in enumerate(((-0.85, 0), (0.85, int(0.0035 * SR)))):
        s.put(stem, gtr_note(f, dur, kind, seed=take, pre=pre, tight=tight, intervals=intervals), pos + off,
              gain=gain, pan=pan, send=send)


# ---------------------------------------------------------------------------- subtractive synth


def subtractive_saw(f: float, dur: float, rel: float, fc0: float, fc1: float, tau: float, q: float = 1.0,
                    a: float = 0.003, env_tau: float | None = None, detunes=(0.0,), hcap: int = 60,
                    tilt: float = 1.0, vib_hz: float = 0.0, vib: float = 0.0) -> np.ndarray:
    """Band-limited saw through a 2-pole low-pass (resonance `q`) whose cutoff glides fc0 -> fc1 (time const `tau`)."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    fc = fc1 + (fc0 - fc1) * np.exp(-t / tau)
    mod = np.zeros(n)
    if vib_hz > 0:
        ramp = np.clip((t - 0.12) / 0.3, 0, 1)
        mod = (f * vib / vib_hz) * ramp * np.sin(2 * np.pi * vib_hz * t)
    out = np.zeros(n)
    for c in detunes:
        ff = f * 2 ** (c / 1200)
        h = np.arange(1, harm_count(ff, hcap) + 1)[:, None]
        r = ff * h / fc[None, :]
        mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (r / q) ** 2)
        ph = 2 * np.pi * ff * t + mod
        out += (h ** -tilt * mag * np.sin(h * ph[None, :])).sum(0)
    return out / len(detunes) * adsr(n, dur, a, rel, tau=env_tau)


def growl_bass(f: float, dur: float, rel: float = 0.03, fc0: float = 2600.0, fc1: float = 520.0, tau: float = 0.12,
               pre: float = 5.0, sub: float = 0.55) -> np.ndarray:
    """Reese-style detuned saws + sub sine, driven: the Doom-ish synth bass snarl under the riffs."""
    saw = subtractive_saw(f, dur, rel, fc0, fc1, tau, q=2.4, a=0.004, detunes=(-9.0, 9.0), hcap=40)
    saw = drive(saw / (np.max(np.abs(saw)) + 1e-9), pre)
    n = len(saw)
    t = taxis(n)
    body = np.sin(2 * np.pi * f * t) * adsr(n, dur, 0.004, rel)
    saw = hp(saw, 70, 2)
    return saw * 0.8 + body * sub


def string_ensemble(f: float, dur: float, stacc: bool = True, bright: float = 4200.0) -> np.ndarray:
    """Stereo saw-ensemble 'strings'; stacc = short bowed-spiccato envelope, otherwise slow swell."""
    if stacc:
        a, rel, env_tau = 0.010, 0.07, max(0.05, dur * 0.7)
    else:
        a, rel, env_tau = 0.16, 0.45, None
    sets = ((-13, -5, 4, 12), (-11, -4, 6, 13))
    chans = [
        subtractive_saw(f, dur, rel, bright, bright * 0.55, 0.35, q=0.9, a=a, env_tau=env_tau, detunes=d, hcap=36,
                        vib_hz=5.2, vib=0.003 if not stacc else 0.0)
        for d in sets
    ]
    return np.stack(chans) * 0.5


def brass_note(f: float, dur: float, rel: float = 0.09, swell: float = 0.07) -> np.ndarray:
    """Synth brass: filter opens quickly after the attack (the 'bwaah'), two detuned saws."""
    return subtractive_saw(f, dur, rel, 450.0, 3300.0, swell, q=1.6, a=0.018, detunes=(-6.0, 6.0), hcap=40,
                           vib_hz=5.6, vib=0.003) * 0.7


_VOWELS = {
    "ah": ((730, 120), (1090, 150), (2440, 220)),
    "oh": ((570, 100), (840, 130), (2410, 220)),
    "oo": ((300, 80), (870, 130), (2240, 220)),
    "ee": ((270, 80), (2290, 220), (3010, 260)),
}


def choir_note(f: float, dur: float, vowel: str = "ah", rel: float = 0.7, a: float = 0.35, seed: int = 0) -> np.ndarray:
    """Formant-filtered saw voices (3 detuned) — a synthetic 'aah' choir, wide stereo."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    rng = np.random.default_rng(seed + int(f))
    chans = [np.zeros(n), np.zeros(n)]
    for i, c in enumerate((-9.0, 0.0, 9.0)):
        ff = f * 2 ** (c / 1200)
        ramp = np.clip((t - 0.3) / 0.5, 0, 1)
        ph = 2 * np.pi * ff * t + (ff * 0.005 / 5.3) * ramp * np.sin(2 * np.pi * 5.3 * t + i)
        h = np.arange(1, harm_count(ff, 36) + 1)[:, None]
        voice = (h ** -1.0 * np.sin(h * ph[None, :])).sum(0)
        shaped = sum(g * bp(voice, fc - bw / 2, fc + bw / 2, 2) for g, (fc, bw) in zip((1.0, 0.55, 0.3), _VOWELS[vowel]))
        shaped = shaped + 0.5 * lp(voice, 450, 2)  # body: formant filters alone drop the fundamental
        pan = (i - 1) * 0.7
        gl, gr = pan_gains(pan)
        chans[0] += shaped * gl
        chans[1] += shaped * gr
    breath = lp(rng.standard_normal(n), 3000) * 0.015
    return (np.stack(chans) / 3 + breath) * adsr(n, dur, a, rel)


def pulse_lead(f: float, dur: float, duty: float = 0.30, rel: float = 0.06, vib_hz: float = 6.0, vib: float = 0.004) -> np.ndarray:
    """Bright band-limited pulse lead with late vibrato: the retro/MIDI-ish hook voice."""
    n = int((dur + rel) * SR)
    t = taxis(n)
    ramp = np.clip((t - 0.10) / 0.2, 0, 1)
    mod = (f * vib / vib_hz) * ramp * np.sin(2 * np.pi * vib_hz * t)
    h = np.arange(1, harm_count(f, 50) + 1)[:, None]
    ph = 2 * np.pi * f * t + mod
    coef = 2 * np.sin(np.pi * h * duty) / (np.pi * h)
    x = (coef * np.cos(h * ph[None, :])).sum(0)
    return x * adsr(n, dur, 0.004, rel) * 0.45


# ---------------------------------------------------------------------------- percussion


def _modal(n: int, partials, rng, noise_hz=(2000, 9000), noise_decay=0.01) -> np.ndarray:
    """Struck-metal model: inharmonic sine partials (freq, amp, decay) + a bright noise click."""
    t = taxis(n)
    x = np.zeros(n)
    for f, a, d in partials:
        x += a * np.sin(2 * np.pi * f * t + rng.random() * 6.28) * np.exp(-t / d)
    x += 0.5 * bp(rng.standard_normal(n), *noise_hz, 2) * np.exp(-t / noise_decay)
    return peak_norm(x * np.minimum(1.0, t / 0.0007))


def make_heavy_kit(rng) -> dict[str, np.ndarray]:
    k: dict[str, np.ndarray] = {}
    # Metal kick: tight sub thump + hard beater click (double-bass friendly).
    n = int(0.40 * SR)
    t = taxis(n)
    ph = 2 * np.pi * np.cumsum(56 + 115 * np.exp(-t / 0.018)) / SR
    click = bp(rng.standard_normal(n), 2400, 6800, 2) * np.exp(-t / 0.0026)
    k["kick"] = peak_norm(np.tanh(1.7 * (np.sin(ph) * np.exp(-t / 0.10) + 0.6 * click)))
    # Big boom kick for downbeats / drops.
    n = int(1.0 * SR)
    t = taxis(n)
    ph = 2 * np.pi * np.cumsum(38 + 95 * np.exp(-t / 0.05)) / SR
    k["boom_kick"] = peak_norm(np.tanh(1.5 * (np.sin(ph) * np.exp(-t / 0.33) + 0.35 * bp(rng.standard_normal(n), 1800, 6000, 2) * np.exp(-t / 0.004))))
    # Snares: crack + body + a little tone (three takes).
    n = int(0.45 * SR)
    t = taxis(n)
    for i in range(3):
        crack = bp(rng.standard_normal(n), 1800, 8500) * np.exp(-t / 0.075)
        body = bp(rng.standard_normal(n), 380, 1500) * np.exp(-t / 0.12) * 0.7
        tone = np.sin(2 * np.pi * 192 * t) * np.exp(-t / 0.07) * 0.5
        k[f"snare{i}"] = peak_norm(np.tanh(1.8 * (crack + body + tone)))
    # Toms (floor, mid, high).
    for name, f0 in (("tom_lo", 78.0), ("tom_mid", 112.0), ("tom_hi", 160.0)):
        n = int(0.5 * SR)
        t = taxis(n)
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.8 * np.exp(-t / 0.03))) / SR
        click = bp(rng.standard_normal(n), 1500, 5000, 2) * np.exp(-t / 0.004)
        k[name] = peak_norm(np.tanh(1.3 * (np.sin(ph) * np.exp(-t / 0.18) + 0.4 * click)))
    # Hats / cymbals.
    for i in range(3):
        n = int(0.12 * SR)
        k[f"hat{i}"] = peak_norm(hp(rng.standard_normal(n), 7500) * np.exp(-taxis(n) / 0.028))
    n = int(0.6 * SR)
    t = taxis(n)
    k["open_hat"] = peak_norm(hp(rng.standard_normal(n), 6500) * np.exp(-t / 0.17) * np.clip((0.6 - t) / 0.1, 0, 1))
    n = int(3.6 * SR)
    t = taxis(n)
    crash = hp(rng.standard_normal(n), 3600) * np.exp(-t / 1.0) * np.minimum(1.0, t / 0.002)
    k["crash"] = peak_norm(crash)
    k["rev_crash"] = peak_norm(crash[::-1] * np.minimum(1.0, np.arange(n) / 4000.0))
    china = bp(rng.standard_normal(n), 1800, 10000, 2) * (1 + 0.7 * np.sin(2 * np.pi * 71 * t)) * np.exp(-t / 0.55)
    k["china"] = peak_norm(china * np.minimum(1.0, t / 0.0015))
    # Industrial metal: anvil, pipe, plate, chain, sub boom.
    k["anvil"] = _modal(int(1.4 * SR), [(330, 1.0, 0.70), (330 * 2.32, 0.8, 0.45), (330 * 4.25, 0.6, 0.30), (330 * 6.63, 0.45, 0.20), (330 * 9.38, 0.3, 0.12)], rng)
    k["pipe"] = _modal(int(1.8 * SR), [(205, 1.0, 0.95), (205 * 1.59, 0.7, 0.8), (205 * 2.14, 0.6, 0.6), (205 * 2.65, 0.5, 0.5), (205 * 3.5, 0.35, 0.35)], rng, noise_hz=(1200, 5000), noise_decay=0.02)
    k["plate"] = _modal(int(2.2 * SR), [(92, 1.0, 1.1), (92 * 2.7, 0.8, 0.8), (92 * 5.1, 0.6, 0.5), (92 * 8.6, 0.4, 0.3)], rng, noise_hz=(600, 3500), noise_decay=0.03)
    n = int(0.5 * SR)
    t = taxis(n)
    chain = np.zeros(n)
    for _ in range(8):
        o = int(rng.uniform(0, 0.28) * SR)
        m = n - o
        chain[o:] += bp(rng.standard_normal(m), 3000, 8500, 2) * np.exp(-taxis(m) / 0.016) * rng.uniform(0.4, 1.0)
    k["chain"] = peak_norm(chain)
    n = int(1.6 * SR)
    t = taxis(n)
    ph = 2 * np.pi * np.cumsum(24 + 70 * np.exp(-t / 0.4)) / SR
    k["sub_boom"] = peak_norm(np.sin(ph) * np.exp(-t / 0.55) + 0.2 * lp(rng.standard_normal(n), 220) * np.exp(-t / 0.25))
    return k


# ---------------------------------------------------------------------------- composing helpers


def parse_riff(spec: str) -> list[tuple[int, int, int, bool]]:
    """One bar of riff as 16 space-separated step tokens -> [(step, semitones, length, power_chord)].

    `.` rest, `_` hold the previous note one more step, `N` note N semitones above the riff root,
    `N+` the same as a power chord (root + fifth). Example: "0 0 . 0 0 . 0 0 0 0 . 0 0 . 6+ 5".
    """
    toks = spec.split()
    assert len(toks) == 16, f"riff bar needs 16 steps, got {len(toks)}: {spec!r}"
    notes: list[list] = []  # [step, semis, length, power]
    for i, tok in enumerate(toks):
        if tok == ".":
            continue
        if tok == "_":
            notes[-1][2] += 1
            continue
        power = tok.endswith("+")
        notes.append([i, int(tok.rstrip("+")), 1, power])
    return [tuple(n) for n in notes]


def place_riff(s: Song, bar: int, spec: str, root: int, gain: float = 1.0, send: float = 0.0, stem: str = "gtr",
               power: bool = False, long_kind_steps: int = 5) -> list[tuple[int, int, int, bool]]:
    """Place one riff bar as double-tracked guitar; returns the parsed notes so callers can lock bass/kick to them."""
    notes = parse_riff(spec)
    for st, semi, ln, pw in notes:
        kind = "sustain" if ln >= long_kind_steps else "chug"
        place_gtr(s, bar, st, root + semi, ln, kind=kind, gain=gain, send=send,
                  intervals=(0, 7) if (pw or power) else (0,), stem=stem)
    return notes


def hit(s: Song, stem: str, sample: np.ndarray, bar: int, step: float, gain: float = 1.0, pan: float = 0.0,
        send: float = 0.0, kick: bool = False, rng=None, jitter: float = 0.12) -> None:
    """Place a drum/percussion sample with light velocity jitter; `kick=True` also feeds the sidechain."""
    v = gain * (1.0 - jitter * float(rng.random())) if rng is not None else gain
    pos = s.at(bar, step)
    if kick:
        s.kicks.append(pos)
    s.put(stem, sample, pos, gain=v, pan=pan, send=send)


def place_notes(s: Song, stem: str, notes, bar0: int, note_fn, gain: float = 1.0, send: float = 0.0, pan: float = 0.0,
                legato: float = 0.96) -> None:
    """Place [(bar_offset, step, midi_note, length_steps), ...] using note_fn(freq_hz, dur_sec) -> samples."""
    for bar_off, step, m, ln in notes:
        s.put(stem, note_fn(hz(m), ln * s.sps / SR * legato), s.at(bar0 + bar_off, step), gain=gain, pan=pan, send=send)


def screamer(f: float, d: float, pre: float = 3.0, rms: float = 0.30) -> np.ndarray:
    """A saw lead pushed into distortion: the synth 'siren/scream' that rides over the riffs."""
    from synthkit.musiclib import lead_note
    x = lead_note(f, d, rel=0.25, vib=0.006, bright=6000.0)
    y = lp(drive(x / (np.max(np.abs(x)) + 1e-9), pre), 5500, 2)
    body = y[: max(1, int(d * SR))]
    return y * (rms / (np.sqrt(np.mean(body**2)) + 1e-9))


def triad(root: int, quality: str = "min") -> list[int]:
    """Root-position triad as MIDI numbers: 'min', 'maj', 'dim', 'sus2', 'pow' (root, fifth, octave)."""
    third = {"min": 3, "maj": 4, "dim": 3, "sus2": 2, "pow": 7}[quality]
    fifth = {"dim": 6, "pow": 12}.get(quality, 7)
    return [root, root + third, root + fifth]


def place_roll(s: Song, bar: int, start_step: float, end_step: float, sample: np.ndarray, rate: float = 0.5,
               g0: float = 0.3, g1: float = 1.0, send: float = 0.2, rng=None, stem: str = "snare") -> None:
    """Crescendo roll of `sample` every `rate` steps (0.5 = 32nd notes) between two steps of a bar."""
    n = max(1, int(round((end_step - start_step) / rate)))
    for i in range(n):
        hit(s, stem, sample, bar, start_step + i * rate, g0 + (g1 - g0) * i / max(1, n - 1), send=send, rng=rng, jitter=0.1)


def bass_follow(s: Song, bar: int, notes, bass_root: int, gain: float = 1.0, sub_gain: float = 1.0,
                even_only: bool = True) -> None:
    """Growl bass + sub locked to a parsed riff bar (8th-note pulse unless a note is held)."""
    step_t = s.sps / SR
    for st, semi, ln, pw in notes:
        if even_only and st % 2 and ln < 2:
            continue
        d = max(step_t * 1.6, ln * step_t * 0.92)
        s.put("bass", growl_bass(hz(bass_root + semi), d), s.at(bar, st), gain=gain)
        s.put("sub", sub_note(hz(bass_root + semi), d, a=0.006, r=0.04), s.at(bar, st), gain=sub_gain)


def near_pitch(pc: int, target: int) -> int:
    """MIDI note with pitch class `pc` closest to `target`."""
    return pc + 12 * round((target - pc) / 12)



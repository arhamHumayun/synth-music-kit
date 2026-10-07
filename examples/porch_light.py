"""Porch Light: a small acoustic folk piece built from the physical-model instruments in synthkit/acoustic.py.

24 bars, 90 BPM, G major (64 seconds). A fingerpicked guitar, upright bass, piano, cello, flute and a soft drum kit, with
the little human touches that make acoustic music feel played: notes land a few milliseconds early or late, every hit has
its own loudness, strums roll from the low string to the high one, and picked notes ring until the next bar.

  bars  0-3    guitar alone, fingerpicking G | Em | C | D
  bars  4-7    upright bass, kick and soft hats join; the piano plays two held chords
  bars  8-11   the cello plays the tune over the guitar, bass and a cross-stick backbeat with shaker
  bars 12-15   the flute takes the tune an octave up, the cello a third below, the piano rolls arpeggios
  bars 16-19   breakdown: cello holds long chord tones, the piano arpeggiates slowly, drums thin to kick and shaker
  bars 20-23   everyone, the tune on cello and flute; the last bar strips to the guitar so the loop returns to the intro

Run it with:  python examples/porch_light.py      (about 40 seconds)
There is no sampling: every sound is computed. Judge it by ear, then tune the gains in STEMS.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from synthkit.acoustic import acoustic_guitar, acoustic_kit, air_flute, cello, felt_piano, upright_bass  # noqa: E402
from synthkit.musiclib import SR, Song, hz, render_mix  # noqa: E402
from synthkit.render import render_track  # noqa: E402
from synthkit.theory import harmonize, off_scale, transpose  # noqa: E402

BPM = 90
BARS = 24
LAST_BAR = BARS - 1
TARGET_LUFS = -16.0
SEED = 90

G_MAJOR = (7, 9, 11, 0, 2, 4, 6)
PROGRESSION = ["G", "Em", "C", "D"]                    # one chord per bar, repeated
# The tune (4 bars over the progression): (bar_offset, step, midi, length_in_16ths).
MELODY = [
    (0, 0, 71, 6), (0, 6, 69, 2), (0, 8, 67, 4), (0, 12, 71, 4),
    (1, 0, 76, 8), (1, 8, 74, 4), (1, 12, 71, 4),
    (2, 0, 72, 6), (2, 6, 71, 2), (2, 8, 67, 4), (2, 12, 76, 4),
    (3, 0, 74, 8), (3, 8, 69, 4), (3, 12, 66, 4),
]
assert not off_scale(MELODY, G_MAJOR)

# Guitar: for fingerpicking, a bass note, an alternate bass and the treble strings; for strums, the whole open chord.
GUITAR = {
    "G": dict(bass=43, alt=50, treble=(55, 59), strum=(43, 47, 50, 55, 59, 67)),
    "Em": dict(bass=40, alt=47, treble=(55, 59), strum=(40, 47, 52, 55, 59, 64)),
    "C": dict(bass=48, alt=43, treble=(55, 64), strum=(48, 52, 55, 60, 64)),
    "D": dict(bass=38, alt=45, treble=(54, 62), strum=(50, 57, 62, 66)),
}
# The picking pattern over a bar: (step, string). The alternating bass is what makes it sound like fingerstyle.
PICKING = [(0, "bass"), (2, 0), (4, 1), (6, 0), (8, "alt"), (10, 0), (12, 1), (14, 0)]
BASS_ROOT = {"G": 31, "Em": 28, "C": 36, "D": 38}      # the upright plays an octave under the guitar's bass strings
PIANO = {                                              # left-hand note and right-hand triad
    "G": (43, (59, 62, 67)), "Em": (40, (59, 64, 67)), "C": (36, (60, 64, 67)), "D": (38, (62, 66, 69)),
}
CELLO_LONG = {"G": 62, "Em": 59, "C": 64, "D": 66}      # a chord tone to hold in the breakdown
PIANO_ROLL = [(0, "lh"), (4, 0), (6, 1), (8, 2), (12, 1), (14, 0)]

# Human feel: standard deviation of timing in seconds, range of loudness, strum spread per string.
TIMING_JITTER_S = 0.004
LOUDNESS = (0.82, 1.0)
STRUM_SPREAD_S = 0.016

# Mix gains (linear) per instrument and the levels the report should hit (RMS dBFS).
STEMS = dict(guitar=0.65, bass=1.35, piano=0.55, cello=0.38, flute=0.3, kick=0.9, snare=1.6, hats=1.1, shaker=0.45)
TARGETS = dict(guitar=-21, bass=-22, piano=-24, cello=-22, flute=-26, kick=-24, snare=-28, hats=-38, shaker=-36)


def build():
    s = Song(BPM, BARS, SEED)
    rng = s.rng
    kit = acoustic_kit(rng)
    bar_t = 16 * s.sps / SR
    step_t = s.sps / SR
    voices: dict = {}

    def voice(name, fn, f, dur):
        key = (name, round(f, 2), round(dur, 2))
        if key not in voices:
            voices[key] = fn(f, dur)
        return voices[key]

    def when(bar, step, extra_s=0.0):
        """A sample position with a little human timing error (never before the start of the loop)."""
        return max(0, s.at(bar, step) + int((rng.normal(0, TIMING_JITTER_S) + extra_s) * SR))

    def loud(base=1.0):
        return base * float(rng.uniform(*LOUDNESS))

    def pluck(bar, step, midi, gain, dur=None):
        left = dur if dur is not None else (16 - step) * step_t + 0.1
        s.put("guitar", voice("gtr", acoustic_guitar, hz(midi), left), when(bar, step), gain=loud(gain), pan=-0.2, send=0.22)

    def strum(bar, step, notes, gain, down=True):
        ordered = notes if down else notes[::-1]
        for i, m in enumerate(ordered):
            s.put("guitar", voice("gtr", acoustic_guitar, hz(m), 1.8), when(bar, step, i * STRUM_SPREAD_S),
                  gain=loud(gain * (1.0 - 0.04 * i)), pan=-0.2, send=0.22)

    def note(stem, name, fn, bar, step, midi, length, gain, pan, send):
        dur = length * step_t
        s.put(stem, voice(name, fn, hz(midi), dur), when(bar, step), gain=loud(gain), pan=pan, send=send)

    def tune(stem, name, fn, notes, bar0, gain, pan, send):
        for b, st, m, ln in notes:
            note(stem, name, fn, bar0 + b, st, m, ln * 0.96, gain, pan, send)

    harmony = harmonize(MELODY, G_MAJOR, -2)             # a voice a third below, on the long notes
    flute_tune = transpose(MELODY, 12)                   # the flute plays the tune an octave up
    assert not off_scale(harmony, G_MAJOR) and not off_scale(flute_tune, G_MAJOR)

    for bar in range(BARS):
        chord = PROGRESSION[bar % 4]
        g = GUITAR[chord]
        last = bar == LAST_BAR
        strong = bar in (12, 20)

        # Guitar: fingerpicking all the way through; a rolled strum opens the big sections.
        if strong:
            strum(bar, 0, g["strum"], 0.8)
        for step, string in PICKING:
            if strong and step == 0:
                continue
            midi = g["bass"] if string == "bass" else g["alt"] if string == "alt" else g["treble"][string]
            pluck(bar, step, midi, 1.0 if step in (0, 8) else 0.72)

        # Upright bass: root, then fifth, from bar 4 until the last bar.
        if 4 <= bar < LAST_BAR:
            root = BASS_ROOT[chord]
            note("bass", "bass", upright_bass, bar, 0, root, 7, 1.0, 0.0, 0.08)
            note("bass", "bass", upright_bass, bar, 8, root + 7, 6, 0.8, 0.0, 0.08)

        # Piano: two held chords in bars 6-7, rolling arpeggios 12-15, slow arpeggios 16-19, comping 20-22.
        lh, rh = PIANO[chord]
        if bar in (6, 7):
            for i, m in enumerate((lh,) + rh):
                s.put("piano", voice("piano", felt_piano, hz(m), 2.4), when(bar, 0, i * 0.02), gain=loud(0.8), pan=0.2, send=0.3)
        elif 12 <= bar < 16 or 20 <= bar < LAST_BAR:
            for step, key in PIANO_ROLL:
                m = lh if key == "lh" else rh[key]
                note("piano", "piano", felt_piano, bar, step, m, 6 if key == "lh" else 4, 0.7 if key == "lh" else 0.55, 0.2, 0.3)
        elif 16 <= bar < 20:
            for i, m in enumerate((lh,) + rh):
                note("piano", "piano", felt_piano, bar, i * 4, m, 8, 0.65, 0.2, 0.3)

        # Cello: long chord tones in the breakdown.
        if 16 <= bar < 20:
            note("cello", "cello", cello, bar, 0, CELLO_LONG[chord], 15, 0.8, -0.1, 0.35)

        # Drums.
        if 4 <= bar < LAST_BAR:
            thin = 16 <= bar < 20
            full = bar >= 8
            kick_steps = (0,) if bar < 8 else (0, 10)
            for st in kick_steps:
                s.put("kick", kit["kick"], when(bar, st), gain=loud(1.0 if st == 0 else 0.7), send=0.15)
                s.kicks.append(s.at(bar, st))
            if full and not thin:
                for st in (4, 12):
                    s.put("snare", kit[f"rim{int(rng.integers(2))}"], when(bar, st), gain=loud(0.9), send=0.2)
            if bar < 8:
                for st in range(0, 16, 2):
                    s.put("hats", kit[f"hat{int(rng.integers(3))}"], when(bar, st), gain=loud(0.5 if st % 4 == 0 else 0.3), pan=0.2, send=0.1)
            else:
                for st in range(0, 16, 2):
                    s.put("shaker", kit[f"shaker{int(rng.integers(2))}"], when(bar, st), gain=loud(0.8 if st % 4 == 0 else 0.5), pan=0.25, send=0.1)
                    s.put("shaker", kit[f"shaker{int(rng.integers(2))}"], when(bar, st + 1), gain=loud(0.3), pan=0.25, send=0.1)
            if bar in (7, 15, 19):                       # a brush swell and an open hat lead into the next section
                s.put("snare", kit["brush"], when(bar, 12), gain=0.9, send=0.3)
                s.put("hats", kit["open_hat"], s.at(bar + 1) - int(0.01 * SR), gain=0.5, pan=0.2, send=0.2)

    # The tune: cello alone, then flute with the cello a third below, then both for the finish.
    tune("cello", "cello", cello, MELODY, 8, 1.0, -0.1, 0.35)
    tune("flute", "flute", air_flute, flute_tune, 12, 0.9, 0.25, 0.4)
    tune("cello", "cello", cello, harmony, 12, 0.7, -0.1, 0.35)
    tune("cello", "cello", cello, MELODY, 20, 1.0, -0.1, 0.35)
    tune("flute", "flute", air_flute, flute_tune, 20, 0.85, 0.25, 0.4)

    return render_mix(s, STEMS, dict(
        rt60=1.5, rev_gain=0.5, delay_steps=3, delay_fb=0.3, delay_taps=3, delay_gain=0.1,
        delay_sources=(("flute", 0.4),), sc={},
    )), s.n, s


if __name__ == "__main__":
    render_track("porch-light", build, STEMS, out_dir=os.path.join(os.path.dirname(__file__), "..", "out"),
                 target_lufs=TARGET_LUFS, targets=TARGETS)

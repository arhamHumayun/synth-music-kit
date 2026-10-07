"""Hello loop: the smallest complete song. Eight bars, a pad, a bass, drums and a four-note tune, in A minor at 100 BPM.

    python examples/hello_loop.py        ->  out/hello-loop.ogg  (a WAV if ffmpeg is missing)

Read it top to bottom: a song is a table of chords, a table of notes, and a few layer calls per bar.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from synthkit.arranger import Arranger  # noqa: E402
from synthkit.render import render_track  # noqa: E402
from synthkit.theory import A_MINOR  # noqa: E402

BPM, BARS = 100, 8                        # BPM must divide 661500 (100 does); see the README
AM, F, C, G = (9, "min"), (5, "maj"), (0, "maj"), (7, "maj")   # a chord is (pitch class, quality); A = 9
PROGRESSION = [AM, F, C, G]               # one chord per bar, repeated
# A note is (bar_offset, step, midi_number, length). A bar has 16 steps (16th notes). 72 is the C above middle C.
TUNE = [(0, 0, 76, 4), (0, 6, 72, 2), (0, 8, 69, 6), (1, 0, 72, 4), (1, 6, 77, 2), (1, 8, 76, 6),
        (2, 0, 79, 4), (2, 6, 76, 2), (2, 8, 72, 6), (3, 0, 74, 8), (3, 8, 71, 8)]
STEMS = dict(kick=0.7, snare=0.5, hats=0.5, bass=0.3, sub=0.2, pad=0.5, lead=0.7, fx=0.5)   # gain per instrument


def build():
    A = Arranger(BPM, BARS, seed=1)
    for bar in range(BARS):
        chord = PROGRESSION[bar % 4]
        A.pad(bar, chord, voice="strings_sus", gain=0.7)         # soft sustained chord
        A.bass(bar, chord, style="8th", gain=0.9)                # growly 8th-note bass plus a sub underneath
        A.drums(bar, "rock" if bar >= 2 else "half", 0.9)        # drums: a kick, snare and hats groove
    for bar0 in (0, 4):                                          # the tune, twice; `scale` checks every note is in key
        A.melody("lead", TUNE, bar0, "lead", scale=A_MINOR)
    A.accent(0, crash=0.8)                                       # a crash on the downbeat
    cfg = dict(rt60=2.2, rev_gain=0.5, delay_steps=3, delay_fb=0.3, delay_taps=3, delay_gain=0.2,
               sc=dict(pad=(0.3, 0.12), bass=(0.3, 0.08), sub=(0.3, 0.08)))   # sidechain: duck these when the kick hits
    return A.finish(STEMS, cfg)


if __name__ == "__main__":
    render_track("hello-loop", build, STEMS, out_dir=os.path.join(os.path.dirname(__file__), "..", "out"))

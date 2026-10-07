"""Stage 1, the short version: the arcade theme squeezed into 16 bars (27 seconds, 140 BPM, D minor).

The same four-bar hook (hero_motif.py, also used by the menu) and the same build as the game's full-length stage 1 track, one section per hook:
  bars  0-3   intro: soft pad, a sparse arp that fills in, hats, a kick roll into the groove
  bars  4-7   groove: full drums, bass and arp
  bars  8-11  lead: the hook enters on the saw lead, with chord stabs and a counter arp
  bars 12-15  final: the hook again, harmonised a third up; the last bar strips back so the loop returns to the intro

Run it with:  python examples/stage1_short.py     (about 10 seconds)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hero_motif import MAIN, MOTIF  # noqa: E402  (the hook is shared with menu_short.py)

from synthkit.musiclib import (  # noqa: E402
    SR, Song, arp_tones, bass_note, bass_root, build_chord_map, chord_runs, hz, lead_note, make_drums, pad_chord,
    place_motif, pluck, render_mix, riser, stab_chord, sub_note, voice_lead,
)
from synthkit.render import render_track  # noqa: E402

BPM = 140
BARS = 16
LAST_BAR = BARS - 1
TARGET_LUFS = -15.5
SEED = 140

SECTIONS = [("intro", 0, MAIN), ("groove", 4, MAIN), ("lead", 8, MAIN), ("final", 12, MAIN)]
FULL_SECTIONS = ("groove", "lead", "final")           # sections with the whole band playing
ARP = [0, 2, 3, 2, 0, 2, 3, 2, 1, 2, 4, 2, 1, 2, 3, 2]            # indices into arp_tones(): root, 3rd, 5th, octave...
COUNTER_ARP = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 3]    # the off-beat arp an octave up
STAB_STEPS = (3, 6, 11)

# Mix gains (linear) per instrument.
STEMS = dict(kick=0.44, snare=0.50, hats=0.62, bass=0.33, pad=0.57, arp=0.66, arp2=0.76, stab=0.96, lead=0.55, fx=0.60)


def build():
    s = Song(BPM, BARS, SEED)
    rng = s.rng
    drums = make_drums(rng)
    chord_at, sect_at = build_chord_map(SECTIONS)
    bar_t = 16 * s.sps / SR
    step_t = s.sps / SR

    # Pads: one voice-led chord per run of identical chords. The intro swells in slowly, later sections hit at once.
    prev = None
    for b, nb, name in chord_runs(chord_at, BARS):
        v = voice_lead(prev, name)
        prev = v
        slow = sect_at[b][0] == "intro"
        s.put("pad", pad_chord([hz(m) for m in v], nb * bar_t - 0.05, 0.9 if slow else 0.06, 0.5, rng), s.at(b),
              gain=1.35 if slow else 1.0, send=0.35)

    # Per bar: bass, arp, counter arp, stabs.
    for bar in range(BARS):
        sect, lb = sect_at[bar]
        ch = chord_at[bar]
        root, tones = hz(bass_root(ch)), arp_tones(ch)
        last = bar == LAST_BAR
        if sect in FULL_SECTIONS and not last:
            for i, off in enumerate([0, 0, 12, 0, 0, 0, 12, 0]):          # 8th-note bass, jumping an octave twice a bar
                s.put("bass", bass_note(root * 2 ** (off / 12), 2 * step_t * 0.88), s.at(bar, i * 2), gain=1.0)
        elif sect == "intro" and lb >= 2:
            s.put("bass", sub_note(root, bar_t * 0.96), s.at(bar), gain=0.8)
        steps = range(0, 16, 2) if (sect == "intro" and lb < 2) else range(16)
        if last:
            steps = range(0, 16, 2)
        for i in steps:
            accent = 1.0 if i % 4 == 0 else (0.82 if i % 2 == 0 else 0.66)
            s.put("arp", pluck(hz(tones[ARP[i]]), step_t * 1.5, k=6.0, tau=0.17), s.at(bar, i), gain=accent * (0.7 if last else 1.0),
                  pan=0.15 if i % 2 else -0.15)
        if sect in ("lead", "final") and not last:
            for i in range(1, 16, 2):
                m = tones[COUNTER_ARP[(i + 5) % 16]] + 12
                s.put("arp2", pluck(hz(m), step_t * 1.4, k=8.0, tau=0.12), s.at(bar, i), gain=0.55, pan=0.3 if i % 4 == 1 else -0.3)
            for st in STAB_STEPS:
                s.put("stab", stab_chord(ch), s.at(bar, st), gain=0.8, send=0.2)

    # The hook on the lead, then again harmonised.
    place_motif(s, "lead", MOTIF, 8, 1, 1.0, 0.28, lambda f, d: lead_note(f, d))
    place_motif(s, "lead", MOTIF, 12, 1, 1.0, 0.28, lambda f, d: lead_note(f, d), harmony=True)

    # Drums.
    for bar in range(BARS):
        sect, lb = sect_at[bar]
        last = bar == LAST_BAR

        def hit(stem, name, step, gain, send=0.0, pan=0.0, kick=False):
            pos = s.at(bar, step)
            if kick:
                s.kicks.append(pos)
            s.put(stem, drums[name], pos, gain=gain * float(rng.uniform(0.85, 1.0)), send=send, pan=pan)

        if sect in FULL_SECTIONS:
            for st in ([0] if last else [0, 4, 8, 12]):
                hit("kick", "kick", st, 1.0, kick=True)
            if not last:
                for st in (4, 12):
                    hit("snare", f"snare{int(rng.integers(3))}", st, 1.0, send=0.18)
                    hit("snare", "clap", st, 0.55, send=0.22)
            for i in range(16):
                open_hat = i % 4 == 2
                hit("hats", "open_hat" if open_hat else f"hat{int(rng.integers(3))}", i,
                    0.55 if open_hat else (0.5 if i % 4 == 0 else 0.28), pan=0.25 if i % 4 else -0.15)
        elif lb >= 2:                                                     # intro: hats arrive, then a kick roll
            for i in range(0, 16, 2):
                hit("hats", f"hat{int(rng.integers(3))}", i, 0.32 + 0.04 * lb, pan=0.25 if i % 4 else -0.15)
            if lb == 3:
                for st in (0, 4, 8, 12):
                    hit("kick", "kick", st, 0.5 + 0.12 * (st // 4), kick=True)
        if bar in (7, 11):                                                # a snare run into the next section
            for st in range(8, 16):
                hit("snare", "snare1", st, 0.3 + 0.1 * (st - 8))

    # Crashes on section starts, reversed swells into them, and a riser into the groove.
    for b in (4, 8, 12):
        s.put("fx", drums["crash"], s.at(b), gain=0.75, send=0.2)
    s.put("fx", drums["crash"], s.at(0), gain=0.45, send=0.2)             # a soft downbeat as the loop comes round
    for b in (4, 8, 12, BARS):
        s.put("fx", drums["rev_crash"][-int(1.1 * SR):], s.at(b) - int(1.1 * SR), gain=0.5, send=0.2)
    s.put("fx", riser(2 * bar_t, rng), s.at(2), gain=0.55, send=0.3)

    return render_mix(s, STEMS, dict(
        rt60=1.7, rev_gain=0.55, delay_steps=3, delay_fb=0.42, delay_taps=6, delay_gain=0.38,
        pad_filter=(650.0, 3600.0, [(0, 0.0), (4, 0.55), (8, 0.7), (12, 0.9), (16, 0.0)]),   # the pad brightens, then resets
        sc=dict(pad=(0.55, 0.14), bass=(0.55, 0.12), arp=(0.35, 0.12), arp2=(0.35, 0.12), stab=(0.3, 0.1),
                lead=(0.12, 0.12), send=(0.3, 0.15)),                                           # duck these on every kick
    )), s.n, s


if __name__ == "__main__":
    render_track("stage1-short", build, STEMS, out_dir=os.path.join(os.path.dirname(__file__), "..", "out"),
                 target_lufs=TARGET_LUFS)

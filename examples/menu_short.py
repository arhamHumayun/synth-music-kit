"""Main menu, the short version: the hero's tune slow and airy (8 bars, 27 seconds, 70 BPM, D minor).

The same hook as the arcade stage (see hero_motif.py and stage1_short.py), at exactly half the tempo, on bells and a soft
lead instead of a saw lead, over a pad and a low drone instead of drums and bass. That is the whole idea of a shared motif.
  bars 0-3   the hook on bells over the pad, a pedal drone and a quarter-note bell arpeggio
  bars 4-7   the hook again on a soft lead, harmonised a third up, with eighth-note bells, a heartbeat kick and whisper hats;
             the last bar strips back so the loop returns to the opening

Run it with:  python examples/menu_short.py     (about 10 seconds)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hero_motif import MAIN, MOTIF  # noqa: E402
from synthkit.musiclib import (  # noqa: E402
    SR, Song, arp_tones, bass_root, bell_note, build_chord_map, chord_runs, hz, make_drums, noise_swell, pad_chord,
    place_motif, render_mix, riser, soft_lead_note, sub_note, taxis, voice_lead,
)
from synthkit.render import render_track  # noqa: E402

BPM = 70                                              # half of the arcade stage's 140
BARS = 8
LAST_BAR = BARS - 1
TARGET_LUFS = -17.5                                   # the menu sits a little under the run music
SEED = 70

SECTIONS = [("a", 0, MAIN), ("b", 4, MAIN)]
BELL_ARP_A, BELL_ARP_B = [0, 2, 3, 2], [0, 2, 3, 2, 4, 3, 2, 1]   # indices into arp_tones()
SPARKLES_PER_BAR = 2                                  # random high chord tones in the airy first half

# Mix gains (linear) per instrument.
STEMS = dict(kick=0.8, snare=0.5, hats=0.25, bass=0.42, pad=0.62, drone=0.17, wind=0.14, bells=0.5, sparkle=0.28,
             lead=0.55, fx=0.28)


def build():
    s = Song(BPM, BARS, SEED)
    rng = s.rng
    drums = make_drums(rng)
    chord_at, sect_at = build_chord_map(SECTIONS)
    bar_t = 16 * s.sps / SR

    # A pedal drone on D: it fits every chord of the progression. Whole cycles per loop keep the seam silent.
    t = taxis(s.n)
    breathe = 0.5 + 0.5 * np.sin(2 * np.pi * t / (BARS * bar_t) - np.pi / 2)
    low, high = s.loop_freq(hz(38)), s.loop_freq(hz(50))
    drone = np.sin(2 * np.pi * low * t) * 0.8 + np.sin(2 * np.pi * high * t) * (0.18 + 0.2 * breathe)
    s.drone("drone", np.stack([drone, drone]), gain=1.0, send=0.15)
    # Wind: filtered noise that swells once over the loop.
    wind_env = (0.5 - 0.5 * np.cos(2 * np.pi * t / (BARS * bar_t))) ** 1.5
    s.drone("wind", noise_swell(s.n, rng) * wind_env, gain=1.0, send=0.5)

    # Pads: one voice-led chord per bar, wide and slow.
    prev = None
    for b, nb, name in chord_runs(chord_at, BARS):
        v = voice_lead(prev, name)
        prev = v
        freqs = [hz(m) for m in v] + [hz(v[2] + 12)]
        s.put("pad", pad_chord(freqs, nb * bar_t, 0.9, 1.6, rng, detune=(-13, 0, 13), width=0.8), s.at(b), gain=1.0, send=0.5)

    for bar in range(BARS):
        sect, _lb = sect_at[bar]
        ch = chord_at[bar]
        tones = arp_tones(ch)
        last = bar == LAST_BAR
        if sect == "b" and not last:                                       # a long soft bass once the pulse arrives
            s.put("bass", sub_note(hz(bass_root(ch)), bar_t * 0.95, a=0.08, r=0.4), s.at(bar), gain=0.8)
        steps, pat, gain = (range(0, 16, 4), BELL_ARP_A, 0.55) if sect == "a" else (range(0, 16, 2), BELL_ARP_B, 0.7)
        if last:
            steps, gain = range(0, 16, 4), 0.5
        for j, i in enumerate(steps):                                      # the bell arpeggio
            m = tones[pat[j % len(pat)]] + 12
            s.put("bells", bell_note(hz(m), ring=2.4, bright=1.9), s.at(bar, i), gain=gain * (1.0 if i % 4 == 0 else 0.7),
                  pan=-0.35 if j % 2 else 0.35, send=0.6)
        if sect == "a":                                                    # scattered high sparkles
            for _ in range(SPARKLES_PER_BAR):
                m = tones[int(rng.integers(3))] + 24
                s.put("sparkle", bell_note(hz(m), ring=3.0, bright=1.2), s.at(bar, int(rng.integers(0, 16))), gain=0.35,
                      pan=float(rng.uniform(-0.8, 0.8)), send=0.8)
        elif not last:                                                     # heartbeat kick on 1 and 3, a clap and whisper hats
            for st in (0, 8):
                pos = s.at(bar, st)
                s.kicks.append(pos)
                s.put("kick", drums["kick"], pos, gain=0.55 if st == 0 else 0.4)
            s.put("snare", drums["clap"], s.at(bar, 8), gain=0.3, send=0.5)
            for i in range(0, 16, 2):
                s.put("hats", drums[f"hat{int(rng.integers(3))}"], s.at(bar, i), gain=0.32 if i % 4 == 0 else 0.2,
                      pan=0.3 if i % 4 else -0.2)

    # The shared hook: bells first, then the soft lead with a harmony a third above.
    place_motif(s, "bells", MOTIF, 0, 1, 0.8, 0.6, lambda f, d: bell_note(f, ring=min(3.4, d + 1.6), bright=2.0))
    place_motif(s, "lead", MOTIF, 4, 1, 0.95, 0.5, lambda f, d: soft_lead_note(f, d), harmony=True, hgain=0.5)

    s.put("fx", riser(2 * bar_t, rng, f0=140, f1=1800), s.at(2), gain=0.5, send=0.5)
    s.put("fx", drums["crash"], s.at(4), gain=0.45, send=0.5)
    s.put("fx", drums["rev_crash"][-int(2.4 * SR):], s.at(4) - int(2.4 * SR), gain=0.4, send=0.5)

    return render_mix(s, STEMS, dict(
        rt60=3.6, rev_gain=0.8, delay_steps=3, delay_fb=0.5, delay_taps=6, delay_gain=0.35,
        pad_filter=(520.0, 2400.0, [(0, 0.25), (4, 0.7), (8, 0.25)]),      # the pad opens for the second half, then closes
        sc=dict(pad=(0.28, 0.4), bells=(0.1, 0.3), send=(0.15, 0.4)),
    )), s.n, s


if __name__ == "__main__":
    render_track("menu-short", build, STEMS, out_dir=os.path.join(os.path.dirname(__file__), "..", "out"),
                 target_lufs=TARGET_LUFS)

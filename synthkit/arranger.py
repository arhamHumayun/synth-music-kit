"""Arranger: reusable per-bar layers for building a full track.

Wraps a `Song` + the drum kit and offers the building blocks a track is made of: chord-following string
ostinatos, pads, bass lines, guitars, brass stabs, choir, drum grooves, fills and section accents. Every
layer takes `span=(lo, hi)` (16th-note steps within the bar) so a chord can change every half bar when a
tune is played at double speed. Chords are (pitch_class, quality) and `t` transposes everything by semitones.
"""

from __future__ import annotations

import numpy as np

from synthkit.heavylib import (
    brass_note, choir_note, growl_bass, hit, make_heavy_kit, near_pitch, parse_riff, place_gtr, place_roll,
    pulse_lead, screamer, string_ensemble,
)
from synthkit.theory import off_scale
from synthkit.musiclib import SR, Song, bell_note, hz, lead_note, render_mix, riser, sub_note
from synthkit.orchlib import (
    flute_note, glock_note, harp_note, organ_note, piano_note, pizz_note, timpani_note, tremolo_strings,
)

Chord = tuple  # (pitch class 0-11, quality)

QUALITY = {
    "min": (0, 3, 7), "maj": (0, 4, 7), "pow": (0, 7, 12), "dim": (0, 3, 6), "sus2": (0, 2, 7),
    "min7": (0, 3, 7, 10), "maj7": (0, 4, 7, 11), "dom7": (0, 4, 7, 10),
    "min9": (0, 3, 7, 10, 14), "maj9": (0, 4, 7, 11, 14),
}
# The string-ostinato shape from Tyrant's Overture: indices into [root, 3rd, 5th, octave].
OSTINATO = [0, 2, 3, 2, 0, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]
# The arcade run theme's arpeggio shape (same four tones).
ARCADE_ARP = [0, 2, 3, 2, 0, 2, 3, 2, 1, 2, 4, 2, 1, 2, 3, 2]
GALLOP = "0+ _ 0 0 0 _ 0 0 0+ _ 0 0 0 _ 0 0"
FULL = (0, 16)


def chord_midis(chord: Chord, t: int, target: int) -> list[int]:
    """Chord tones stacked upward from the root nearest `target` (after transposing by `t`)."""
    root = near_pitch((chord[0] + t) % 12, target)
    return [root + i for i in QUALITY[chord[1]]]


def chord_pcs(chord: Chord, t: int = 0) -> set[int]:
    return {(chord[0] + t + i) % 12 for i in QUALITY[chord[1]]}


class Arranger:
    def __init__(self, bpm: int, bars: int, seed: int):
        self.s = Song(bpm, bars, seed)
        self.rng = self.s.rng
        self.kit = make_heavy_kit(self.rng)
        self.bar_t = 16 * self.s.sps / SR
        self.step_t = self.s.sps / SR
        self._voices: dict = {}
        self._table = {
            "strings": lambda f, d: string_ensemble(f, d, True),
            "strings_sus": lambda f, d: string_ensemble(f, d, False, bright=3000.0),
            "tremolo": lambda f, d: tremolo_strings(f, d),
            "pizz": lambda f, d: pizz_note(f, min(d, 0.3)),
            "piano": lambda f, d: piano_note(f, d),
            "harp": lambda f, d: harp_note(f),
            "organ": lambda f, d: organ_note(f, d),
            "flute": lambda f, d: flute_note(f, d),
            "brass": lambda f, d: brass_note(f, d),
            "pulse": lambda f, d: pulse_lead(f, d, duty=0.28),
            "lead": lambda f, d: lead_note(f, d, rel=0.25, vib=0.005, bright=5000.0),
            "screamer": lambda f, d: screamer(f, d, pre=2.0),
            "glock": lambda f, d: glock_note(f),
            "bell": lambda f, d: bell_note(f, ring=min(3.4, d + 1.6), bright=1.9),
        }

    # ------------------------------------------------------------------ voices
    def voice(self, name: str):
        """Memoised voice: (hz, seconds) -> samples. Riffs repeat the same notes thousands of times."""
        fn = self._table[name]

        def cached(f: float, d: float) -> np.ndarray:
            key = (name, round(f, 2), round(d, 3))
            if key not in self._voices:
                self._voices[key] = fn(f, d)
            return self._voices[key]

        return cached

    # ------------------------------------------------------------------ melody
    def melody(self, stem: str, notes: list, bar0: int, voice: str, t: int = 0, gain: float = 1.0, pan: float = 0.0,
               send: float = 0.3, legato: float = 0.96, octave: int = 0, scale: tuple | None = None,
               allowed: tuple = ()) -> None:
        """Place [(bar_offset, step, midi, length)] with `voice`. If `scale` (pitch classes of the *written* key)
        is given, every note must fit it (plus `allowed` extra pitch classes) after transposing by `t`."""
        notes = [(b, st, m + t + 12 * octave, ln) for b, st, m, ln in notes]
        if scale is not None:
            pcs = tuple((p + t) % 12 for p in scale)
            extra = tuple((p + t) % 12 for p in allowed)
            bad = off_scale(notes, pcs, extra)
            assert not bad, f"{stem}/{voice} at bar {bar0}: notes outside the key: {bad[:4]}"
        fn = self.voice(voice)
        for b, st, m, ln in notes:
            self.s.put(stem, fn(hz(m), max(0.05, ln * self.step_t * legato)), self.s.at(bar0 + b, st),
                       gain=gain, pan=pan, send=send)

    # ------------------------------------------------------------------ layers
    def ostinato(self, bar: int, chord: Chord, t: int = 0, voice: str = "strings", gain: float = 0.8,
                 stem: str = "strings", pattern=OSTINATO, span=FULL, stride: int = 1, target: int = 57,
                 send: float = 0.2, octave: int = 0) -> None:
        tones = chord_midis(chord, t, target)[:3]
        tones = tones + [tones[0] + 12, tones[1] + 12]   # [root, 3rd, 5th, root+8ve, 3rd+8ve]
        fn = self.voice(voice)
        for i in range(span[0], span[1], stride):
            m = tones[pattern[i]] + 12 * octave
            self.s.put(stem, fn(hz(m), self.step_t * 0.85), self.s.at(bar, i),
                       gain=gain * (1.0 if i % 4 == 0 else 0.72), pan=0.25 if i % 2 else -0.25, send=send)

    def pad(self, bar: int, chord: Chord, t: int = 0, voice: str = "strings_sus", gain: float = 0.7,
            stem: str = "pad", span=FULL, nbars: int = 1, target: int = 48, send: float = 0.35, tones: int = 4) -> None:
        dur = (span[1] - span[0]) * self.step_t + (nbars - 1) * self.bar_t
        fn = self.voice(voice)
        for m in chord_midis(chord, t, target)[:tones]:
            self.s.put(stem, fn(hz(m), dur * 0.98), self.s.at(bar, span[0]), gain=gain, send=send)

    def bass(self, bar: int, chord: Chord, t: int = 0, style: str = "8th", gain: float = 1.0, span=FULL) -> None:
        root = near_pitch((chord[0] + t) % 12, 33)
        lo, hi = span
        s = self.s
        if style == "8th":
            for st in range(lo, hi, 2):
                off = 12 if st % 8 == 6 else 0
                s.put("bass", growl_bass(hz(root + off), self.step_t * 1.8), s.at(bar, st), gain=gain)
        elif style == "pulse":
            for st in range(lo, hi):
                off = 12 if st % 4 == 2 else 0
                s.put("bass", growl_bass(hz(root + off), self.step_t * 0.85), s.at(bar, st), gain=gain * 0.9)
        # every style carries a sustained sub underneath
        s.put("sub", sub_note(hz(root), (hi - lo) * self.step_t * 0.98, a=0.02, r=0.12), s.at(bar, lo), gain=gain)

    def gtr(self, bar: int, chord: Chord, t: int = 0, style: str = "gallop", gain: float = 1.0, span=FULL,
            lock_kick: bool = False, send: float = 0.05, target: int = 45) -> list:
        root = near_pitch((chord[0] + t) % 12, target)
        lo, hi = span
        if style == "power":
            place_gtr(self.s, bar, lo, root, hi - lo, kind="sustain", gain=gain, send=send, intervals=(0, 7))
            return []
        spec = GALLOP if style == "gallop" else "0+ . 0 . 0 . 0 . 0+ . 0 . 0 . 0 ."
        notes = [n for n in parse_riff(spec) if lo <= n[0] < hi]
        for st, semi, ln, pw in notes:
            place_gtr(self.s, bar, st, root + semi, ln, kind="sustain" if ln >= 5 else "chug", gain=gain, send=send,
                      intervals=(0, 7) if pw else (0,))
            if lock_kick:
                hit(self.s, "kick", self.kit["kick"], bar, st, 0.85 if st % 4 else 1.0, kick=True, rng=self.rng)
        return notes

    def stabs(self, bar: int, chord: Chord, t: int = 0, voice: str = "brass", gain: float = 0.6, steps=(0, 6, 10),
              stem: str = "brass", target: int = 53, span=FULL, send: float = 0.3) -> None:
        mids = chord_midis(chord, t, target)[:3]
        fn = self.voice(voice)
        for st in steps:
            if span[0] <= st < span[1]:
                for m in mids:
                    self.s.put(stem, fn(hz(m), self.step_t * 1.6), self.s.at(bar, st), gain=gain, pan=-0.15, send=send)

    def choir(self, bar: int, chord: Chord, t: int = 0, gain: float = 0.8, vowel: str = "ah", nbars: int = 1,
              span=FULL, target: int = 52, send: float = 0.5) -> None:
        dur = (span[1] - span[0]) * self.step_t + (nbars - 1) * self.bar_t
        for k, m in enumerate(chord_midis(chord, t, target)[:3]):
            key = ("choir", vowel, round(hz(m), 2), round(dur, 3), k)
            if key not in self._voices:
                self._voices[key] = choir_note(hz(m), dur, vowel, rel=0.9, a=0.5, seed=k)
            self.s.put("choir", self._voices[key], self.s.at(bar, span[0]), gain=gain, send=send)

    def layers(self, bar: int, chord: Chord, t: int = 0, span=FULL, ost=None, pad=None, bass=None, gtr=None,
               stabs=None, choir=None) -> None:
        """Place the standard layers for one bar (or half bar); each argument is a dict of that layer's options."""
        if ost is not None:
            self.ostinato(bar, chord, t, span=span, **ost)
        if pad is not None:
            self.pad(bar, chord, t, span=span, **pad)
        if bass is not None:
            self.bass(bar, chord, t, span=span, **bass)
        if gtr is not None:
            self.gtr(bar, chord, t, span=span, **gtr)
        if stabs is not None:
            self.stabs(bar, chord, t, span=span, **stabs)
        if choir is not None:
            self.choir(bar, chord, t, span=span, **choir)

    # ------------------------------------------------------------------ drums
    def drums(self, bar: int, style: str, level: float = 1.0, span=FULL) -> None:
        k, s, rng = self.kit, self.s, self.rng
        lo, hi = span
        gallop_steps = (0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15)
        kicks, snares, hats = (), (4, 12), "8th"
        if style == "rock":
            kicks = (0, 6, 10)
        elif style == "drive":
            kicks = (0, 4, 8, 12)
        elif style == "double":
            kicks = gallop_steps
        elif style == "half":
            kicks, snares, hats = (0, 10), (8,), "quarter"
        elif style == "march":
            kicks, snares, hats = (0, 8), (), "none"
        elif style == "none":
            kicks, snares, hats = (), (), "none"
        for st in kicks:
            if lo <= st < hi:
                big = style == "half" and st == 0
                hit(s, "kick", k["boom_kick" if big else "kick"], bar, st, level * (1.0 if st % 4 == 0 else 0.85),
                    kick=True, rng=rng)
        for st in snares:
            if lo <= st < hi:
                hit(s, "snare", k[f"snare{int(rng.integers(3))}"], bar, st, level, send=0.25, rng=rng)
        if style == "march":
            for st, tom in ((4, "tom_mid"), (12, "tom_lo")):
                if lo <= st < hi:
                    hit(s, "snare", k[tom], bar, st, level * 0.8, send=0.4, rng=rng)
        if hats != "none":
            for st in range(lo, hi, 4 if hats == "quarter" else 2):
                name = "open_hat" if (hats == "8th" and st % 8 == 6) else f"hat{int(rng.integers(3))}"
                hit(s, "hats", k[name], bar, st, level * (0.5 if st % 4 == 0 else 0.3), pan=0.3 if st % 4 else -0.2, rng=rng)

    def fill(self, bar: int, kind: str, level: float = 1.0) -> None:
        k, rng = self.kit, self.rng
        if kind == "snare4":
            place_roll(self.s, bar, 12.0, 16.0, k["snare1"], 1.0, 0.35 * level, level, rng=rng)
        elif kind == "snare8":
            place_roll(self.s, bar, 8.0, 16.0, k["snare1"], 1.0, 0.3 * level, level, rng=rng)
        elif kind == "roll":
            place_roll(self.s, bar, 0.0, 16.0, k["snare2"], 0.5, 0.2 * level, level, rng=rng)
        elif kind == "toms":
            for j, tom in enumerate(("tom_hi", "tom_mid", "tom_mid", "tom_lo")):
                hit(self.s, "snare", k[tom], bar, 12 + j, level * (0.7 + 0.1 * j), send=0.3, rng=rng)
        elif kind == "tomroll":
            place_roll(self.s, bar, 4.0, 16.0, k["tom_mid"], 2.0, 0.3 * level, level, rng=rng)

    def accent(self, bar: int, crash: float = 0.8, boom: float = 0.0, rev: bool = True, china: bool = False) -> None:
        s, k = self.s, self.kit
        if crash:
            hit(s, "fx", k["china" if china else "crash"], bar, 0, crash, send=0.2)
        if boom:
            hit(s, "fx", k["sub_boom"], bar, 0, boom, send=0.2)
        pos = s.at(bar) - int(1.3 * SR)
        if rev and pos >= 0:
            s.put("fx", k["rev_crash"][-int(1.3 * SR):], pos, gain=0.5, send=0.2)

    def timp(self, bar: int, step: float, note: int, gain: float = 0.9) -> None:
        self.s.put("timp", timpani_note(hz(note)), self.s.at(bar, step), gain=gain, send=0.4)

    def swell(self, bar: int, nbars: float, lo: float = 160.0, hi: float = 3200.0, gain: float = 0.6) -> None:
        self.s.put("fx", riser(nbars * self.bar_t, self.rng, lo, hi), self.s.at(bar), gain=gain, send=0.3)

    def orch_hit(self, bar: int, step: float, chord: Chord, t: int = 0, gain: float = 1.0, beats: float = 1.0) -> None:
        """Tutti stab: brass + strings + timpani + boom, the 'everything lands at once' punctuation."""
        mids = chord_midis(chord, t, 48)[:3]
        mids = mids + [mids[0] + 12]
        dur = self.step_t * 4 * beats
        for m in mids:
            self.s.put("brass", self.voice("brass")(hz(m), dur), self.s.at(bar, step), gain=gain, send=0.35)
            self.s.put("strings", self.voice("strings")(hz(m + 12), min(dur, self.step_t * 3)), self.s.at(bar, step), gain=gain * 0.8, send=0.3)
        self.timp(bar, step, mids[0] - 12, gain)
        hit(self.s, "fx", self.kit["boom_kick"], bar, step, gain * 0.9, send=0.2)

    # ------------------------------------------------------------------ output
    def finish(self, stem_gains: dict, cfg: dict):
        return render_mix(self.s, stem_gains, cfg), self.s.n, self.s

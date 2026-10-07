---
name: add-instrument
description: Use when a track needs a sound the kit does not have yet. Covers writing a synthesized instrument function that fits the library's conventions.
---

# Add an instrument

1. **Check what exists** in `synthkit/musiclib.py`, `heavylib.py`, `orchlib.py` and `machinelib.py`. Reuse the building blocks (`adsr`, `lp`, `hp`, `bp`, `drive`, `subtractive_saw`, `harm_count`, `peak_norm`) before writing new math.
2. **Signature:** `def my_voice(f: float, dur: float, ...) -> np.ndarray`, returning a mono array a little longer than `dur` (a release tail). Drum and one-shot sounds take a seeded `rng` instead of a duration.
3. **Keep it deterministic:** seed any noise from the pitch (`np.random.default_rng(int(f * 10))`), so renders are reproducible.
4. **Keep it band-limited:** build harmonics only up to `harm_count(f, cap)` so high notes do not alias into harsh noise.
5. **For a real instrument, model how it makes sound** (see `synthkit/acoustic.py`) instead of filtering an oscillator, then verify with numbers you can compute: pitch within a few cents (FFT peak), decay time, brightness (spectral centroid), steady level for bowed or blown notes (`tests/test_acoustic.py` has the pattern). Two detuned oscillators on a solo voice produce a slow level wobble.
6. **Shape it:** every sound needs an attack, a decay or sustain, and a release. A raw oscillator sounds like a test tone.
7. **Wire it in:** add it per-track with `A._table["name"] = lambda f, d: my_voice(f, d)`, or to the voice table in `Arranger.__init__` if it is generally useful. Then call it with `A.melody(stem, notes, bar, "name", ...)`.
8. **Check it** by rendering a short loop with a single note in it, and look at the stem level in the report. Ask the user to listen and describe what they hear; adjust the filter cutoff, decay and drive in small steps.

Give it a short, honest docstring that says what the sound is meant to be, so a later reader can tell what to tune.

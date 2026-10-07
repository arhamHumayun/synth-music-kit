# AGENTS.md: synth-music-kit

Python library for writing game music as code. Notes are data, instruments are functions, and the renderer reports its own mix so you can work without hearing anything. `README.md` has the quick start.

## Conventions

1. **A song is a module** with a `build()` that returns `Arranger.finish(...)` (or `(mix, n_samples, song)`), a `STEMS` dict of per-instrument gains, and optionally `TARGETS` (RMS dBFS goals). Render it with `synthkit.render.render_track`. Copy `examples/hello_loop.py` to start.
2. **Notes are tuples** `(bar_offset, step, midi, length)`, 16 steps per bar. Keep tunes in named tables at the top of the file, not inline in logic.
3. **Check keys in code.** Pass `scale=` to `Arranger.melody` so every note is verified; allow extra pitch classes with `allowed=`.
4. **Name every tuned number.** Gains, durations and thresholds are `UPPER_CASE` constants with a comment, so tuning is searchable.
5. **BPM must divide 661500** (a 16th note is a whole number of samples). `Song` asserts it.
6. **Instruments** take `(frequency_hz, duration_s)` and return a mono numpy array. Keep them deterministic (seed `np.random.default_rng` from the pitch) and band-limited. Cache repeated notes; riffs repeat the same note thousands of times.
7. **Do not add samples.** Everything is synthesized, which is what keeps the output original and license-free.
8. **For acoustic sounds, model the instrument, not the waveform** (`synthkit/acoustic.py`): a plucked string is a delay line with a lossy loop, a piano is stiff-string partials, a bowed string is a saw with a slow attack and vibrato. Add human feel in the song: timing jitter, varied loudness, rolled strums. Beating detuned oscillators sound like a tremolo on a solo instrument, so only use them for ensembles.
9. **Verify by the report**, not by guessing: read the stem levels, the loop seam line, and the LUFS line after each render. You cannot hear the result, so say so, and ask the user to listen and describe problems.

## Skills

`.claude/skills/compose-track`, `read-the-render`, `add-instrument`.

---
name: read-the-render
description: Use after rendering a track with this kit, to interpret the report (stem levels, loop seam, loudness) and decide what to change without hearing the audio.
---

# Read the render

`render_track` prints three things. Check each after every render.

**Stem levels.** One line per instrument: RMS dBFS over the song, then peak. If you passed `targets`, it also prints the gain that would hit the target. Look for:
- A stem missing from the list: it never played (a wrong stem name or an empty section).
- One stem 8 dB or more above its neighbours: it will bury the rest. Lower its gain in `STEMS`.
- A melody stem far below the accompaniment: raise it. Lead and alarm-style voices are the usual culprits for being too loud or too quiet.
- Drums: kick near -18, snare near -23, hats near -38 is a sane starting balance.

Change gains in the track's `STEMS` dict (linear, 1.0 = unchanged), then re-render. Do not chase exact numbers; they are a guide and the user's ears decide.

**Loop seam.** `loop seam step X vs 99.9th-pct sample step Y`. X (the jump from the last sample to the first) should be well below Y. If X is bigger, something is cut off at the loop point: a note or tail that was not folded back, or a long note that ends abruptly on the last bar.

**Loudness.** The mastered file's LUFS and true peak. Keep every track in one game at the same target (this kit's examples use -15.5 LUFS) so tracks do not jump in volume. True peak should stay under about -1 dBFS.

Also read any assertion errors. `notes outside the key` means a melody note is not in the declared scale: fix the note, or pass it in `allowed=` if it is a deliberate chromatic note. A BPM assertion means the tempo does not divide 661500.

Report the findings to the user in plain language, and say what you changed.

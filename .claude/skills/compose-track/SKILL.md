---
name: compose-track
description: Use when the user wants a new song, theme or loop for their game made with this kit. Covers the concept conversation, writing the track module, and rendering it.
---

# Compose a track

Do these in order. Do not write code before step 2 is settled.

1. **Look first.** Read `README.md` and `examples/hello_loop.py`. If the user has existing tracks, read one so the new one matches.
2. **Concept conversation.** Agree, in a few lines, on: where it plays (menu, level, boss), the feel (relentless, airy, grand, mechanical), tempo and key, how long, and what repeats. Propose a motif: a short idea (a rhythm, an interval shape, a few notes) that the whole track is built from. Prefer an idea that survives being transposed or re-orchestrated, since games reuse themes across levels. Give a recommendation, not a menu. If the game already has themes, make this one clearly distinct from them (different key, rhythm, instrument family).
3. **Structure.** Plan sections in bars: intro, groove, breakdown, main, climax, loop-in. A loop's last bars should lead back into its first.
4. **Write a new track file** (copy `examples/hello_loop.py` into e.g. `tracks/<name>.py`, keeping its `sys.path` line pointing at the repo root). Put the tune, chords and riff in named tables at the top. Pass `scale=` to every melody call. Use `Arranger` layers per bar; vary density between sections so it builds.
5. **Pick BPM from 90, 100, 105, 126, 135, 140, 147, 150** (must divide 661500).
6. **Render** and run the `read-the-render` skill on the output. Fix problems it finds, then re-render.
7. **Hand it to the user to listen.** You cannot hear the audio. Say that plainly, summarise what you built bar by bar, and ask them to describe what to change (too loud, shrill, empty, busy). Adjust `STEMS` gains and re-render; it takes seconds.

Never claim the music sounds good. Report what you measured and what you chose.

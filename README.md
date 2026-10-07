# synth-music-kit

Write game music as Python and render it to a seamlessly looping `.ogg`. Every sound is synthesized from math (no samples), so the music is original and safe to ship. Built to be driven by a coding agent such as Claude Code: the repo includes skills that teach the agent how to compose with it.

Made for a twin-stick shooter. `examples/stage1_short.py` and `examples/menu_short.py` are short versions of that game's stage 1 and menu music.

## Quick start

You need Python 3.10+, `numpy`, `scipy`, and [ffmpeg](https://ffmpeg.org/download.html) on your PATH. ffmpeg is only needed for `.ogg` output and loudness measurement; without it you get a `.wav`.

```
pip install -r requirements.txt
python examples/hello_loop.py          # 8 bars, 19 seconds -> out/hello-loop.ogg
python examples/stage1_short.py        # the stage 1 theme in 16 bars, 27 seconds
python examples/menu_short.py          # the same tune slow and airy: the main menu, 8 bars, 27 seconds
python examples/porch_light.py         # an acoustic folk piece from physical-model instruments, 24 bars, 64 seconds
```

## Using it with an agent

Open this folder in Claude Code and describe the music you want. Skills in `.claude/skills/` cover composing a track, reading the render report, and adding instruments. `AGENTS.md` describes the layout. Start with a conversation about the idea (feel, tempo, key, what should repeat) before asking for code.

## How a song is written

A song is data plus a few layer calls:

```python
A = Arranger(100, 8, seed=1)                       # 100 BPM, 8 bars
A.pad(bar, (9, "min"), voice="strings_sus")        # a chord is (pitch class, quality); A = 9
A.bass(bar, chord, style="8th")
A.drums(bar, "rock")
A.melody("lead", TUNE, 0, "lead", scale=A_MINOR)   # notes are (bar, step, midi, length); the scale checks them
return A.finish(STEMS, cfg)                        # mixed audio; `render_track` masters and saves it
```

Time is a grid: a bar has 16 steps (16th notes). Everything is deterministic (seeded), so a render is reproducible.

## The report it prints

`render_track` prints each instrument's level in the mix, with the gain that would hit a target if you pass `targets`. It also checks that the loop seam has no click and measures loudness (LUFS). That is how an agent, which cannot hear, can still tell when something is too loud, missing, or clicking. You are the ears: listen, then say what to change.

## Layout

| File | What it is |
|------|-----------|
| `synthkit/musiclib.py` | Theory helpers, oscillators, filters, envelopes, basic instruments, drums, reverb, delay, sidechain, the `Song` grid, mastering |
| `synthkit/heavylib.py` | Distorted guitar, growl bass, strings, brass, choir, a metal drum kit |
| `synthkit/orchlib.py` | Piano, organ, flute, harp, pizzicato, tremolo strings, timpani, glockenspiel |
| `synthkit/acoustic.py` | Physical-model instruments: fingerpicked guitar, upright bass, piano, cello, flute, and a soft acoustic drum kit |
| `synthkit/machinelib.py` | Klaxon, gong, bowed-metal drone, tin clank, bit-crusher |
| `synthkit/arranger.py` | Per-bar layers: ostinato, pads, bass, guitar, stabs, choir, drums, fills, accents |
| `synthkit/theory.py` | Scales and motif transformations (faster, slower, inverted, harmonised, transposed) |
| `synthkit/render.py` | `render_track`: report, master, write the file |
| `examples/` | `hello_loop.py` (start here), `stage1_short.py` and `menu_short.py` (two tracks sharing one motif from `hero_motif.py`), `porch_light.py` (acoustic folk, shows the physical-model instruments and human timing) |

## Rules to know

- **Tempo:** a 16th note must be a whole number of samples, so BPM must divide 661500. Good values: 90, 100, 105, 126, 135, 140, 147, 150. `Song` asserts this.
- **Loops:** every note and reverb tail is rendered past the end and folded back onto the start, so files loop with no gap or click.
- **No samples:** instruments are additive, subtractive, modal or physical-model synthesis. Drums are shaped noise and sine bodies.
- **Realistic is relative:** the plucked strings and piano are the most convincing, the bowed cello and flute the least. The numbers are tested (`tests/test_acoustic.py`), the sound is not: listen and tell the agent what to change.

## Run the tests

```
pip install -r requirements-dev.txt
pytest
```

## License

MIT. The music you render is yours.

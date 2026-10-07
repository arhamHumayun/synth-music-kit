"""Render driver: build a track, report its mix, master it and write a looping Ogg Vorbis (or WAV).

A track is any function returning `(mix, n_samples, song)`; `Arranger.finish(...)` returns exactly that. `render_track`
prints each stem's level (and the gain that would hit your target), checks the loop seam, masters to a loudness target
and writes the file. Run an example with `python examples/hello_loop.py`.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
from scipy.io import wavfile

from synthkit.musiclib import SR, fold_tail, master, measure, seam_report, stem_report, to_ogg


def render_track(name: str, builder, stem_gain: dict[str, float], out_dir: str = "out", target_lufs: float = -15.5,
                 targets: dict[str, float] | None = None, stems_dir: str | None = None, ogg: bool = True) -> str:
    """Render `builder()` to `<out_dir>/<name>.ogg` (or `.wav` if ffmpeg is missing / `ogg=False`); returns the path."""
    print(f"Rendering {name} ...")
    mix, n, song = builder()
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
        for k, v in song.stems.items():
            wavfile.write(os.path.join(stems_dir, f"{name}_{k}.wav"), SR, (fold_tail(v, n).T * 0.5).astype(np.float32))
    stem_report(song.stems, n, stem_gain, targets)
    y = master(mix, n, target_lufs)
    seam_report(y, name)
    os.makedirs(out_dir, exist_ok=True)
    use_ogg = ogg and shutil.which("ffmpeg") is not None
    if ogg and not use_ogg:
        print("  ffmpeg not found on PATH: writing a WAV instead")
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, f"{name}.wav")
        wavfile.write(wav, SR, (np.clip(y, -1, 1).T * 32767).astype(np.int16))
        lufs, peak = measure(wav) if shutil.which("ffmpeg") else (float("nan"), float("nan"))
        path = os.path.join(out_dir, f"{name}.{'ogg' if use_ogg else 'wav'}")
        if use_ogg:
            to_ogg(wav, path)
        else:
            shutil.copy(wav, path)
    print(f"  {name}: {n / SR:.2f}s, {lufs:.1f} LUFS, true peak {peak:.1f} dBFS -> {path}")
    return path

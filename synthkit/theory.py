"""Music-theory helpers: notes as data, scales, and the transformations that develop a motif.

A note is (bar_offset, step, midi, length_in_16ths). A tune is a list of notes. Transformations return a new list in
the same format (steps may be fractional when a tune is sped up), so one short motif can come back faster, slower,
upside down, harmonised or in another key. `off_scale` is how an arrangement checks every note against its key.
"""

from __future__ import annotations

Note = tuple  # (bar_offset, step, midi, length)

# Scales as pitch-class sets (0 = C). Put your own next to these.
D_MINOR = (2, 4, 5, 7, 9, 10, 0)
A_MINOR = (9, 11, 0, 2, 4, 5, 7)
E_MINOR = (4, 6, 7, 9, 11, 0, 2)
F_MINOR = (5, 7, 8, 10, 0, 1, 3)
E_PHRYGIAN = (4, 5, 7, 9, 11, 0, 2)


def from_abs(rows: list[tuple]) -> list[Note]:
    """[(absolute_step, midi, length)] -> [(bar_offset, step, midi, length)] sorted by time."""
    out = []
    for a, m, ln in sorted(rows, key=lambda r: (r[0], r[1])):
        out.append((int(a // 16), a - 16 * int(a // 16), m, ln))
    return out


def to_abs(notes: list[Note]) -> list[tuple]:
    return [(b * 16 + st, m, ln) for b, st, m, ln in notes]


def scale_notes(pcs: tuple, lo: int = 24, hi: int = 120) -> list[int]:
    return [m for m in range(lo, hi) if m % 12 in pcs]


def snap_scale(m: int, pcs: tuple) -> int:
    """Nearest scale tone to `m` (ties go down)."""
    best = min(scale_notes(pcs), key=lambda x: (abs(x - m), x))
    return best


def off_scale(notes: list[Note], pcs: tuple, allowed: tuple = ()) -> list[Note]:
    """Notes whose pitch class is outside the scale (and outside `allowed` extra pitch classes)."""
    return [n for n in notes if n[2] % 12 not in pcs and n[2] % 12 not in allowed]


def transpose(notes: list[Note], semis: int) -> list[Note]:
    return [(b, st, m + semis, ln) for b, st, m, ln in notes]


def retime(notes: list[Note], factor: float) -> list[Note]:
    """Scale time by `factor` (0.5 = twice as fast, 2.0 = half speed); pitches untouched."""
    return from_abs([(a * factor, m, ln * factor) for a, m, ln in to_abs(notes)])


def diminute(notes: list[Note], factor: float = 2.0) -> list[Note]:
    """Speed the tune up by `factor`: a 4-bar hook becomes 2 bars."""
    return retime(notes, 1.0 / factor)


def augment(notes: list[Note], factor: float = 2.0) -> list[Note]:
    """Slow the tune down by `factor`: a 4-bar hook becomes 8 bars."""
    return retime(notes, factor)


def invert(notes: list[Note], pivot: int, pcs: tuple | None = None) -> list[Note]:
    """Mirror the melody around `pivot` (contrary motion); snap to a scale if `pcs` is given."""
    out = []
    for b, st, m, ln in notes:
        mirrored = 2 * pivot - m
        out.append((b, st, snap_scale(mirrored, pcs) if pcs else mirrored, ln))
    return out


def harmonize(notes: list[Note], pcs: tuple, degrees: int = 2, min_len: float = 4) -> list[Note]:
    """A second voice `degrees` scale steps above (2 = a third, 5 = a sixth), long notes only."""
    scale = scale_notes(pcs)
    out = []
    for b, st, m, ln in notes:
        if ln < min_len:
            continue
        snapped = snap_scale(m, pcs)
        out.append((b, st, scale[scale.index(snapped) + degrees], ln))
    return out


def fragment(notes: list[Note], bar0: int, nbars: int) -> list[Note]:
    """Just bars [bar0, bar0 + nbars) of the tune, re-based to start at bar 0."""
    return [(b - bar0, st, m, ln) for b, st, m, ln in notes if bar0 <= b < bar0 + nbars]


def repeat(notes: list[Note], times: int, period_bars: int) -> list[Note]:
    out = []
    for r in range(times):
        out.extend((b + r * period_bars, st, m, ln) for b, st, m, ln in notes)
    return out


def snap_to_chords(notes: list[Note], chord_pcs_at, max_move: int = 2, min_len: float = 4) -> list[Note]:
    """Pull long notes to the nearest tone of the chord sounding at their start so a derived counter-line
    (e.g. an inversion) stays consonant. `chord_pcs_at(bar_offset, step)` returns the set of chord pitch classes."""
    out = []
    for b, st, m, ln in notes:
        if ln < min_len:
            out.append((b, st, m, ln))
            continue
        pcs = chord_pcs_at(b, st)
        cands = [m + d for d in range(-max_move, max_move + 1) if (m + d) % 12 in pcs]
        out.append((b, st, min(cands, key=lambda x: (abs(x - m), x)) if cands else m, ln))
    return out


def run(start: int, end: int, pcs: tuple | None, bar: int, step0: float, n_notes: int, note_len: float) -> list[Note]:
    """A scalar (or chromatic if `pcs` is None) run of `n_notes` notes from `start` to `end`, evenly spaced."""
    pool = scale_notes(pcs) if pcs else list(range(0, 128))
    i0 = pool.index(snap_scale(start, pcs)) if pcs else start
    i1 = pool.index(snap_scale(end, pcs)) if pcs else end
    out = []
    for k in range(n_notes):
        idx = round(i0 + (i1 - i0) * k / max(1, n_notes - 1))
        out.append(from_abs([((bar * 16) + step0 + k * note_len, pool[idx], note_len)])[0])
    return out


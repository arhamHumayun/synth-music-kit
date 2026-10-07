"""The hero's tune, written once and shared by every track that uses it.

A motif is just data: a list of notes and the chords it sits on. The arcade stage (stage1_short.py) plays it fast on a saw
lead; the main menu (menu_short.py) plays it slow on bells and a soft lead. Same notes, different tempo, instruments and
mood, which is why a game's music feels like one story. Change a note here and both tracks change.
"""

# The hook, 4 bars over Dm | Bb | F | C: (step in the 64-step phrase, note name, length in 16ths).
MOTIF = [
    (0, "D5", 6), (6, "F5", 2), (8, "A5", 6), (14, "G5", 2),
    (16, "F5", 6), (22, "D5", 2), (24, "Bb4", 4), (28, "D5", 4),
    (32, "A5", 6), (38, "C6", 2), (40, "A5", 4), (44, "F5", 4),
    (48, "G5", 6), (54, "E5", 2), (56, "C5", 4), (60, "E5", 2), (62, "D5", 2),
]
# The chords the hook was written over, one per bar.
MAIN = ["Dm", "Bb", "F", "C"]

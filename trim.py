#!/usr/bin/env python3
"""Build a pf expert-width trim mask from `pf --calib` stats.

usage: python3 trim.py OUT.mask WIDTH STATS [STATS ...]

Each STATS file comes from one `pf --lines --calib STATS -f CORPUS` run; files
are weighted equally, so use one per language/domain you serve (e.g. English,
Japanese, your logs). Keeps the WIDTH highest-scoring of the 640 intermediate
neurons of every expert (WIDTH must be a multiple of 8). Standard library only.
"""
import sys
from array import array

NL, NE, FF = 8, 128, 640


def main(argv):
    if len(argv) < 4:
        sys.exit(__doc__)
    out, width, files = argv[1], int(argv[2]), argv[3:]
    if width % 8 or not 8 <= width <= FF:
        sys.exit("WIDTH must be a multiple of 8 in 8..640")
    score = [0.0] * (NL * NE * FF)
    for path in files:
        a = array("f")
        with open(path, "rb") as f:
            a.frombytes(f.read())
        if len(a) != len(score):
            sys.exit(f"{path}: not a pf --calib stats file")
        for i, v in enumerate(a):
            score[i] += v
    bits = bytearray(NL * NE * FF // 8)
    for x in range(NL * NE):
        s = score[x * FF:(x + 1) * FF]
        for j in sorted(range(FF), key=lambda j: -s[j])[:width]:
            i = x * FF + j
            bits[i >> 3] |= 1 << (i & 7)
    with open(out, "wb") as f:
        f.write(b"PFTRIM1\0" + width.to_bytes(4, "little") + bytes(bits))


if __name__ == "__main__":
    main(sys.argv)

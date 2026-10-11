#!/usr/bin/env python3
"""Fail if any `sorry` remains in FCPA/*.lean or FCPA.lean (comments and strings ignored).

Files listed in CITED.txt (one filename per line) are CITED, not proved: their sorries are reported
separately and only fail the check under --strict.  Exit 0 = clean.
"""
import re, sys, pathlib

root = pathlib.Path(__file__).resolve().parent
strict = "--strict" in sys.argv
cited = set()
cf = root / "CITED.txt"
if cf.exists():
    cited = {l.strip() for l in cf.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}


def strip(src: str) -> str:
    out, i, n, depth = [], 0, len(src), 0
    while i < n:
        if src.startswith("/-", i):
            depth += 1; i += 2; continue
        if depth and src.startswith("-/", i):
            depth -= 1; i += 2; continue
        if depth:
            out.append("\n" if src[i] == "\n" else " "); i += 1; continue
        if src.startswith("--", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src[i] == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
            i += 1; out.append('""'); continue
        out.append(src[i]); i += 1
    return "".join(out)


bad, cit = [], []
for f in sorted(list((root / "FCPA").glob("*.lean")) + [root / "FCPA.lean"]):
    for ln, line in enumerate(strip(f.read_text(encoding="utf-8")).splitlines(), 1):
        if re.search(r"\bsorry\b", line):
            (cit if f.name in cited else bad).append(f"{f.relative_to(root)}:{ln}: {line.strip()}")

for b in bad:
    print("SORRY", b)
for c in cit:
    print("CITED-SORRY", c)
n = len(bad) + (len(cit) if strict else 0)
print(f"check_no_sorry: {len(bad)} sorry, {len(cit)} cited-sorry" + (" (strict)" if strict else ""))
sys.exit(1 if n else 0)

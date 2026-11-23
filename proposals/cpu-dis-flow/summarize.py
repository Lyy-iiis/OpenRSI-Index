"""Summarize a complete split and apply the frozen criteria to its measurements."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

METHODS = ("fast", "medium", "extra-iteration", "full-grid", "adaptive")
METRICS = (("epe_sum", "n"), ("occ_sum", "occ_n"), ("boundary_sum", "boundary_n"))


def load(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    splits = {r["split"] for r in rows}
    if len(splits) != 1 or not splits <= {"dev", "holdout"}:
        raise ValueError(f"{path}: expected exactly one nonempty split")
    split = splits.pop()
    manifest = Path(__file__).with_name("pairs.tsv").read_text().splitlines()
    pairs = {(scene, frame) for group, scene, frame in map(str.split, manifest) if group == split}
    expected = {(s, f, m) for s, f in pairs for m in METHODS}
    actual = {(r["scene"], r["frame"], r["method"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"{path}: incomplete, duplicate, or unexpected pair/method")
    groups = defaultdict(list)
    counts = {}
    for row in rows:
        if not math.isfinite(float(row["median_ms"])) or float(row["median_ms"]) <= 0:
            raise ValueError("invalid latency")
        sizes = tuple(int(row[n]) for _, n in METRICS)
        if sizes[0] <= 0 or any(v < 0 or v > sizes[0] for v in sizes[1:]):
            raise ValueError("invalid pixel counts")
        pair = (row["scene"], row["frame"])
        if counts.setdefault(pair, sizes) != sizes:
            raise ValueError("pixel counts differ between methods")
        for key, _ in METRICS:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("invalid EPE sum")
        groups[row["method"]].append(row)
    return groups


def mean_epe(rows, metric):
    key, count = METRICS[metric]
    return sum(float(r[key]) for r in rows) / sum(int(r[count]) for r in rows)


def stats(rows):
    times = sorted(float(r["median_ms"]) for r in rows)
    return [mean_epe(rows, i) for i in range(3)] + [times[math.ceil(.95 * len(times)) - 1]]


def summarize(path):
    groups = load(path)
    print(f"\n{path.name}\nmethod             all-EPE   occ-EPE  band-EPE   P95-ms")
    for method in METHODS:
        print(f"{method:16}", " ".join(f"{v:9.4f}" for v in stats(groups[method])))
    base, candidate = (stats(groups[m]) for m in ("fast", "adaptive"))
    scenes = sorted({r["scene"] for r in groups["fast"]})
    wins = 0
    print("scene             FAST-EPE  adaptive-EPE")
    for scene in scenes:
        before, after = (mean_epe([r for r in groups[m] if r["scene"] == scene], 0)
                         for m in ("fast", "adaptive"))
        wins += after < before
        print(f"{scene:16} {before:9.4f} {after:12.4f}")
    ratio = candidate[3] / base[3]
    passed = candidate[0] < base[0] and candidate[2] < base[2] and wins >= 3 and ratio <= 1.10
    print(f"Scene wins {wins}/{len(scenes)}, P95 ratio {ratio:.3f}, gate {'PASS' if passed else 'FAIL'}")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        summarize(path)


if __name__ == "__main__":
    main()

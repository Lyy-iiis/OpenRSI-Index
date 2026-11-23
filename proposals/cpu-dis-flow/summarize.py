"""Summarize a complete split and apply the frozen criteria to its measurements."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

METHODS = ("fast", "medium", "extra-iteration", "full-grid", "adaptive")
RANDOM = ("random-17", "random-29", "random-43")
METRICS = (("epe_sum", "n"), ("occ_sum", "occ_n"), ("boundary_sum", "boundary_n"))


def load(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    splits = {r["split"] for r in rows}
    if len(splits) != 1 or not splits <= {"dev", "holdout", "middlebury"}:
        raise ValueError(f"{path}: expected exactly one nonempty split")
    split = splits.pop()
    allocation = "checked" in rows[0]
    methods = ("fast", "adaptive", *RANDOM) if allocation else METHODS
    manifest_name = "middlebury.tsv" if split == "middlebury" else "pairs.tsv"
    manifest = Path(__file__).with_name(manifest_name).read_text().splitlines()
    pairs = {(scene, frame) for group, scene, frame in map(str.split, manifest) if group == split}
    expected = {(s, f, m) for s, f in pairs for m in methods}
    actual = {(r["scene"], r["frame"], r["method"]) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError(f"{path}: incomplete, duplicate, or unexpected pair/method")
    groups = defaultdict(list)
    counts = {}
    quotas = {}
    for row in rows:
        if not math.isfinite(float(row["median_ms"])) or float(row["median_ms"]) <= 0:
            raise ValueError("invalid latency")
        sizes = tuple(int(row[n]) for _, n in METRICS)
        if sizes[0] <= 0 or any(v < 0 or v > sizes[0] for v in sizes[1:]):
            raise ValueError("invalid pixel counts")
        pair = (row["scene"], row["frame"])
        if counts.setdefault(pair, sizes) != sizes:
            raise ValueError("pixel counts differ between methods")
        if allocation and row["method"] != "fast":
            eligible, checked = int(row["eligible"]), int(row["checked"])
            if not 0 <= checked <= eligible or quotas.setdefault(pair, (eligible, checked)) != (eligible, checked):
                raise ValueError("unmatched site counts")
        for key, _ in METRICS:
            if not math.isfinite(float(row[key])) or float(row[key]) < 0:
                raise ValueError("invalid EPE sum")
        groups[row["method"]].append(row)
    return groups


def mean_epe(rows, metric):
    key, count = METRICS[metric]
    n = sum(int(r[count]) for r in rows)
    return sum(float(r[key]) for r in rows) / n if n else math.nan


def stats(rows):
    times = sorted(float(r["median_ms"]) for r in rows)
    return [mean_epe(rows, i) for i in range(3)] + [times[math.ceil(.95 * len(times)) - 1]]


def allocation_report(groups):
    def pooled(method, scenes):
        return stats([r for r in groups[method] if r["scene"] in scenes])

    scenes = sorted({r["scene"] for r in groups["fast"]})
    is_transfer = groups["fast"][0]["split"] == "middlebury"
    print("method             all-EPE   P95-ms")
    for method in ("fast", "adaptive", *RANDOM):
        all_epe, _, _, latency = pooled(method, scenes)
        print(f"{method:16} {all_epe:9.5f} {latency:9.4f}")
    print("scene             FAST-EPE  adaptive-EPE   random-mean")
    wins = 0
    for scene in scenes:
        base, adaptive, *random = (pooled(m, [scene])[0] for m in ("fast", "adaptive", *RANDOM))
        wins += adaptive < base
        print(f"{scene:16} {base:9.5f} {adaptive:12.5f} {statistics.mean(random):13.5f}")
    base, _, _, fast_ms = pooled("fast", scenes)
    adaptive, _, _, adaptive_ms = pooled("adaptive", scenes)
    random = statistics.mean(pooled(m, scenes)[0] for m in RANDOM)
    checked = sum(int(r["checked"]) for r in groups["adaptive"])
    eligible = sum(int(r["eligible"]) for r in groups["adaptive"])
    print(f"Centers checked {checked}/{eligible}, FAST scene wins {wins}/{len(scenes)}")
    print(f"Random minus residual EPE {random - adaptive:+.5f}, P95 ratio {adaptive_ms / fast_ms:.3f}")
    if base > adaptive:
        print(f"Share of FAST gain over random {100 * (random - adaptive) / (base - adaptive):.1f}%")
    if is_transfer:
        passed = adaptive < min(base, random) and wins > len(scenes) / 2 and adaptive_ms <= 1.10 * fast_ms
        print(f"Transfer criteria this run: {'PASS' if passed else 'FAIL'}")
        other = set(scenes) - {"RubberWhale"}
        print("Excluding RubberWhale, FAST / adaptive / random mean:",
              f"{pooled('fast', other)[0]:.5f} / {pooled('adaptive', other)[0]:.5f} /",
              f"{statistics.mean(pooled(m, other)[0] for m in RANDOM):.5f}")


def summarize(path):
    groups = load(path)
    if RANDOM[0] in groups:
        print(f"\n{path.name}")
        allocation_report(groups)
        return groups
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

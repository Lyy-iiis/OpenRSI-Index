"""Report a scene-balanced loss. Confidence uses Brier, midpoint MSE, tracking final capped EPE."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def scores(path):
    by_scene = defaultdict(lambda: [0.0,0])
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row.get("step", "5") != "5":
                continue
            value = float(row["loss"] if "loss" in row else row["squared_error"])
            n = int(row["n"])
            if n < 0 or not math.isfinite(value) or value < 0:
                raise ValueError("invalid count or loss")
            totals = by_scene[row["method"],row["scene"]]
            totals[0] += value
            totals[1] += n
    out = defaultdict(dict)
    for (method,scene),(total,n) in by_scene.items():
        if not n:
            raise ValueError(f"empty scene: {scene}")
        out[method][scene] = total/n
    return out


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, nargs="+")
    for path in p.parse_args().csv:
        print(path)
        for method, scenes in scores(path).items():
            print(f"  {method:16s} {mean(scenes.values()):.7f}", " ".join(f"{k}={v:.5f}" for k,v in scenes.items()))

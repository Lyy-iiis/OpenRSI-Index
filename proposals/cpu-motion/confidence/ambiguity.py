"""Conditional Brier loss on the fixed uncertainty band of the released calibrator."""
import argparse
import json
from collections import defaultdict

from pilot import HERE, csv_out, examples, np, observations

SPREAD_EDGES = [0.5, 1, 2, 4, 8, 16]
OFFSETS = np.linspace(-3, 3, 61)


def cohort(reference):
    return (reference >= 0.25) & (reference <= 0.75)


def adjust(probability, offset):
    return 1 / (1 + np.exp(-(np.log(probability / (1 - probability)) + offset)))


def batches(root, split):
    table = np.asarray(json.loads((HERE / "model.json").read_text())["fb_photo"])
    for scene, start in examples(HERE, split):
        cycle, photo, labels, spread = observations(root, scene, start, with_disagreement=True)
        ref = table[cycle, photo]
        keep = cohort(ref)
        yield scene, start, len(ref), labels[keep], ref[keep], np.digitize(spread[keep], SPREAD_EDGES)


def fit(root):
    # Count by scene and baseline probability to optimize the scene-macro Work objective.
    counts = defaultdict(lambda: [0, 0])
    scene_sizes = defaultdict(int)
    for scene, _, _, labels, ref, bins in batches(root, "dev"):
        scene_sizes[scene] += len(ref)
        for b in np.unique(bins):
            for p in np.unique(ref[bins == b]):
                subset = (bins == b) & (ref == p)
                row = counts[int(b), scene, float(p)]
                row[0] += int(subset.sum())
                row[1] += int(labels[subset].sum())
    if not scene_sizes or min(scene_sizes.values()) == 0:
        raise ValueError("empty Work cohort")
    loss = np.zeros((len(SPREAD_EDGES)+1, len(OFFSETS)))
    for (b, scene, p), (n, bad) in counts.items():
        q = adjust(p, OFFSETS)
        loss[b] += (bad*(1-q)**2 + (n-bad)*q**2) / scene_sizes[scene]
    # Smallest absolute offset breaks ties and gives unseen bins zero adjustment.
    indices = sorted(range(len(OFFSETS)), key=lambda i: abs(OFFSETS[i]))
    return [float(OFFSETS[min(indices, key=lambda i: row[i])]) for row in loss]


def score(root, split, offsets):
    rows = []
    for scene, start, total, labels, ref, bins in batches(root, split):
        methods = (("constant-half", np.full(len(ref), 0.5)), ("cycle+appearance", ref),
                   ("motion-offset", adjust(ref, np.asarray(offsets)[bins])))
        for method, p in methods:
            rows.append(dict(scene=scene, frame=start, method=method, valid=total,
                             n=len(ref), bad=int(labels.sum()),
                             squared_error=float(np.square(p-labels).sum())))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("split", choices=("dev", "pilot", "validation", "test"))
    parser.add_argument("output")
    parser.add_argument("--fit", action="store_true", help="fit offsets on Work and write them")
    args = parser.parse_args()
    artifact = HERE / "motion-offsets.json"
    if args.fit:
        artifact.write_text(json.dumps(fit(args.root)) + "\n")
    csv_out(args.output, score(args.root, args.split, json.loads(artifact.read_text())))

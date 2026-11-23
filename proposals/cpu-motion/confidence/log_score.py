"""Predictive log loss on the frozen cohort, with a Work-fitted speed ablation."""
import argparse
import json
from collections import defaultdict

from ambiguity import HERE, batches, csv_out, np


def event_loss(logits, labels):
    # A single softplus handles both outcomes without clipping a probability.
    return np.logaddexp(0.0, np.where(labels, -logits, logits))


def examples_with_logits(root, split):
    offsets = np.asarray(json.loads((HERE / "motion-offsets.json").read_text()))
    for scene, start, valid, y, p0, spread_bin, speed in batches(root, split):
        two = np.log(p0) - np.log1p(-p0)
        yield scene, start, valid, y, two, two + offsets[spread_bin], speed


def fit_speed(root):
    # A Work-only ablation: allow a different intercept below 1, 4 and 16 px/frame.
    # Optimize log loss with equal scene weights. The prior adds one unit of
    # curvature per Work scene-average times the stated small regularizer.
    scene_sizes = defaultdict(int)
    records = []
    for scene, _, _, y, _, base, speed in examples_with_logits(root, "dev"):
        records.append((scene, y, base, np.digitize(speed, [1, 4, 16])))
        scene_sizes[scene] += len(y)
    if min(scene_sizes.values()) == 0:
        raise ValueError("empty Work cohort")
    offsets = np.zeros(4)
    for _ in range(12):
        gradient = 0.001 * offsets
        curvature = np.full(4, 0.001)
        for scene, y, base, group in records:
            p = 1 / (1 + np.exp(-(base + offsets[group])))
            gradient += np.bincount(group, weights=p-y, minlength=4) / scene_sizes[scene]
            curvature += np.bincount(group, weights=p*(1-p), minlength=4) / scene_sizes[scene]
        offsets -= np.clip(gradient / curvature, -1, 1)
    return offsets.tolist()


def score(root, split, speed_offsets):
    rows = []
    for scene, start, valid, y, two, three, speed in examples_with_logits(root, split):
        methods = (("constant-half", np.zeros_like(two)), ("cycle+appearance", two),
                   ("motion-offset", three),
                   ("speed-ablation", three + np.asarray(speed_offsets)[np.digitize(speed, [1, 4, 16])]))
        for name, logits in methods:
            p = 1 / (1 + np.exp(-logits))
            rows.append(dict(scene=scene, frame=start, method=name, valid=valid, n=len(y),
                             bad=int(y.sum()), loss=float(event_loss(logits, y).sum()),
                             squared_error=float(np.square(p-y).sum())))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("split", choices=("dev", "pilot", "validation", "test"))
    parser.add_argument("output")
    parser.add_argument("--fit", action="store_true", help="reproduce the speed diagnostic on Work")
    args = parser.parse_args()
    path = HERE / "speed-offsets.json"
    if args.fit:
        path.write_text(json.dumps(fit_speed(args.root)) + "\n")
    csv_out(args.output, score(args.root, args.split, json.loads(path.read_text())))

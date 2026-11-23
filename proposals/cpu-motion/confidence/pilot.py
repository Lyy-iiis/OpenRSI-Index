"""Predict the probability that a frozen flow vector is wrong by more than 3 pixels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import consistency, csv_out, cv, dis_pair, examples, np, read

HERE = Path(__file__).parent
FB_EDGES = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
PHOTO_EDGES = [2, 4, 8, 16, 32, 64]


def observations(root, scene, start, *, with_disagreement=False):
    a, b = (read(root, scene, "final", k) for k in (start, start+1))
    flow, back = dis_pair(a, b)
    fb, x, y = consistency(flow, back)
    warped = cv.remap(b, x, y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    residual = np.abs(a.astype(np.int16) - warped)
    truth = read(root, scene, "flow", start)
    valid = read(root, scene, "invalid", start)[::4, ::4] == 0
    bad = np.linalg.norm(flow - truth, axis=2) > 3
    signals = [np.digitize(fb, FB_EDGES), np.digitize(residual, PHOTO_EDGES), bad]
    if with_disagreement:
        h, w = flow.shape[:2]
        padded = np.pad(flow, ((4, 4), (4, 4), (0, 0)), mode="edge")
        spread = np.zeros((h, w), dtype=flow.dtype)
        for dy, dx in ((0, 4), (0, -4), (4, 0), (-4, 0)):
            neighbor = padded[4+dy:4+dy+h, 4+dx:4+dx+w]
            spread = np.maximum(spread, np.abs(neighbor-flow).sum(axis=2))
        signals.extend((spread, np.linalg.norm(flow, axis=2)))
    return tuple(v[::4, ::4][valid] for v in signals)


def fit(root):
    shape = (len(FB_EDGES)+1, len(PHOTO_EDGES)+1)
    n, bad = np.zeros(shape), np.zeros(shape)
    for scene, start in examples(HERE, "dev"):
        fb, photo, y = observations(root, scene, start)
        np.add.at(n, (fb, photo), 1)
        np.add.at(bad, (fb, photo), y)
    prior = (bad.sum()+1) / (n.sum()+2)
    one = (bad.sum(axis=1)+32*prior) / (n.sum(axis=1)+32)
    two = (bad+32*one[:, None]) / (n+32)
    return {"constant": float(prior), "fb": one.tolist(), "fb_photo": two.tolist()}


def score(root, split, model):
    rows = []
    for scene, start in examples(HERE, split):
        fb, photo, y = observations(root, scene, start)
        predictions = [np.full(len(y), model["constant"]), np.asarray(model["fb"])[fb],
                       np.asarray(model["fb_photo"])[fb, photo]]
        for name, p in zip(("prevalence", "cycle", "cycle+appearance"), predictions):
            rows.append(dict(scene=scene, frame=start, method=name, n=len(y), bad=int(y.sum()),
                             squared_error=float(np.square(p - y).sum())))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("split", choices=("dev","pilot","validation","test"))
    parser.add_argument("output")
    parser.add_argument("--fit", action="store_true", help="write model.json from Work scenes")
    args = parser.parse_args()
    artifact = HERE / "model.json"
    if args.fit:
        artifact.write_text(json.dumps(fit(args.root), indent=2) + "\n")
    csv_out(args.output, score(args.root, args.split, json.loads(artifact.read_text())))

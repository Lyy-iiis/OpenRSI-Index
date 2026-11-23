"""Bidirectional midpoint splatting. The withheld image is used only by scoring."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import consistency, csv_out, dis_pair, examples, np, read

HERE = Path(__file__).parent


def splat(image, flow, trust):
    h, w = image.shape
    y, x = np.indices((h, w), dtype=np.float32)
    tx, ty = x + flow[..., 0] * .5, y + flow[..., 1] * .5
    ix, iy = np.floor(tx).astype(np.int32), np.floor(ty).astype(np.int32)
    dx, dy = tx-ix, ty-iy
    mass = np.zeros(h*w)
    value = np.zeros(h*w)
    for ox, oy, bilinear in ((0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)),
                             (0,1,(1-dx)*dy), (1,1,dx*dy)):
        xx, yy = ix+ox, iy+oy
        keep = (xx>=0) & (xx<w) & (yy>=0) & (yy<h)
        indices = (yy[keep]*w + xx[keep]).ravel()
        weights = (trust * bilinear)[keep]
        mass += np.bincount(indices, weights, minlength=h*w)
        value += np.bincount(indices, weights*image[keep], minlength=h*w)
    return value.reshape(h,w), mass.reshape(h,w)


def synthesize(a, b):
    forward, backward = dis_pair(a, b)
    mix = (a.astype(np.float32) + b) * .5
    yield "dissolve", mix
    for mode in ("equal", "cycle-weighted"):
        values, masses = [], []
        for source, flow, rev in ((a,forward,backward), (b,backward,forward)):
            if mode == "equal":
                trust = 1.
            else:
                error, _, _ = consistency(flow, rev)
                trust = 1 / (1 + np.minimum(error, 16))
            v, m = splat(source, flow, trust)
            values.append(v)
            masses.append(m)
        mass = masses[0] + masses[1]
        yield mode, np.divide(values[0]+values[1], mass, out=mix.astype(float), where=mass>1e-8)


def score(root, split):
    rows = []
    for scene, start in examples(HERE, split):
        a, target, b = (read(root, scene, "final", k) for k in range(start,start+3))
        for method, prediction in synthesize(a, b):
            error = np.square(prediction.astype(float)-target)
            rows.append(dict(scene=scene, frame=start, method=method, n=error.size,
                             squared_error=float(error.sum())))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("split", choices=("dev","pilot","validation","test"))
    p.add_argument("output")
    args = p.parse_args()
    csv_out(args.output, score(args.root, args.split))

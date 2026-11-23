"""Compare incremental and first-frame references on fixed five-transition tracks."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import csv_out, cv, examples, np, read

HERE = Path(__file__).parent
LK = dict(winSize=(15,15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,10,.03))


def visible_targets(root, scene, start, points):
    """Compose GT bilinearly only where all four donors have compatible visible flow."""
    pos = points.astype(np.float64).copy()
    visible = np.ones(len(points), dtype=bool)
    for t in range(start, start+5):
        flow = read(root, scene, "flow", t)
        h, w = flow.shape[:2]
        usable = (read(root, scene, "invalid", t) == 0) & (read(root, scene, "occlusions", t) == 0)
        x, y = pos.T
        visible &= (x>=0) & (x<w-1) & (y>=0) & (y<h-1)
        xx, yy = np.clip(x,0,w-2), np.clip(y,0,h-2)
        ix, iy = xx.astype(int), yy.astype(int)
        dx, dy = xx-ix, yy-iy
        displacement = np.zeros_like(pos)
        neighbors = []
        for ox, oy, weight in ((0,0,(1-dx)*(1-dy)), (1,0,dx*(1-dy)),
                               (0,1,(1-dx)*dy), (1,1,dx*dy)):
            donor = flow[iy+oy, ix+ox]
            visible &= usable[iy+oy,ix+ox]
            displacement += donor * weight[:,None]
            neighbors.append(donor)
        visible &= np.ptp(neighbors,axis=0).max(axis=1) <= 3
        pos += displacement
        visible &= (pos[:,0]>=0) & (pos[:,0]<w) & (pos[:,1]>=0) & (pos[:,1]<h)
        yield pos.copy(), visible.copy()


def proposal(before, after, points, seed=None):
    dst, status, cost = cv.calcOpticalFlowPyrLK(before, after, points, seed, **LK,
                             flags=cv.OPTFLOW_USE_INITIAL_FLOW if seed is not None else 0)
    dst = dst.reshape(-1,2)
    h,w = after.shape
    good = status.ravel().astype(bool) & np.isfinite(dst).all(axis=1)
    good &= (dst[:,0]>=0) & (dst[:,0]<w) & (dst[:,1]>=0) & (dst[:,1]<h)
    return dst, good, cost.ravel()


def track(frames, initial, mode):
    pos = initial.copy()
    active = np.ones(len(pos), dtype=bool)
    for t in range(1, len(frames)):
        step, step_ok, step_cost = proposal(frames[t-1], frames[t], pos)
        step_ok &= active
        seed = np.where(step_ok[:,None], step, pos)
        anchor, anchor_ok, anchor_cost = proposal(frames[0], frames[t], initial, seed.copy())
        if mode == "incremental":
            take = np.zeros(len(pos), dtype=bool)
        elif mode == "first-frame":
            take = anchor_ok
        else:
            take = anchor_ok & (~step_ok | (anchor_cost <= step_cost))
        active = np.where(take, anchor_ok, step_ok)
        best = np.where(take[:,None], anchor, step)
        pos = np.where(active[:,None], best, pos)
        yield pos.copy(), active.copy()


def score(root, split):
    rows = []
    for scene, start in examples(HERE, split):
        frames = [read(root, scene, "final", t) for t in range(start,start+6)]
        q = cv.goodFeaturesToTrack(frames[0], maxCorners=500, qualityLevel=.01, minDistance=7, blockSize=7)
        if q is None:
            raise ValueError(f"no initial features: {scene}/{start}")
        q = q.reshape(-1,2)
        targets = list(visible_targets(root,scene,start,q))
        for mode in ("incremental","first-frame","patch-choice"):
            for t, ((pos, active), (gt, visible)) in enumerate(zip(track(frames,q,mode),targets),1):
                loss = np.where(active, np.minimum(20,np.linalg.norm(pos-gt,axis=1)),20)
                rows.append(dict(scene=scene, frame=start, step=t, method=mode, n=int(visible.sum()),
                                 lost=int((visible & ~active).sum()), loss=float(loss[visible].sum())))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("split", choices=("dev","pilot","validation","test"))
    p.add_argument("output")
    args = p.parse_args()
    csv_out(args.output, score(args.root,args.split))

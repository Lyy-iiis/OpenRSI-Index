"""Shared Sintel pilot I/O. All timed vision operations use CPU arrays."""
import csv
from pathlib import Path

import cv2 as cv
import numpy as np

cv.setNumThreads(1)
cv.ocl.setUseOpenCL(False)


def member(scene, kind, i):
    ext = "flo" if kind == "flow" else "png"
    return f"training/{kind}/{scene}/frame_{i:04d}.{ext}"


def read(root, scene, kind, i):
    path = str(Path(root) / member(scene, kind, i))
    mat = cv.readOpticalFlow(path) if kind == "flow" else cv.imread(path, cv.IMREAD_GRAYSCALE)
    if mat is None:
        raise ValueError(f"missing or unreadable: {path}")
    if kind == "flow" and not np.isfinite(mat).all():
        raise ValueError(f"nonfinite flow: {path}")
    return mat


def examples(folder, split):
    with (Path(folder) / "manifest.tsv").open() as stream:
        return [(r["scene"], int(r["start"])) for r in csv.DictReader(stream, delimiter="\t")
                if r["split"] == split]


def dis_pair(a, b):
    dis = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(a, b, None), dis.calc(b, a, None)


def consistency(flow, reverse):
    y, x = np.indices(flow.shape[:2], dtype=np.float32)
    dst_x, dst_y = x + flow[..., 0], y + flow[..., 1]
    back = cv.remap(reverse, dst_x, dst_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    error = np.linalg.norm(flow + back, axis=2)
    outside = (dst_x < 0) | (dst_y < 0) | (dst_x > x.shape[1]-1) | (dst_y > x.shape[0]-1)
    return np.where(outside, np.inf, error), dst_x, dst_y


def csv_out(path, rows):
    if not rows:
        raise ValueError("empty split")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

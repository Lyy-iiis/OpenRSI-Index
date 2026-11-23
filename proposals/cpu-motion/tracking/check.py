"""A stationary textured clip keeps visible queries at their initial positions."""
from pilot import cv, np, track
rng=np.random.default_rng(7)
img=rng.integers(0,256,(96,128),dtype=np.uint8)
q=cv.goodFeaturesToTrack(img,50,.01,7,blockSize=7).reshape(-1,2)
for mode in ("incremental","first-frame","patch-choice"):
    for pos,active in track([img]*6,q,mode):
        assert active.all() and np.max(np.abs(pos-q))<.001

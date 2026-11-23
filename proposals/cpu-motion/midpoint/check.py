"""Integer half-warps conserve source mass and land at the expected pixel."""
from pilot import np, splat
a = np.zeros((32,48),np.uint8)
a[12,13]=200
flow=np.zeros((*a.shape,2),np.float32)
for dx in (0,2):
    flow[...,0]=dx
    value,mass=splat(a,flow,1.)
    assert value[12,13+dx//2]==200 and value.sum()==200
    assert mass.sum()==32*(48-dx//2)

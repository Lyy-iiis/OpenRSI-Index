"""Identity images imply zero cycle and photometric errors for zero flow."""
from pilot import consistency, cv, np
a = np.arange(256, dtype=np.uint8).reshape(16,16)
zero = np.zeros((16,16,2), np.float32)
fb,x,y = consistency(zero,zero)
assert np.all(fb==0) and np.array_equal(a,cv.remap(a,x,y,cv.INTER_LINEAR))
shift = zero.copy()
shift[...,0] = 1
fb,_,_ = consistency(shift,-shift)
assert np.all(fb[:,:-1]==0) and np.all(np.isinf(fb[:,-1]))

from ambiguity import adjust, cohort
probs = np.array([0.249999, 0.25, 0.5, 0.75, 0.750001])
assert cohort(probs).tolist() == [False, True, True, True, False]
assert np.allclose(adjust(probs, 0), probs)
assert np.all(adjust(probs, 1) > probs)

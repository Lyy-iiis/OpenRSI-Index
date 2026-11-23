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

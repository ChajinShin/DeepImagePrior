from ImageDataProcess.functional import _normalize
import numpy as np
import torch



x = np.random.randn(256, 256, 3, 5)
y = _normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
print(y.shape)





from patchify import patchify
import numpy as np

def slice_image(image):
    patches = patchify(image, (64, 64, 3), step=64)
    patched = np.array([patches[i,j,0,:,:,:] for i in range(patches.shape[0]) for j in range(patches.shape[1])])
    return patched
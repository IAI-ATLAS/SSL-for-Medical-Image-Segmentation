import numpy as np
import matplotlib.pyplot as plt

# custom Albumentation transforms

def norm_min_max(image):
    image = image.astype("float32")

    min_val = np.min(image)
    max_val = np.max(image)
    if min_val!=max_val:
        image = (image-min_val)/(max_val-min_val)
    
    return image
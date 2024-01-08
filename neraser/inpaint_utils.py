import numpy as np
import cv2
from scipy import ndimage

def dilate_mask(mask_img, radius, invert=False):
    mask_img = np.bitwise_not(mask_img) if invert else mask_img
    dilated = ndimage.maximum_filter(mask_img, size=radius).astype(np.uint8)
    # binarize
    dilated[dilated>0] = 255
    return dilated
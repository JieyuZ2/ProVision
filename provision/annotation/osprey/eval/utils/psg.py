import cv2
import os
import json
from pathlib import Path
from panopticapi.utils import rgb2id
import numpy as np

def load_image(image) -> np.ndarray:
    img_bgr = cv2.imread(str(image))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def create_segmentations(pan_seg_image, segments_info, ):
    im = rgb2id(load_image(pan_seg_image))
    segmentations = []
    for segment_info in segments_info:
        mask = im == segment_info['id']
        segmentations.append(mask)
        assert np.sum(mask) > 0
        
    return np.array(segmentations, dtype=bool)

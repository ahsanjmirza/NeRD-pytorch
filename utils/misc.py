import numpy as np
from imageio.v2 import imread
import os

def get_shape(dir):
    train_list = [
        f for f in os.listdir(dir)
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    img = imread(os.path.join(dir, train_list[0]))
    return img.shape[0], img.shape[1]
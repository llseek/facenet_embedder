import numpy as np
import cv2 # imread BGR
#import matplotlib.pyplot as cv2 # imread RGB

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y;

def l2_normalize(x):
    sum = 0
    for v in x:
        sum += np.power(v, 2)
    sqrt = np.max([np.sqrt(sum), 0.0000000001])
    y = np.zeros((x.shape))
    for (i, v) in enumerate(x):
        y[i] = v/sqrt
    return y.astype(np.float32)

def l2_distance(x1, x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

def read_image(f):
    return prewhiten(cv2.imread(f)).transpose((2, 0, 1))

class Embedder:
    pass

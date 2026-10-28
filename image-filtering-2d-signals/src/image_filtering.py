import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.util import random_noise
from skimage.feature import canny

# ============================================================
# 0. CONFIGURATION
# ============================================================

NAME = "Bagada Vishal Jayantilal"
ROLL_NO = "23EE38004"

# Setup relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the image in the results folder (which is sibling to src)
FILE_PATH = os.path.join(BASE_DIR, '..', 'results', 'lena_color_512.tif')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'results')

# Salt-and-pepper probabilities
SALT_PROB = 0.02
PEPPER_PROB = 0.05
RANDOM_SEED = 10

MEAN_KERNEL = np.ones((3, 3), dtype=np.float64) / 9.0
GAUSSIAN_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
LAPLACIAN_KERNEL = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)

# ... (The rest of the script functions remain unchanged) ...

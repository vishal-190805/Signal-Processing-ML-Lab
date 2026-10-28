import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ============================================================
# USER PARAMETERS
# ============================================================

FILTER_ORDER = 32
LAMBDA = 0.99
DELTA = 1.0

# Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, '..', 'results')
OUTPUT_FOLDER = os.path.join(BASE_DIR, '..', 'results')

CLEAN_FILE = "sp26"
NOISY_FILES = {
    0: "sp26_restaurant_sn0",
    5: "sp26_restaurant_sn5",
    10: "sp26_restaurant_sn10",
    15: "sp26_restaurant_sn15"
}

# ... (The rest of the script functions remain unchanged) ...

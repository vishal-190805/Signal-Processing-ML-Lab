import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt
from scipy.stats import kurtosis as _scipy_kurtosis

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

# Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, '..', 'results')

# If this script uses any data files, they should be referenced via RESULTS_ROOT
# Example: data_path = os.path.join(RESULTS_ROOT, 'some_file.mat')

# ... (The rest of the script functions remain unchanged) ...

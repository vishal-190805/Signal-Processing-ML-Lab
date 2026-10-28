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

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the experiment root, then into results
RESULTS_ROOT = os.path.join(BASE_DIR, '..', 'results')

# Dataset locations (Relative to results root)
TRAIN_DATA_DIR = os.path.join(RESULTS_ROOT, 'RollingElementBearingFaultDiagnosis-Data-master', 'train_data')
TEST_DATA_DIR = os.path.join(RESULTS_ROOT, 'RollingElementBearingFaultDiagnosis-Data-master', 'test_data')

# Where to save the generated figures
OUTPUT_DIR = RESULTS_ROOT

# --- Bearing geometry ---
N_ROLLERS = 8
BALL_DIAMETER = 0.235
PITCH_DIAMETER = 1.245
CONTACT_ANGLE_DEG = 0.0

INNER_FAULT_FILE = "InnerRaceFault_vload_1.mat"
BASELINE_FILE = "baseline_1.mat"
OUTER_FAULT_FILE = "OuterRaceFault_1.mat"

STFT_WINDOW_SIZE = 1024
STFT_HOP_SIZE = 256

FIR_NUMTAPS = 513
INNER_FILTER_FC = 4200.0
INNER_FILTER_BW = 2000.0
OUTER_FILTER_FC = 2670.0
OUTER_FILTER_BW = 763.0

ENVELOPE_SPECTRUM_MAX_FREQ = 1000.0
N_HARMONICS_TO_MARK = 8

SHOW_PLOTS = True
EPS = 1e-12

# ... (Rest of the logic functions remain the same as they are mathematical) ...

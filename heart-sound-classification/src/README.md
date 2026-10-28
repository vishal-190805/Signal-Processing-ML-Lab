# Heart Sound Classification and Analysis

This project focuses on the processing and analysis of Phonocardiogram (PCG) signals to classify heart sounds, utilizing advanced digital filtering and signal processing techniques.

## Objectives
The primary objective is to implement a robust filtering pipeline to remove ambient noise and baseline wander from heart sound recordings, enabling clearer identification of S1 and S2 heart sounds.

## Methodology
The signal processing pipeline consists of the following stages:

1.  **Data Acquisition**: Loading heart sound datasets in `.mat` format (e.g., PhysioNet data).
2.  **Preprocessing**: 
    *   **Normalization**: Scaling the signal to a standard range to ensure consistency across different recordings.
    *   **DC Offset Removal**: Subtracting the mean to center the signal around zero.
3.  **Digital Filtering**:
    *   **High-Pass Filtering**: Implementing a high-pass filter to eliminate low-frequency baseline wander caused by patient breathing or movement.
    *   **Low-Pass Filtering**: Implementing a low-pass filter to remove high-frequency noise and electronic interference.
    *   **Band-Pass Integration**: Combining these to isolate the primary frequency range of heart sounds (typically 20Hz to 200Hz).
4.  **Signal Analysis**: 
    *   **Time-Domain Analysis**: Observing the periodicity of S1 (lub) and S2 (dub) sounds.
    *   **Spectral Analysis**: Using Fourier transforms to analyze the frequency components of the heart sounds.

## Technical Implementation
The implementation emphasizes the use of optimized digital filters to maximize the Signal-to-Noise Ratio (SNR) without distorting the morphology of the heart sound peaks.

## Dependencies
*   `numpy`: Numerical computations and array handling.
*   `scipy.signal`: Implementation of Butterworth/Chebyshev filters and signal processing functions.
*   `matplotlib`: Visualization of filtered vs. raw signals.

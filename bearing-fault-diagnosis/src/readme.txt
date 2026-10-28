
==============================================================================
 Bearing Fault Diagnosis using Signal Processing Techniques
 EE69205 - Signal Processing and System Design Laboratory, IIT Kharagpur
==============================================================================

What this script does
----------------------
Implements every step asked for in the lab manual, for all three parts:

    Part 1 : Inner raceway fault detection
    Part 2 : Normal (baseline / healthy) condition
    Part 3 : Outer raceway fault detection

For each condition it:
    1. Loads the .mat file (MFPT bearing dataset format: a struct called
       'bearing' with fields gs, sr, rate, load, [BPFO, BPFI, FTF, BSF]).
    2. Plots the signal in time and frequency domain.
    3. Computes the FFT and the STFT of the signal FROM SCRATCH (own
       recursive radix-2 Cooley-Tukey FFT, no numpy.fft / scipy.fft calls
       anywhere in this file) and plots the results.
    4. Designs a bandpass FIR filter around the resonance band that carries
       the fault impacts.
    5. Computes the envelope of the filtered signal using the Hilbert
       transform, implemented FROM SCRATCH on top of the same FFT.
    6. Computes the one-sided power spectrum of the envelope FROM SCRATCH.
    7. Plots the envelope spectrum and overlays the theoretical BPFI / BPFO
       harmonics for comparison, exactly like the "expected graph" figure in
       the manual.
    Additionally (Part 3): computes and compares the kurtosis of every
       signal in the dataset, and applies the specific fc = 2670 Hz,
       BW = 763 Hz bandpass filter requested for the outer race case.

Required third-party libraries
-------------------------------
    numpy      - array maths
    scipy      - only used for: reading .mat files (scipy.io.loadmat),
                 designing the FIR filter (scipy.signal.firwin) and applying
                 it (scipy.signal.filtfilt), and kurtosis (scipy.stats.kurtosis).
                 NONE of these are used for FFT/STFT/Hilbert/power-spectrum -
                 those are implemented from scratch as required by the manual.
    matplotlib - plotting

Install with:
    pip install numpy scipy matplotlib

Directory layout expected (edit the constants below if yours differs)
------------------------------------------------------------------------
    E:\\7th sem\\SPML Lab\\Exp 4\\RollingElementBearingFaultDiagnosis-Data-master\\train_data
    E:\\7th sem\\SPML Lab\\Exp 4\\RollingElementBearingFaultDiagnosis-Data-master\\test_data

Author: written for Fighter's SPSD Lab, Experiment 4.
==============================================================================
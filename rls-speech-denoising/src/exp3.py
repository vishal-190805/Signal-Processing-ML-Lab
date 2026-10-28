"""
Experiment 3
Denoising Noisy Audio Speech Signal using Recursive Least Squares (RLS)

Files expected:
    sp26.wav
    sp26_restaurant_sn0.wav
    sp26_restaurant_sn5.wav
    sp26_restaurant_sn10.wav
    sp26_restaurant_sn15.wav

Outputs:
    RLS_Results/
        denoised_sn0.wav
        denoised_sn5.wav
        denoised_sn10.wav
        denoised_sn15.wav
        comparison_sn0.png
        comparison_sn5.png
        comparison_sn10.png
        comparison_sn15.png
        sdr_results.csv
        sdr_results.txt
"""

import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# ============================================================
# USER PARAMETERS
# ============================================================

# RLS filter order
FILTER_ORDER = 32

# Forgetting factor lambda
# Usually very close to 1
LAMBDA = 0.99

# Initial inverse correlation matrix scaling
# The lab manual says R[0] = I.
DELTA = 1.0

# Folder containing audio files
DATA_FOLDER = "."

# Output folder
OUTPUT_FOLDER = "RLS_Results"


# ============================================================
# FILE NAMES
# ============================================================

CLEAN_FILE = "sp26"

NOISY_FILES = {
    0: "sp26_restaurant_sn0",
    5: "sp26_restaurant_sn5",
    10: "sp26_restaurant_sn10",
    15: "sp26_restaurant_sn15"
}


# ============================================================
# FIND AUDIO FILE
# ============================================================

def find_audio_file(base_name, folder="."):
    """
    Find an audio file using its base filename.

    For example:
        sp26
    can match:
        sp26.wav
        sp26.WAV
        sp26.flac
    """

    extensions = [
        ".wav",
        ".WAV",
        ".flac",
        ".FLAC",
        ".mp3",
        ".MP3",
        ".ogg",
        ".OGG"
    ]

    # First try exact filename + extension
    for ext in extensions:
        path = os.path.join(folder, base_name + ext)

        if os.path.isfile(path):
            return path

    # Then search more generally
    pattern = os.path.join(folder, base_name + ".*")
    matches = glob.glob(pattern)

    if len(matches) > 0:
        return matches[0]

    raise FileNotFoundError(
        f"\nCould not find audio file: {base_name}\n"
        f"Looked inside: {os.path.abspath(folder)}\n"
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(filename):
    """
    Load an audio file and convert it to mono floating-point signal.

    Returns:
        sample_rate
        signal
    """

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".wav":
        sample_rate, signal = wavfile.read(filename)

    else:
        # librosa is used for formats other than WAV
        try:
            import librosa

            signal, sample_rate = librosa.load(
                filename,
                sr=None,
                mono=True
            )

            signal = signal.astype(np.float64)

            return sample_rate, signal

        except ImportError:
            raise ImportError(
                "Install librosa to read non-WAV audio files:\n"
                "pip install librosa"
            )

    # Convert stereo to mono
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Convert integer audio to floating point
    if np.issubdtype(signal.dtype, np.integer):

        info = np.iinfo(signal.dtype)

        max_value = max(abs(info.min), info.max)

        signal = signal.astype(np.float64) / max_value

    else:
        signal = signal.astype(np.float64)

    return sample_rate, signal


# ============================================================
# NORMALIZE SIGNAL
# ============================================================

def normalize_signal(signal):
    """
    Normalize signal to [-1, 1].
    """

    max_value = np.max(np.abs(signal))

    if max_value == 0:
        return signal.copy()

    return signal / max_value


# ============================================================
# RLS ALGORITHM
# ============================================================

def rls_denoise(noisy_signal, clean_signal, filter_order, lam, delta=1.0):
    """
    RLS adaptive filtering according to the formulation
    given in the laboratory manual.

    Input:
        noisy_signal  -> x[n]
        clean_signal  -> desired signal d[n]

    Returns:
        denoised_signal
        error_signal
        final_weights
    """

    # --------------------------------------------------------
    # Make both signals the same length
    # --------------------------------------------------------

    N = min(len(noisy_signal), len(clean_signal))

    noisy_signal = noisy_signal[:N]
    clean_signal = clean_signal[:N]

    # --------------------------------------------------------
    # Initialize
    #
    # W[0] = 0
    # R[0] = I
    # --------------------------------------------------------

    W = np.zeros(filter_order, dtype=np.float64)

    R = delta * np.eye(filter_order, dtype=np.float64)

    # Output signal
    denoised_signal = np.zeros(N, dtype=np.float64)

    # Error signal
    error_signal = np.zeros(N, dtype=np.float64)

    # --------------------------------------------------------
    # RLS processing
    # --------------------------------------------------------

    for n in range(filter_order, N):

        # ----------------------------------------------------
        # Input vector:
        #
        # X[n] = [x[n-1], x[n-2], ..., x[n-M]]
        # ----------------------------------------------------

        X = noisy_signal[
            
            n - filter_order:n
        ][::-1]

        # ----------------------------------------------------
        # Estimated output
        #
        # d_hat[n] = W[n-1]^T X[n]
        # ----------------------------------------------------

        estimated_output = np.dot(W, X)

        denoised_signal[n] = estimated_output

        # ----------------------------------------------------
        # Estimation error
        #
        # e[n] = d[n] - d_hat[n]
        # ----------------------------------------------------

        error = clean_signal[n] - estimated_output

        error_signal[n] = error

        # ----------------------------------------------------
        # Kalman gain
        #
        # K[n] =
        # R[n-1] X[n]
        # -------------------------
        # lambda + X[n]^T R[n-1] X[n]
        # ----------------------------------------------------

        RX = np.dot(R, X)

        denominator = lam + np.dot(X, RX)

        # Avoid numerical problems
        if abs(denominator) < 1e-12:
            denominator = 1e-12

        K = RX / denominator

        # ----------------------------------------------------
        # Update filter coefficients
        #
        # W[n] = W[n-1] + K[n] e[n]
        # ----------------------------------------------------

        W = W + K * error

        # ----------------------------------------------------
        # Update inverse correlation matrix
        #
        # R[n] = (1/lambda)
        #        [R[n-1] - K[n] X[n]^T R[n-1]]
        # ----------------------------------------------------

        R = (
            R
            - np.outer(K, np.dot(X, R))
        ) / lam

    return denoised_signal, error_signal, W


# ============================================================
# SDR CALCULATION
# ============================================================

def calculate_sdr(clean_signal, estimated_signal):
    """
    Calculate Signal-to-Distortion Ratio (SDR).

        SDR = 10 log10(
            sum(clean^2) /
            sum((clean - estimated)^2)
        )

    Higher SDR = better reconstruction.
    """

    N = min(len(clean_signal), len(estimated_signal))

    clean = clean_signal[:N]
    estimated = estimated_signal[:N]

    distortion = clean - estimated

    signal_power = np.sum(clean ** 2)
    distortion_power = np.sum(distortion ** 2)

    if distortion_power == 0:
        return np.inf

    if signal_power == 0:
        return -np.inf

    sdr = 10 * np.log10(
        signal_power / distortion_power
    )

    return sdr


# ============================================================
# MSE CALCULATION
# ============================================================

def calculate_mse(clean_signal, estimated_signal):
    """
    Mean Squared Error.
    """

    N = min(len(clean_signal), len(estimated_signal))

    clean = clean_signal[:N]
    estimated = estimated_signal[:N]

    mse = np.mean(
        (clean - estimated) ** 2
    )

    return mse


# ============================================================
# RMSE CALCULATION
# ============================================================

def calculate_rmse(clean_signal, estimated_signal):
    """
    Root Mean Squared Error.
    """

    mse = calculate_mse(
        clean_signal,
        estimated_signal
    )

    return np.sqrt(mse)


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(
    clean_signal,
    noisy_signal,
    denoised_signal,
    sample_rate,
    snr,
    output_folder
):
    """
    Plot:
        1. Clean speech
        2. Noisy speech
        3. RLS denoised speech

    Also plot a zoomed section for easier comparison.
    """

    N = min(
        len(clean_signal),
        len(noisy_signal),
        len(denoised_signal)
    )

    clean_signal = clean_signal[:N]
    noisy_signal = noisy_signal[:N]
    denoised_signal = denoised_signal[:N]

    time = np.arange(N) / sample_rate

    # --------------------------------------------------------
    # Full signal comparison
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True
    )

    axes[0].plot(time, clean_signal)
    axes[0].set_title("Clean Speech Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(time, noisy_signal)
    axes[1].set_title(
        f"Noisy Speech Signal - SNR = {snr} dB"
    )
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(time, denoised_signal)
    axes[2].set_title(
        f"RLS Denoised Speech - SNR = {snr} dB"
    )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.tight_layout()

    filename = os.path.join(
        output_folder,
        f"comparison_sn{snr}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# PLOT ZOOMED RESULTS
# ============================================================

def plot_zoomed_results(
    clean_signal,
    noisy_signal,
    denoised_signal,
    sample_rate,
    snr,
    output_folder
):
    """
    Plot the first 0.1 seconds for detailed comparison.
    """

    N = min(
        len(clean_signal),
        len(noisy_signal),
        len(denoised_signal)
    )

    # 0.1 second section
    zoom_samples = min(
        N,
        int(0.1 * sample_rate)
    )

    clean = clean_signal[:zoom_samples]
    noisy = noisy_signal[:zoom_samples]
    denoised = denoised_signal[:zoom_samples]

    time = np.arange(zoom_samples) / sample_rate

    plt.figure(figsize=(14, 7))

    plt.plot(
        time,
        clean,
        label="Clean"
    )

    plt.plot(
        time,
        noisy,
        label="Noisy"
    )

    plt.plot(
        time,
        denoised,
        label="RLS Denoised"
    )

    plt.title(
        f"Zoomed Comparison - SNR = {snr} dB"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    filename = os.path.join(
        output_folder,
        f"zoomed_sn{snr}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# SAVE AUDIO
# ============================================================

def save_audio(filename, sample_rate, signal):
    """
    Save floating point signal as 16-bit PCM WAV.
    """

    signal = normalize_signal(signal)

    signal_int16 = np.int16(
        signal * 32767
    )

    wavfile.write(
        filename,
        sample_rate,
        signal_int16
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("RLS SPEECH DENOISING EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find clean signal
    # --------------------------------------------------------

    clean_path = find_audio_file(
        CLEAN_FILE,
        DATA_FOLDER
    )

    print("\nClean file:")
    print(clean_path)

    # --------------------------------------------------------
    # Load clean signal
    # --------------------------------------------------------

    clean_fs, clean_signal = load_audio(
        clean_path
    )

    print(
        f"Sampling frequency: {clean_fs} Hz"
    )

    print(
        f"Clean signal length: {len(clean_signal)} samples"
    )

    # --------------------------------------------------------
    # Process each SNR
    # --------------------------------------------------------

    results = []

    for snr, noisy_base_name in NOISY_FILES.items():

        print("\n" + "=" * 70)
        print(f"PROCESSING SNR = {snr} dB")
        print("=" * 70)

        # ----------------------------------------------------
        # Find noisy file
        # ----------------------------------------------------

        noisy_path = find_audio_file(
            noisy_base_name,
            DATA_FOLDER
        )

        print(
            f"Noisy file: {noisy_path}"
        )

        # ----------------------------------------------------
        # Load noisy signal
        # ----------------------------------------------------

        noisy_fs, noisy_signal = load_audio(
            noisy_path
        )

        print(
            f"Noisy sampling frequency: {noisy_fs} Hz"
        )

        print(
            f"Noisy signal length: {len(noisy_signal)} samples"
        )

        # ----------------------------------------------------
        # Check sampling frequency
        # ----------------------------------------------------

        if noisy_fs != clean_fs:

            raise ValueError(
                f"Sampling frequency mismatch!\n"
                f"Clean: {clean_fs} Hz\n"
                f"Noisy: {noisy_fs} Hz"
            )

        # ----------------------------------------------------
        # Match signal lengths
        # ----------------------------------------------------

        N = min(
            len(clean_signal),
            len(noisy_signal)
        )

        clean = clean_signal[:N]
        noisy = noisy_signal[:N]

        # ----------------------------------------------------
        # Normalize both signals
        #
        # This avoids numerical problems.
        # ----------------------------------------------------

        clean = normalize_signal(clean)
        noisy = normalize_signal(noisy)

        # ----------------------------------------------------
        # Calculate noisy-signal SDR
        # ----------------------------------------------------

        noisy_sdr = calculate_sdr(
            clean,
            noisy
        )

        noisy_mse = calculate_mse(
            clean,
            noisy
        )

        noisy_rmse = calculate_rmse(
            clean,
            noisy
        )

        # ----------------------------------------------------
        # Run RLS
        # ----------------------------------------------------

        print("\nRunning RLS...")

        denoised, error_signal, weights = rls_denoise(
            noisy_signal=noisy,
            clean_signal=clean,
            filter_order=FILTER_ORDER,
            lam=LAMBDA,
            delta=DELTA
        )

        print("RLS completed.")

        # ----------------------------------------------------
        # Calculate denoised SDR
        # ----------------------------------------------------

        denoised_sdr = calculate_sdr(
            clean,
            denoised
        )

        denoised_mse = calculate_mse(
            clean,
            denoised
        )

        denoised_rmse = calculate_rmse(
            clean,
            denoised
        )

        # ----------------------------------------------------
        # SDR improvement
        # ----------------------------------------------------

        sdr_improvement = (
            denoised_sdr - noisy_sdr
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "SNR (dB)": snr,
            "Noisy SDR (dB)": noisy_sdr,
            "Denoised SDR (dB)": denoised_sdr,
            "SDR Improvement (dB)": sdr_improvement,
            "Noisy MSE": noisy_mse,
            "Denoised MSE": denoised_mse,
            "Noisy RMSE": noisy_rmse,
            "Denoised RMSE": denoised_rmse
        })

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\nRESULTS")
        print("-" * 50)

        print(
            f"Input SNR              : {snr} dB"
        )

        print(
            f"Noisy SDR              : {noisy_sdr:.4f} dB"
        )

        print(
            f"Denoised SDR           : {denoised_sdr:.4f} dB"
        )

        print(
            f"SDR Improvement        : {sdr_improvement:.4f} dB"
        )

        print(
            f"Noisy MSE              : {noisy_mse:.8f}"
        )

        print(
            f"Denoised MSE           : {denoised_mse:.8f}"
        )

        print(
            f"Noisy RMSE             : {noisy_rmse:.8f}"
        )

        print(
            f"Denoised RMSE          : {denoised_rmse:.8f}"
        )

        # ----------------------------------------------------
        # Save denoised audio
        # ----------------------------------------------------

        output_audio = os.path.join(
            OUTPUT_FOLDER,
            f"denoised_sn{snr}.wav"
        )

        save_audio(
            output_audio,
            clean_fs,
            denoised
        )

        print(
            f"\nDenoised audio saved to:\n"
            f"{output_audio}"
        )

        # ----------------------------------------------------
        # Plot results
        # ----------------------------------------------------

        plot_results(
            clean_signal=clean,
            noisy_signal=noisy,
            denoised_signal=denoised,
            sample_rate=clean_fs,
            snr=snr,
            output_folder=OUTPUT_FOLDER
        )

        plot_zoomed_results(
            clean_signal=clean,
            noisy_signal=noisy,
            denoised_signal=denoised,
            sample_rate=clean_fs,
            snr=snr,
            output_folder=OUTPUT_FOLDER
        )

    # ========================================================
    # SAVE CSV RESULTS
    # ========================================================

    csv_file = os.path.join(
        OUTPUT_FOLDER,
        "sdr_results.csv"
    )

    fieldnames = [
        "SNR (dB)",
        "Noisy SDR (dB)",
        "Denoised SDR (dB)",
        "SDR Improvement (dB)",
        "Noisy MSE",
        "Denoised MSE",
        "Noisy RMSE",
        "Denoised RMSE"
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    # ========================================================
    # SAVE TEXT RESULTS
    # ========================================================

    txt_file = os.path.join(
        OUTPUT_FOLDER,
        "sdr_results.txt"
    )

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "RLS SPEECH DENOISING EXPERIMENT\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Filter Order M : {FILTER_ORDER}\n"
        )

        f.write(
            f"Forgetting Factor lambda : {LAMBDA}\n"
        )

        f.write(
            f"Initial Matrix Scaling : {DELTA}\n\n"
        )

        for result in results:

            f.write(
                f"SNR = {result['SNR (dB)']} dB\n"
            )

            f.write(
                f"Noisy SDR = "
                f"{result['Noisy SDR (dB)']:.6f} dB\n"
            )

            f.write(
                f"Denoised SDR = "
                f"{result['Denoised SDR (dB)']:.6f} dB\n"
            )

            f.write(
                f"SDR Improvement = "
                f"{result['SDR Improvement (dB)']:.6f} dB\n"
            )

            f.write(
                f"Noisy MSE = "
                f"{result['Noisy MSE']:.10f}\n"
            )

            f.write(
                f"Denoised MSE = "
                f"{result['Denoised MSE']:.10f}\n"
            )

            f.write("\n")

    # ========================================================
    # FINAL SUMMARY TABLE
    # ========================================================

    print("\n\n")
    print("=" * 85)
    print("FINAL SDR RESULTS")
    print("=" * 85)

    print(
        f"{'SNR':>8}"
        f"{'Noisy SDR':>18}"
        f"{'Denoised SDR':>20}"
        f"{'Improvement':>20}"
    )

    print("-" * 85)

    for result in results:

        print(
            f"{result['SNR (dB)']:>8}"
            f"{result['Noisy SDR (dB)']:>18.4f}"
            f"{result['Denoised SDR (dB)']:>20.4f}"
            f"{result['SDR Improvement (dB)']:>20.4f}"
        )

    print("=" * 85)

    print(
        "\nAll results have been saved in:"
    )

    print(
        os.path.abspath(OUTPUT_FOLDER)
    )

    print("\nExperiment completed successfully.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
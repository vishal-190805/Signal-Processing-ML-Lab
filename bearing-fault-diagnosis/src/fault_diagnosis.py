import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import firwin, filtfilt
from scipy.stats import kurtosis as _scipy_kurtosis

# ==============================================================================
# 0. CONFIGURATION -- edit these to match your machine
# ==============================================================================

# --- Dataset locations (as given) ---
TRAIN_DATA_DIR = r"E:\7th sem\SPML Lab\Exp 4\RollingElementBearingFaultDiagnosis-Data-master\train_data"
TEST_DATA_DIR = r"E:\7th sem\SPML Lab\Exp 4\RollingElementBearingFaultDiagnosis-Data-master\test_data"

# --- Where to save the generated figures ---
OUTPUT_DIR = r"E:\7th sem\SPML Lab\Exp 4\results"

# --- Bearing geometry (MFPT test-rig bearing: fixed for the whole dataset) ---
N_ROLLERS = 8          # n  : number of rolling elements
BALL_DIAMETER = 0.235  # d  : rolling element diameter (inches)
PITCH_DIAMETER = 1.245  # D  : pitch diameter (inches)
CONTACT_ANGLE_DEG = 0.0  # phi: bearing contact angle (degrees)

# --- Representative files used for the three parts of the experiment ---
# (Edit these if you want to analyse a different file for a given part.)
INNER_FAULT_FILE = "InnerRaceFault_vload_1.mat"   # Part 1
BASELINE_FILE = "baseline_1.mat"                  # Part 2
OUTER_FAULT_FILE = "OuterRaceFault_1.mat"         # Part 3 (constant-load file)

# --- STFT settings (window length MUST be a power of two for our from-scratch FFT) ---
STFT_WINDOW_SIZE = 1024
STFT_HOP_SIZE = 256

# --- Bandpass filter settings ---
FIR_NUMTAPS = 513          # FIR filter order + 1 (odd -> Type I linear phase)
INNER_FILTER_FC = 4200.0   # center frequency (Hz) of the resonance band chosen
INNER_FILTER_BW = 2000.0   # bandwidth (Hz) of the bandpass filter for Part 1
OUTER_FILTER_FC = 2670.0   # <-- explicitly given in the manual for Part 3
OUTER_FILTER_BW = 763.0    # <-- explicitly given in the manual for Part 3

# --- Envelope spectrum plotting ---
ENVELOPE_SPECTRUM_MAX_FREQ = 1000.0  # Hz, x-axis limit (matches manual's Fig. 2)
N_HARMONICS_TO_MARK = 8              # how many BPFI/BPFO harmonics to draw

# --- Misc ---
SHOW_PLOTS = True     # set False to only save PNGs without popping up windows
EPS = 1e-12           # small number to avoid log(0) / divide-by-zero


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_bearing_file(filepath):
    """
    Loads one MFPT-format .mat file and returns a plain dict with the fields
    we need, regardless of exactly how MATLAB serialised the struct.

    The real dataset stores everything inside a struct called 'bearing'
    with fields: gs, sr, rate, load, and (for fault files) BPFO, BPFI, FTF, BSF.
    This function is defensive: it will also work if the struct has a
    different variable name, or if some fields are missing (in which case
    the missing fault frequencies are computed from geometry instead).
    """
    raw = loadmat(filepath, squeeze_me=True, struct_as_record=False)

    # Find the first variable that isn't a private/meta key added by scipy.
    candidate_keys = [k for k in raw.keys() if not k.startswith("__")]
    if not candidate_keys:
        raise ValueError(f"No usable variables found in {filepath}")
    struct = raw[candidate_keys[0]]

    def _get(name, default=None):
        # scipy gives us either a mat_struct (dot-access) or, if it decided
        # the struct was record-like, a numpy void/structured array.
        if hasattr(struct, name):
            return getattr(struct, name)
        try:
            return struct[name]
        except (KeyError, IndexError, TypeError):
            return default

    gs = np.asarray(_get("gs")).astype(float).ravel()
    sr = float(_get("sr"))
    rate = float(_get("rate"))       # shaft speed, fr, in Hz
    load = float(_get("load", np.nan))

    bpfo = _get("BPFO")
    bpfi = _get("BPFI")
    ftf = _get("FTF")
    bsf = _get("BSF")

    computed = compute_fault_frequencies(rate)
    bpfo = float(bpfo) if bpfo is not None else computed["BPFO"]
    bpfi = float(bpfi) if bpfi is not None else computed["BPFI"]
    ftf = float(ftf) if ftf is not None else computed["FTF"]
    bsf = float(bsf) if bsf is not None else computed["BSF"]

    return {
        "filename": os.path.basename(filepath),
        "gs": gs,
        "sr": sr,
        "rate": rate,
        "load": load,
        "BPFO": bpfo,
        "BPFI": bpfi,
        "FTF": ftf,
        "BSF": bsf,
    }


def compute_fault_frequencies(fr, n=N_ROLLERS, d=BALL_DIAMETER, D=PITCH_DIAMETER,
                               phi_deg=CONTACT_ANGLE_DEG):
    """
    Computes BPFO, BPFI, FTF and BSF from shaft speed fr (Hz) and bearing
    geometry, using exactly the formulae given in the lab manual. Used as a
    fallback whenever a .mat file doesn't carry these fields itself.
    """
    phi = np.deg2rad(phi_deg)
    ratio = (d / D) * np.cos(phi)
    bpfo = (n * fr / 2.0) * (1 - ratio)
    bpfi = (n * fr / 2.0) * (1 + ratio)
    ftf = (fr / 2.0) * (1 - ratio)
    bsf = (D / (2.0 * d)) * fr * (1 - ratio ** 2)
    return {"BPFO": bpfo, "BPFI": bpfi, "FTF": ftf, "BSF": bsf}


def find_file(filename, search_dirs):
    """Locates `filename` inside any of `search_dirs`."""
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in {search_dirs}")


# ==============================================================================
# 2. FFT / IFFT -- IMPLEMENTED FROM SCRATCH (recursive radix-2 Cooley-Tukey)
# ==============================================================================

def next_power_of_two(n):
    """Smallest power of two >= n."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def prev_power_of_two(n):
    """Largest power of two <= n."""
    return 2 ** int(np.floor(np.log2(n)))


def fft_scratch(x):
    """
    Recursive radix-2 Cooley-Tukey FFT, implemented from first principles.

    Splits the sequence into even- and odd-indexed samples, recursively
    transforms each half, then combines them using the "butterfly"
    operation with the twiddle factors exp(-2j*pi*k/N).

    Parameters
    ----------
    x : array_like, complex, length MUST be a power of two.

    Returns
    -------
    X : ndarray (complex), the DFT of x, length N.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N & (N - 1) != 0:
        raise ValueError("fft_scratch: input length must be a power of two "
                          f"(got N={N}). Use next_power_of_two()/prev_power_of_two() "
                          "to pad or trim the signal first.")

    if N <= 1:
        return x

    even = fft_scratch(x[0::2])
    odd = fft_scratch(x[1::2])

    twiddle = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    twiddle_odd = twiddle * odd

    X = np.empty(N, dtype=complex)
    X[: N // 2] = even + twiddle_odd
    X[N // 2:] = even - twiddle_odd
    return X


def ifft_scratch(X):
    """
    Inverse FFT built on top of fft_scratch using the classic identity:
        ifft(X) = conj( fft( conj(X) ) ) / N
    This lets us reuse a single from-scratch FFT routine for both directions.
    """
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    x = fft_scratch(np.conjugate(X))
    return np.conjugate(x) / N


def fft_with_freqs(x, fs, pad_to_pow2=True):
    """
    Convenience wrapper: zero-pads x to the next power of two (needed
    because our radix-2 FFT only supports power-of-two lengths), computes
    its FFT from scratch, and returns the matching frequency axis.
    """
    x = np.asarray(x, dtype=float)
    N = next_power_of_two(len(x)) if pad_to_pow2 else len(x)
    x_padded = np.zeros(N, dtype=complex)
    x_padded[: len(x)] = x
    X = fft_scratch(x_padded)
    freqs = np.arange(N) * fs / N
    return freqs, X


# ==============================================================================
# 3. SHORT-TIME FOURIER TRANSFORM -- FROM SCRATCH
# ==============================================================================

def hann_window(length):
    """Hann window, computed directly from its definition (no scipy.signal.get_window)."""
    n = np.arange(length)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (length - 1))


def stft_scratch(x, fs, window_size=STFT_WINDOW_SIZE, hop_size=STFT_HOP_SIZE):
    """
    Short-Time Fourier Transform, implemented from scratch on top of
    fft_scratch: slide a Hann-windowed frame across the signal, take the
    FFT of every frame, and stack the magnitudes into a spectrogram matrix.

    `window_size` MUST be a power of two (default 1024) so each frame can be
    fed directly into fft_scratch without extra padding.

    Returns
    -------
    times  : ndarray, center time (s) of each frame
    freqs  : ndarray, one-sided frequency axis (Hz)
    S      : ndarray shape (len(freqs), len(times)), magnitude spectrogram
    """
    if window_size & (window_size - 1) != 0:
        raise ValueError("STFT_WINDOW_SIZE must be a power of two.")

    window = hann_window(window_size)
    n_frames = max(1, 1 + (len(x) - window_size) // hop_size)
    n_bins = window_size // 2 + 1
    freqs = np.arange(n_bins) * fs / window_size

    S = np.zeros((n_bins, n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_size
        frame = np.zeros(window_size)
        segment = x[start:start + window_size]
        frame[: len(segment)] = segment * window[: len(segment)]

        spectrum = fft_scratch(frame.astype(complex))
        S[:, i] = np.abs(spectrum[:n_bins])
        times[i] = (start + window_size / 2) / fs

    return times, freqs, S


# ==============================================================================
# 4. HILBERT TRANSFORM / ENVELOPE -- FROM SCRATCH
# ==============================================================================

def hilbert_scratch(x):
    """
    Analytic signal via the FFT-domain definition of the Hilbert transform,
    built entirely on fft_scratch / ifft_scratch:

        1. X = FFT(x)
        2. Zero out the negative-frequency half of the spectrum and double
           the positive-frequency half (DC and Nyquist bins are left alone).
        3. analytic_signal = IFFT(H)

    `x` is trimmed to the largest power-of-two length first (rather than
    zero-padded) so that no artificial samples are introduced into the
    signal -- zero-padding a real signal before a Hilbert transform distorts
    the analytic signal near the padding boundary.

    Returns
    -------
    analytic_signal : ndarray (complex), same length as the trimmed input.
    """
    N = prev_power_of_two(len(x))
    x_trimmed = np.asarray(x[:N], dtype=complex)

    X = fft_scratch(x_trimmed)

    h = np.zeros(N)
    h[0] = 1.0                    # DC
    if N % 2 == 0:
        h[N // 2] = 1.0           # Nyquist bin stays as-is
        h[1: N // 2] = 2.0        # positive frequencies doubled
    else:
        h[1:(N + 1) // 2] = 2.0

    analytic_signal = ifft_scratch(X * h)
    return analytic_signal


def envelope_scratch(x):
    """Envelope of a (real) signal = magnitude of its analytic signal."""
    analytic = hilbert_scratch(x)
    return np.abs(analytic)


# ==============================================================================
# 5. ONE-SIDED POWER SPECTRUM -- FROM SCRATCH
# ==============================================================================

def power_spectrum_scratch(x, fs):
    """
    One-sided power spectrum of x, computed entirely from fft_scratch:

        P[k] = |X[k]|^2 / N        (average power per sample in bin k)

    then folded to one-sided form by doubling every bin except DC and
    Nyquist (their energy is not split between positive/negative frequency).

    `x` is trimmed to the largest power-of-two length (see hilbert_scratch
    for why we trim instead of pad here).

    Returns
    -------
    freqs     : ndarray, one-sided frequency axis (Hz)
    power     : ndarray, one-sided power spectrum
    amplitude : ndarray, sqrt(power) -- handy since the manual's example
                figure is labelled "Peak Amplitude" rather than power.
    """
    N = prev_power_of_two(len(x))
    x_trimmed = np.asarray(x[:N], dtype=complex)

    X = fft_scratch(x_trimmed)
    power_two_sided = (np.abs(X) ** 2) / N

    half = N // 2
    power = power_two_sided[: half + 1].copy()
    power[1:-1] *= 2.0  # fold negative-frequency energy into positive side

    freqs = np.arange(half + 1) * fs / N
    amplitude = np.sqrt(power)
    return freqs, power, amplitude


# ==============================================================================
# 6. BANDPASS FIR FILTER (scipy.signal is used here -- only the four DSP
#    building blocks explicitly marked "from Scratch" in the manual are
#    hand-implemented above; filter *design* is a standard library task)
# ==============================================================================

def design_bandpass_fir(fs, center_freq, bandwidth, numtaps=FIR_NUMTAPS):
    """
    Designs a linear-phase FIR bandpass filter around `center_freq` with the
    given `bandwidth`, using a Hamming window (scipy.signal.firwin).
    """
    low = max(center_freq - bandwidth / 2.0, 1.0)
    high = min(center_freq + bandwidth / 2.0, fs / 2.0 - 1.0)
    if low >= high:
        raise ValueError(f"Invalid filter band [{low}, {high}] Hz for fs={fs} Hz.")
    taps = firwin(numtaps, [low, high], pass_zero=False, window="hamming", fs=fs)
    return taps, (low, high)


def apply_filter(x, taps):
    """Zero-phase filtering (filtfilt) so the envelope isn't shifted in time."""
    return filtfilt(taps, [1.0], x)


# ==============================================================================
# 7. KURTOSIS
# ==============================================================================

def compute_kurtosis(x):
    """
    Standard (Pearson, non-excess) kurtosis: 3.0 for a Gaussian signal,
    higher values indicate the "peaky, impulsive" time series typical of a
    bearing fault. Uses scipy.stats.kurtosis(fisher=False).
    """
    return float(_scipy_kurtosis(x, fisher=False))


# ==============================================================================
# 8. PLOTTING HELPERS
# ==============================================================================

def _savefig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved: {path}")


def plot_time_domain(t, x, title, output_dir, tag):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, x, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (g)")
    ax.set_title(f"Time Domain: {title}")
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, f"{tag}_time_domain.png")
    return fig


def plot_frequency_domain(freqs, X, title, output_dir, tag, max_freq=None):
    N = len(X)
    half = N // 2
    mag = np.abs(X[:half]) / N * 2  # single-sided, amplitude-normalised

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(freqs[:half], mag, linewidth=0.6)
    if max_freq:
        ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Frequency Domain (FFT, from scratch): {title}")
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, f"{tag}_frequency_domain.png")
    return fig


def plot_stft(times, freqs, S, title, output_dir, tag, max_freq=None):
    S_db = 20 * np.log10(S + EPS)
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(times, freqs, S_db, shading="auto", cmap="viridis")
    if max_freq:
        ax.set_ylim(0, max_freq)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"STFT Spectrogram (from scratch): {title}")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    _savefig(fig, output_dir, f"{tag}_stft.png")
    return fig


def plot_filtered_and_envelope(t, filtered, env, title, output_dir, tag):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, filtered, linewidth=0.5)
    axes[0].set_title(f"Bandpass-Filtered Signal: {title}")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, filtered, linewidth=0.4, alpha=0.5, label="Filtered signal")
    axes[1].plot(t[: len(env)], env, linewidth=1.2, color="crimson", label="Envelope (|Hilbert|)")
    axes[1].set_title("Envelope (from-scratch Hilbert transform)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (g)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    _savefig(fig, output_dir, f"{tag}_filtered_envelope.png")
    return fig


def plot_envelope_spectrum(freqs, amplitude, fault_freq, fault_label, title,
                            output_dir, tag, max_freq=ENVELOPE_SPECTRUM_MAX_FREQ,
                            n_harmonics=N_HARMONICS_TO_MARK):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, amplitude, linewidth=0.8, label="Envelope Spectrum")

    if fault_freq is not None:
        for k in range(1, n_harmonics + 1):
            harmonic = k * fault_freq
            if harmonic > max_freq:
                break
            ax.axvline(harmonic, color="darkorange", linestyle="--", linewidth=1,
                       label=f"{fault_label} Harmonics" if k == 1 else None)

    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Peak Amplitude")
    ax.set_title(f"Envelope Spectrum: {title}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    _savefig(fig, output_dir, f"{tag}_envelope_spectrum.png")
    return fig


def plot_kurtosis_comparison(labels, values, output_dir):
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 4.5))
    colors = ["seagreen" if "baseline" in lbl.lower() else
              "steelblue" if "inner" in lbl.lower() else
              "indianred" for lbl in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(3.0, color="black", linestyle=":", linewidth=1, label="Gaussian reference (=3)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis Comparison Across All Available Signals")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _savefig(fig, output_dir, "kurtosis_comparison_all_signals.png")
    return fig


# ==============================================================================
# 9. FULL PER-FILE PIPELINE (steps 1-7 of the manual, run for one file)
# ==============================================================================

def run_full_pipeline(filepath, title, tag, output_dir, fault_key=None,
                       filter_fc=None, filter_bw=None):
    """
    Runs the complete Part-1/2/3 style pipeline on a single .mat file:
    load -> time/freq domain plots -> STFT -> bandpass filter -> envelope
    (Hilbert) -> envelope power spectrum -> comparison with BPFI/BPFO.

    Parameters
    ----------
    filepath   : path to the .mat file to analyse
    title      : human-readable title used in plot headers
    tag        : short filename-safe tag used as a prefix for saved PNGs
    output_dir : directory to save figures into
    fault_key  : "BPFI", "BPFO", or None (for the healthy baseline, where
                 there is no fault frequency to overlay)
    filter_fc, filter_bw : bandpass filter center frequency / bandwidth (Hz).
                 If None, a default resonance band is chosen automatically.

    Returns
    -------
    dict of the key computed quantities (kurtosis values, fault frequency, etc.)
    """
    print(f"\n{'=' * 70}\nAnalysing: {title}  ({os.path.basename(filepath)})\n{'=' * 70}")

    data = load_bearing_file(filepath)
    gs, sr, rate = data["gs"], data["sr"], data["rate"]
    t = np.arange(len(gs)) / sr

    print(f"  sample rate      : {sr:.1f} Hz")
    print(f"  shaft speed (fr) : {rate:.3f} Hz")
    print(f"  load             : {data['load']}")
    print(f"  BPFO / BPFI / FTF / BSF : "
          f"{data['BPFO']:.2f} / {data['BPFI']:.2f} / {data['FTF']:.2f} / {data['BSF']:.2f} Hz")

    # --- Step 2: time & frequency domain plots ---
    plot_time_domain(t, gs, title, output_dir, tag)

    freqs, X = fft_with_freqs(gs, sr)
    plot_frequency_domain(freqs, X, title, output_dir, tag, max_freq=sr / 2)

    # --- Step 3: STFT ---
    stft_times, stft_freqs, S = stft_scratch(gs, sr)
    plot_stft(stft_times, stft_freqs, S, title, output_dir, tag, max_freq=min(sr / 2, 10000))

    raw_kurtosis = compute_kurtosis(gs)
    print(f"  Kurtosis (raw signal) = {raw_kurtosis:.3f}")

    # --- Step 4: bandpass FIR filter ---
    if filter_fc is None or filter_bw is None:
        # Sensible default: a broad high-frequency resonance band, common in
        # bearing analysis, when the student hasn't picked one explicitly.
        filter_fc = min(0.25 * sr, 4000.0)
        filter_bw = filter_fc  # roughly one octave wide
    taps, (low, high) = design_bandpass_fir(sr, filter_fc, filter_bw)
    print(f"  Bandpass filter      : {low:.1f} - {high:.1f} Hz "
          f"(center={filter_fc:.1f} Hz, BW={filter_bw:.1f} Hz, taps={len(taps)})")
    filtered = apply_filter(gs, taps)
    filtered_kurtosis = compute_kurtosis(filtered)
    print(f"  Kurtosis (filtered)  = {filtered_kurtosis:.3f}")

    # --- Step 5: envelope via from-scratch Hilbert transform ---
    env = envelope_scratch(filtered)
    envelope_kurtosis = compute_kurtosis(env)
    print(f"  Kurtosis (envelope)  = {envelope_kurtosis:.3f}")
    plot_filtered_and_envelope(t, filtered, env, title, output_dir, tag)

    # --- Step 6: one-sided power spectrum of the envelope, from scratch ---
    env_freqs, env_power, env_amplitude = power_spectrum_scratch(env, sr)

    # --- Step 7: plot & compare with the theoretical fault frequency ---
    fault_freq = data.get(fault_key) if fault_key else None
    plot_envelope_spectrum(env_freqs, env_amplitude, fault_freq, fault_key or "",
                            title, output_dir, tag)

    if fault_freq is not None:
        # Report the closest detected spectral peak to the expected fault
        # frequency, to make the "compare with given BPFI/BPFO" step concrete.
        band = (env_freqs > fault_freq * 0.85) & (env_freqs < fault_freq * 1.15)
        if np.any(band):
            peak_idx_local = np.argmax(env_amplitude[band])
            detected_freq = env_freqs[band][peak_idx_local]
            print(f"  Expected {fault_key}   = {fault_freq:.2f} Hz")
            print(f"  Detected peak near {fault_key} = {detected_freq:.2f} Hz "
                  f"(error = {abs(detected_freq - fault_freq):.2f} Hz)")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

    return {
        "filename": data["filename"],
        "condition": title,
        "raw_kurtosis": raw_kurtosis,
        "filtered_kurtosis": filtered_kurtosis,
        "envelope_kurtosis": envelope_kurtosis,
        "BPFO": data["BPFO"],
        "BPFI": data["BPFI"],
        "FTF": data["FTF"],
        "BSF": data["BSF"],
    }


# ==============================================================================
# 10. KURTOSIS ACROSS EVERY SIGNAL IN THE DATASET (Part 3, step 3)
# ==============================================================================

def compare_kurtosis_across_dataset(search_dirs, output_dir):
    """
    Loads every .mat file found in `search_dirs`, computes the raw-signal
    kurtosis of each, prints a summary table, and saves a comparison bar
    chart. This directly answers "Compute the Kurtosis value of each signal
    and compare" (Part 3, step 3).
    """
    print(f"\n{'=' * 70}\nKurtosis comparison across the full dataset\n{'=' * 70}")

    filepaths = []
    for d in search_dirs:
        filepaths.extend(sorted(glob.glob(os.path.join(d, "*.mat"))))

    labels, values = [], []
    for fp in filepaths:
        try:
            data = load_bearing_file(fp)
        except Exception as exc:  # pragma: no cover - defensive only
            print(f"  [skipped] {os.path.basename(fp)}: {exc}")
            continue
        k = compute_kurtosis(data["gs"])
        labels.append(os.path.splitext(data["filename"])[0])
        values.append(k)
        print(f"  {data['filename']:<32s} kurtosis = {k:6.3f}")

    if labels:
        plot_kurtosis_comparison(labels, values, output_dir)
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close("all")

    return dict(zip(labels, values))


# ==============================================================================
# 11. MAIN
# ==============================================================================

def main(train_dir=TRAIN_DATA_DIR, test_dir=TEST_DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    search_dirs = [train_dir, test_dir]

    # ---------------- Part 1: Inner Raceway Fault ----------------
    inner_path = find_file(INNER_FAULT_FILE, search_dirs)
    run_full_pipeline(
        inner_path, title="Inner Raceway Fault", tag="part1_inner",
        output_dir=output_dir, fault_key="BPFI",
        filter_fc=INNER_FILTER_FC, filter_bw=INNER_FILTER_BW,
    )

    # ---------------- Part 2: Normal (Healthy) Condition ----------------
    baseline_path = find_file(BASELINE_FILE, search_dirs)
    run_full_pipeline(
        baseline_path, title="Normal / Baseline Condition", tag="part2_baseline",
        output_dir=output_dir, fault_key=None,
        filter_fc=INNER_FILTER_FC, filter_bw=INNER_FILTER_BW,
    )

    # ---------------- Part 3: Outer Raceway Fault ----------------
    outer_path = find_file(OUTER_FAULT_FILE, search_dirs)
    run_full_pipeline(
        outer_path, title="Outer Raceway Fault", tag="part3_outer",
        output_dir=output_dir, fault_key="BPFO",
        filter_fc=OUTER_FILTER_FC, filter_bw=OUTER_FILTER_BW,
    )

    # ---------------- Part 3, step 3: kurtosis of every signal ----------------
    compare_kurtosis_across_dataset(search_dirs, output_dir)

    print(f"\nAll figures have been saved to: {output_dir}")


if __name__ == "__main__":
    main()
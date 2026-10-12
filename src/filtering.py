
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, sosfreqz


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

LOWCUT = 100       # Lower cutoff frequency in Hz
HIGHCUT = 4000     # Upper cutoff frequency in Hz
FILTER_ORDER = 4


# ============================================================
# 2. DESIGN AND APPLY BAND-PASS FILTER
# ============================================================

def filter_audio(
    audio,
    sample_rate,
    lowcut=LOWCUT,
    highcut=HIGHCUT,
    order=FILTER_ORDER
):
    """
    Apply a Butterworth band-pass filter.

    Inputs:
        audio: 1-D NumPy array containing mono audio
        sample_rate: Sampling rate in Hz
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        order: Butterworth filter order

    Returns:
        filtered_audio: Filtered audio array
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1:
        raise ValueError("Input audio must be mono.")

    if len(audio) == 0:
        raise ValueError("Input audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    nyquist = sample_rate / 2

    if not (0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Cutoffs must satisfy 0 < lowcut < highcut "
            f"< Nyquist frequency ({nyquist} Hz)."
        )

    # Design a Butterworth band-pass filter in SOS form.
    sos = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    # Forward-backward filtering avoids phase delay.
    filtered_audio = sosfiltfilt(sos, audio)

    return filtered_audio


# ============================================================
# 3. PLOT FILTER FREQUENCY RESPONSE
# ============================================================

def plot_filter_response(sample_rate):
    """Plot the magnitude response of the designed filter."""

    sos = butter(
        FILTER_ORDER,
        [LOWCUT, HIGHCUT],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    frequencies, response = sosfreqz(
        sos,
        worN=4096,
        fs=sample_rate
    )

    magnitude_db = 20 * np.log10(
        np.maximum(np.abs(response), 1e-10)
    )

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, magnitude_db)

    plt.axvline(
        LOWCUT, linestyle="--",
        label=f"Low cutoff: {LOWCUT} Hz"
    )
    plt.axvline(
        HIGHCUT, linestyle="--",
        label=f"High cutoff: {HIGHCUT} Hz"
    )

    plt.title("Butterworth Band-Pass Filter Frequency Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, min(sample_rate / 2, 8000))
    plt.ylim(-80, 5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "filter_frequency_response.png"
    plt.savefig(path, dpi=200)
    plt.show()

    print(f"Filter response saved to: {path}")


# ============================================================
# 4. COMPARE ORIGINAL AND FILTERED AUDIO
# ============================================================

def plot_comparison(audio, filtered_audio, sample_rate):
    """Plot waveforms and frequency spectra."""

    time = np.arange(len(audio)) / sample_rate

    # FFT magnitude spectra
    frequencies = np.fft.rfftfreq(
        len(audio), d=1 / sample_rate
    )

    original_spectrum = np.abs(np.fft.rfft(audio))
    filtered_spectrum = np.abs(
        np.fft.rfft(filtered_audio)
    )

    original_spectrum /= max(
        np.max(original_spectrum), 1e-12
    )
    filtered_spectrum /= max(
        np.max(filtered_spectrum), 1e-12
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(time, audio, linewidth=0.6)
    axes[0, 0].set_title("Original Audio")
    axes[0, 0].set_xlabel("Time (seconds)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True)

    axes[0, 1].plot(time, filtered_audio, linewidth=0.6)
    axes[0, 1].set_title("Filtered Audio")
    axes[0, 1].set_xlabel("Time (seconds)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(True)

    max_frequency = min(sample_rate / 2, 8000)

    axes[1, 0].plot(frequencies, original_spectrum)
    axes[1, 0].set_title("Original Frequency Spectrum")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Normalized Magnitude")
    axes[1, 0].set_xlim(0, max_frequency)
    axes[1, 0].grid(True)

    axes[1, 1].plot(frequencies, filtered_spectrum)
    axes[1, 1].set_title("Filtered Frequency Spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Normalized Magnitude")
    axes[1, 1].set_xlim(0, max_frequency)
    axes[1, 1].grid(True)

    plt.tight_layout()

    path = OUTPUT_DIR / "filter_comparison.png"
    plt.savefig(path, dpi=200)
    plt.show()

    print(f"Comparison saved to: {path}")


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

def main():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    audio, sample_rate = sf.read(AUDIO_PATH)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    filtered_audio = filter_audio(audio, sample_rate)

    # Save filtered audio
    output_path = OUTPUT_DIR / "filtered_audio.wav"

    sf.write(
        output_path,
        filtered_audio,
        sample_rate
    )

    print("FILTERING RESULTS")
    print("-" * 40)
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Low cutoff: {LOWCUT} Hz")
    print(f"High cutoff: {HIGHCUT} Hz")
    print(f"Filter order: {FILTER_ORDER}")
    print(f"Original RMS: {np.sqrt(np.mean(audio ** 2)):.6f}")
    print(
        "Filtered RMS: "
        f"{np.sqrt(np.mean(filtered_audio ** 2)):.6f}"
    )
    print(f"Filtered audio saved to: {output_path}")

    plot_filter_response(sample_rate)
    plot_comparison(audio, filtered_audio, sample_rate)


if __name__ == "__main__":
    main()


import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path

from filtering import filter_audio
from speech_detection import detect_speech, get_speech_segments


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

SNR_LEVELS = [20, 10, 5]
RANDOM_SEED = 42


# ============================================================
# 2. ADD WHITE GAUSSIAN NOISE
# ============================================================

def add_white_noise(audio, snr_db, rng):
    """
    Add white Gaussian noise at a specified SNR.

    SNR (dB) = 10 * log10(signal power / noise power)
    """

    audio = np.asarray(audio, dtype=np.float64)

    signal_power = np.mean(audio ** 2)

    if signal_power <= 0:
        raise ValueError("Cannot add noise to a silent signal.")

    noise = rng.normal(size=len(audio))
    noise_power = np.mean(noise ** 2)

    target_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise *= np.sqrt(target_noise_power / noise_power)

    # Do not normalize or clip: either would change the SNR.
    return audio + noise


# ============================================================
# 3. RUN SPEECH DETECTION
# ============================================================

def run_detection(audio, sample_rate):
    """
    Run the existing RMS-based detector and extract segments.
    """

    mask, times, rms, threshold = detect_speech(
        audio, sample_rate
    )

    segments = get_speech_segments(mask, times)

    return mask, times, rms, threshold, segments


# ============================================================
# 4. MAIN EVALUATION
# ============================================================

def main():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    # Load reference recording
    audio, sample_rate = sf.read(AUDIO_PATH)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        raise ValueError("The input audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    print("\nFILTERING AND SPEECH DETECTION EVALUATION")
    print("=" * 55)
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    for snr_db in SNR_LEVELS:

        # Generate noisy signal
        noisy_audio = add_white_noise(
            audio, snr_db, rng
        )

        # Apply the existing band-pass filter
        filtered_audio = filter_audio(
            noisy_audio, sample_rate
        )

        # Run detection before and after filtering
        noisy_result = run_detection(
            noisy_audio, sample_rate
        )

        filtered_result = run_detection(
            filtered_audio, sample_rate
        )

        results.append({
            "snr": snr_db,
            "noisy_result": noisy_result,
            "filtered_result": filtered_result
        })

        # Save audio files for listening
        sf.write(
            OUTPUT_DIR / f"noisy_{snr_db}dB.wav",
            noisy_audio,
            sample_rate
        )

        sf.write(
            OUTPUT_DIR / f"filtered_{snr_db}dB.wav",
            filtered_audio,
            sample_rate
        )

        print(f"\nSNR: {snr_db} dB")
        print("-" * 40)

        print(
            f"Before filtering: "
            f"{len(noisy_result[4])} segments"
        )
        print(
            f"After filtering : "
            f"{len(filtered_result[4])} segments"
        )

        print(
            f"Threshold before: {noisy_result[3]:.5f}"
        )
        print(
            f"Threshold after : {filtered_result[3]:.5f}"
        )

        print("Filtered speech intervals:")

        if filtered_result[4]:
            for i, (start, end) in enumerate(
                filtered_result[4], 1
            ):
                print(
                    f"  Segment {i}: "
                    f"{start:.2f}s - {end:.2f}s"
                )
        else:
            print("  No speech segments detected.")

    # ========================================================
    # 5. PLOT RMS, THRESHOLDS, AND DETECTED SPEECH
    # ========================================================

    fig, axes = plt.subplots(
        3, 2, figsize=(15, 12), sharex=True
    )

    for row, result in enumerate(results):

        noisy_result = result["noisy_result"]
        filtered_result = result["filtered_result"]

        # Unfiltered signal
        noisy_mask, noisy_times, noisy_rms, noisy_threshold, _ = (
            noisy_result
        )

        ax = axes[row, 0]

        ax.plot(
            noisy_times,
            noisy_rms,
            linewidth=0.8,
            label="RMS energy"
        )

        ax.axhline(
            noisy_threshold,
            linestyle="--",
            label="Threshold"
        )

        ax.fill_between(
            noisy_times,
            0,
            noisy_rms,
            where=noisy_mask,
            alpha=0.25,
            label="Detected speech frames"
        )

        ax.set_title(
            f"Before Filtering: {result['snr']} dB SNR"
        )

        ax.set_ylabel("RMS energy")
        ax.grid(True)
        ax.legend()

        # Filtered signal
        (
            filtered_mask,
            filtered_times,
            filtered_rms,
            filtered_threshold,
            _
        ) = filtered_result

        ax = axes[row, 1]

        ax.plot(
            filtered_times,
            filtered_rms,
            linewidth=0.8,
            label="RMS energy"
        )

        ax.axhline(
            filtered_threshold,
            linestyle="--",
            label="Threshold"
        )

        ax.fill_between(
            filtered_times,
            0,
            filtered_rms,
            where=filtered_mask,
            alpha=0.25,
            label="Detected speech frames"
        )

        ax.set_title(
            f"After Filtering: {result['snr']} dB SNR"
        )

        ax.set_ylabel("RMS energy")
        ax.grid(True)
        ax.legend()

    axes[-1, 0].set_xlabel("Time (seconds)")
    axes[-1, 1].set_xlabel("Time (seconds)")

    fig.suptitle(
        "Speech Detection Before and After Band-Pass Filtering",
        fontsize=14
    )

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "filter_evaluation.png"

    plt.savefig(plot_path, dpi=200)
    plt.show()

    print("\nEvaluation completed.")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()

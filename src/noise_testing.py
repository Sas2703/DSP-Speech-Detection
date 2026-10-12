
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path
from speech_detection import detect_speech, get_speech_segments


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

SNR_LEVELS = [20, 10, 5]
RANDOM_SEED = 42


# --------------------------------------------------
# ADD WHITE NOISE AT A SPECIFIED SNR
# --------------------------------------------------

def add_white_noise(audio, snr_db, rng):
    """
    Add white Gaussian noise at the requested SNR in dB.
    """

    audio = np.asarray(audio, dtype=np.float64)

    signal_power = np.mean(audio ** 2)

    if signal_power <= 0:
        raise ValueError("Cannot add SNR noise to a silent signal.")

    noise = rng.normal(0, 1, len(audio))
    noise_power = np.mean(noise ** 2)

    desired_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise = noise * np.sqrt(
        desired_noise_power / noise_power
    )

    noisy_audio = audio + noise

    # Prevent clipping from corrupting the SNR calculation.
    peak = np.max(np.abs(noisy_audio))
    if peak > 1:
        noisy_audio = noisy_audio / peak

    return noisy_audio


# --------------------------------------------------
# MAIN TEST
# --------------------------------------------------

def main():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    audio, sample_rate = sf.read(AUDIO_PATH)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    # Reproducible noise for each run
    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for snr_db in SNR_LEVELS:

        noisy_audio = add_white_noise(
            audio, snr_db, rng
        )

        # Save the noisy audio for listening and later use
        noisy_path = (
            OUTPUT_DIR / f"noisy_{snr_db}dB.wav"
        )

        sf.write(noisy_path, noisy_audio, sample_rate)

        # Run the existing speech detector
        mask, frame_times, rms_values, threshold = (
            detect_speech(noisy_audio, sample_rate)
        )

        segments = get_speech_segments(
            mask, frame_times
        )

        results.append({
            "snr": snr_db,
            "audio": noisy_audio,
            "mask": mask,
            "times": frame_times,
            "rms": rms_values,
            "threshold": threshold,
            "segments": segments
        })

        print(f"\nSNR: {snr_db} dB")
        print(f"Threshold: {threshold:.5f}")
        print(f"Detected segments: {len(segments)}")

        for i, (start, end) in enumerate(segments, 1):
            print(
                f"  Segment {i}: "
                f"{start:.2f}s - {end:.2f}s"
            )

    # --------------------------------------------------
    # COMPARISON PLOT
    # --------------------------------------------------

    time = np.arange(len(audio)) / sample_rate

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 11), sharex=True
    )

    axes[0].plot(time, audio, linewidth=0.6)
    axes[0].set_title("Original Audio")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    for ax, result in zip(axes[1:], results):

        ax.plot(
            result["times"],
            result["rms"],
            linewidth=0.8,
            label="RMS"
        )

        ax.axhline(
            result["threshold"],
            linestyle="--",
            label="Threshold"
        )

        ax.set_title(
            f"Speech Detection at {result['snr']} dB SNR"
        )

        ax.set_ylabel("RMS")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "noise_comparison.png"

    plt.savefig(plot_path, dpi=200)
    plt.show()

    print(f"\nComparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

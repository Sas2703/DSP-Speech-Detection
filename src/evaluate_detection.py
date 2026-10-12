
import csv
import numpy as np
import soundfile as sf

from pathlib import Path

from filtering import filter_audio
from speech_detection import detect_speech, get_speech_segments


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

HOP_MS = 10

# IMPORTANT:
# Replace these example intervals with speech intervals
# verified by listening to your original audio.
#
# Format: (start_time_seconds, end_time_seconds)
GROUND_TRUTH = [
    
    (2.65, 4.30),
    (5.00, 5.92),
    (6.25, 11.10),
    (12.85, 13.60)
    # Add other verified speech intervals here.
]


# ============================================================
# CONVERT TIME INTERVALS TO FRAME LABELS
# ============================================================

def intervals_to_mask(times, intervals):
    """Create a Boolean mask for manually labelled speech frames."""

    mask = np.zeros(len(times), dtype=bool)

    for start, end in intervals:
        mask |= (times >= start) & (times < end)

    return mask


# ============================================================
# CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(reference, predicted):
    """Calculate precision, recall, F1, and frame accuracy."""

    tp = int(np.sum(reference & predicted))
    fp = int(np.sum(~reference & predicted))
    fn = int(np.sum(reference & ~predicted))
    tn = int(np.sum(~reference & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (
        (tp + tn) / len(reference)
        if len(reference) else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio not found: {AUDIO_PATH}"
        )

    if not GROUND_TRUTH:
        raise ValueError(
            "Add manually verified speech intervals to GROUND_TRUTH."
        )

    original, sample_rate = sf.read(AUDIO_PATH)

    if original.ndim == 2:
        original = original.mean(axis=1)

    original = np.asarray(original, dtype=np.float64)

    # Use the same filter as the current detector
    filtered = filter_audio(original, sample_rate)

    (
        predicted_mask,
        frame_times,
        rms_values,
        start_threshold,
        continue_threshold,
    ) = detect_speech(filtered, sample_rate)

    # Reference labels are defined on the same time axis
    reference_mask = intervals_to_mask(
        frame_times, GROUND_TRUTH
    )

    metrics = calculate_metrics(
        reference_mask, predicted_mask
    )

    # Report
    print("\nSPEECH DETECTION EVALUATION")
    print("=" * 45)
    print(f"Audio duration: {len(original) / sample_rate:.2f} s")
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Start threshold: {start_threshold:.6f}")
    print(f"Continue threshold: {continue_threshold:.6f}")

    print("\nGround-truth intervals:")
    for start, end in GROUND_TRUTH:
        print(f"  {start:.2f} s - {end:.2f} s")

    print("\nFRAME-LEVEL METRICS")
    print("-" * 45)

    for name, value in metrics.items():
        if name in ("Precision", "Recall", "F1", "Accuracy"):
            print(f"{name:12s}: {value:.4f} ({value * 100:.2f}%)")
        else:
            print(f"{name:12s}: {value}")

    # Save detailed frame labels for inspection
    csv_path = OUTPUT_DIR / "detection_evaluation.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "time_seconds",
            "rms",
            "reference_speech",
            "predicted_speech",
        ])

        for i, time in enumerate(frame_times):
            writer.writerow([
                f"{time:.3f}",
                f"{rms_values[i]:.8f}",
                int(reference_mask[i]),
                int(predicted_mask[i]),
            ])

    print(f"\nFrame labels saved to: {csv_path}")


if __name__ == "__main__":
    main()

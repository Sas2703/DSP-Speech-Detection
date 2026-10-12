
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path

from filtering import filter_audio
from speech_detection import detect_speech


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNR_LEVELS = [None, 20, 10, 5]
RANDOM_SEED = 42

# IMPORTANT:
# These are provisional reference intervals.
# Verify the boundaries by listening to the audio.
GROUND_TRUTH = [
    (2.20, 4.10),
    (5.00, 5.80),
    (6.20, 11.00),
    (12.90, 13.40),
]


# ============================================================
# 2. ADD WHITE GAUSSIAN NOISE AT A SPECIFIED SNR
# ============================================================

def add_white_noise(audio, snr_db, rng):
    """Add white noise at the requested signal-to-noise ratio."""

    audio = np.asarray(audio, dtype=np.float64)

    signal_power = np.mean(audio ** 2)

    if signal_power <= 0:
        raise ValueError("Input audio has zero signal power.")

    noise = rng.normal(0.0, 1.0, size=len(audio))
    noise_power = np.mean(noise ** 2)

    target_noise_power = signal_power / (
        10.0 ** (snr_db / 10.0)
    )

    noise *= np.sqrt(target_noise_power / noise_power)

    return audio + noise


# ============================================================
# 3. GROUND-TRUTH LABELS
# ============================================================

def make_reference_mask(frame_times):
    """
    Create reference labels on the exact timestamp array
    returned by the speech detector.
    """

    reference = np.zeros(len(frame_times), dtype=bool)

    for start, end in GROUND_TRUTH:
        reference |= (
            (frame_times >= start) &
            (frame_times < end)
        )

    return reference


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(reference, predicted):
    """Calculate frame-level precision, recall, F1 and accuracy."""

    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)

    if reference.shape != predicted.shape:
        raise ValueError(
            "Reference and prediction lengths differ: "
            f"{len(reference)} versus {len(predicted)}."
        )

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
# 5. EVALUATE ONE SIGNAL
# ============================================================

def evaluate_signal(audio, sample_rate):
    """
    Run the existing detector and evaluate it against reference
    labels generated using that detector's exact frame timestamps.
    """

    (
        predicted_mask,
        frame_times,
        rms_values,
        start_threshold,
        continue_threshold,
    ) = detect_speech(audio, sample_rate)

    reference_mask = make_reference_mask(frame_times)

    metrics = calculate_metrics(
        reference_mask,
        predicted_mask
    )

    return {
        "mask": predicted_mask,
        "times": frame_times,
        "rms": rms_values,
        "reference": reference_mask,
        "start_threshold": start_threshold,
        "continue_threshold": continue_threshold,
        "metrics": metrics,
    }


# ============================================================
# 6. PRINT RESULTS
# ============================================================

def print_metrics(condition, stage, result):

    metrics = result["metrics"]

    print(f"\n{condition} | {stage}")
    print("-" * 48)

    print(f"Start threshold:    {result['start_threshold']:.6f}")
    print(f"Continue threshold: {result['continue_threshold']:.6f}")

    print(f"True positives:  {metrics['TP']}")
    print(f"False positives: {metrics['FP']}")
    print(f"False negatives: {metrics['FN']}")
    print(f"True negatives:  {metrics['TN']}")

    print(
        f"Precision: {metrics['Precision']:.4f} "
        f"({metrics['Precision'] * 100:.2f}%)"
    )

    print(
        f"Recall:    {metrics['Recall']:.4f} "
        f"({metrics['Recall'] * 100:.2f}%)"
    )

    print(
        f"F1 score:  {metrics['F1']:.4f} "
        f"({metrics['F1'] * 100:.2f}%)"
    )

    print(
        f"Accuracy:  {metrics['Accuracy']:.4f} "
        f"({metrics['Accuracy'] * 100:.2f}%)"
    )


# ============================================================
# 7. MAIN EVALUATION
# ============================================================

def main():

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    audio, sample_rate = sf.read(AUDIO_PATH)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float64)

    if len(audio) == 0:
        raise ValueError("Input audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Input audio contains invalid values.")

    print("\nNOISY SPEECH DETECTION EVALUATION")
    print("=" * 65)

    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Ground-truth intervals: {len(GROUND_TRUTH)}")

    # Reproducible noise generation
    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for snr_db in SNR_LEVELS:

        # ----------------------------------------------------
        # A. CREATE TEST SIGNAL
        # ----------------------------------------------------

        if snr_db is None:
            test_audio = audio.copy()
            condition = "Clean"
        else:
            test_audio = add_white_noise(
                audio,
                snr_db,
                rng
            )
            condition = f"{snr_db} dB SNR"

        # ----------------------------------------------------
        # B. DETECT BEFORE FILTERING
        # ----------------------------------------------------

        before = evaluate_signal(
            test_audio,
            sample_rate
        )

        # ----------------------------------------------------
        # C. FILTER THE SAME TEST SIGNAL
        # ----------------------------------------------------

        filtered_audio = filter_audio(
            test_audio,
            sample_rate
        )

        # ----------------------------------------------------
        # D. DETECT AFTER FILTERING
        # ----------------------------------------------------

        after = evaluate_signal(
            filtered_audio,
            sample_rate
        )

        # Report both stages
        print_metrics(
            condition,
            "Before filtering",
            before
        )

        print_metrics(
            condition,
            "After filtering",
            after
        )

        results.append({
            "condition": condition,
            "snr_db": snr_db,
            "before": before,
            "after": after,
        })

    # ========================================================
    # 8. SAVE SUMMARY CSV
    # ========================================================

    csv_path = OUTPUT_DIR / "noise_evaluation_summary.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Condition",
            "Stage",
            "Start threshold",
            "Continue threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
        ])

        for item in results:

            for stage in ["before", "after"]:

                result = item[stage]
                metrics = result["metrics"]

                writer.writerow([
                    item["condition"],
                    stage,
                    result["start_threshold"],
                    result["continue_threshold"],
                    metrics["TP"],
                    metrics["FP"],
                    metrics["FN"],
                    metrics["TN"],
                    metrics["Precision"],
                    metrics["Recall"],
                    metrics["F1"],
                    metrics["Accuracy"],
                ])

    print(f"\nSummary CSV saved to: {csv_path}")

    # ========================================================
    # 9. SAVE FRAME-LEVEL RESULTS
    # ========================================================

    frame_csv = OUTPUT_DIR / "noise_evaluation_frames.csv"

    with open(frame_csv, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Condition",
            "Stage",
            "Time_seconds",
            "RMS",
            "Reference",
            "Prediction",
        ])

        for item in results:

            for stage in ["before", "after"]:

                result = item[stage]

                for i, timestamp in enumerate(result["times"]):

                    writer.writerow([
                        item["condition"],
                        stage,
                        f"{timestamp:.4f}",
                        f"{result['rms'][i]:.8f}",
                        int(result["reference"][i]),
                        int(result["mask"][i]),
                    ])

    print(f"Frame-level CSV saved to: {frame_csv}")

    # ========================================================
    # 10. PLOT F1 SCORE COMPARISON
    # ========================================================

    labels = [item["condition"] for item in results]

    f1_before = [
        item["before"]["metrics"]["F1"]
        for item in results
    ]

    f1_after = [
        item["after"]["metrics"]["F1"]
        for item in results
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        f1_before,
        width,
        label="Before filtering"
    )

    ax.bar(
        x + width / 2,
        f1_after,
        width,
        label="After filtering"
    )

    ax.set_title(
        "Speech Detection Performance at Different SNR Levels"
    )

    ax.set_xlabel("Noise condition")
    ax.set_ylabel("Frame-level F1 score")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)

    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "noise_evaluation_f1.png"

    try:
        fig.savefig(
            str(plot_path),
            format="png",
            dpi=200,
            bbox_inches="tight"
        )
        print(f"F1 comparison plot saved to: {plot_path}")

    except OSError as error:
        print(f"Could not save F1 plot: {error}")

    plt.show()
    plt.close(fig)

    # ========================================================
    # 11. FINAL SUMMARY TABLE
    # ========================================================

    print("\n\nFINAL F1 SCORE SUMMARY")
    print("=" * 65)
    print(f"{'Condition':<15}{'Before':>12}{'After':>12}{'Change':>12}")

    for item in results:

        before_f1 = item["before"]["metrics"]["F1"]
        after_f1 = item["after"]["metrics"]["F1"]

        change = after_f1 - before_f1

        print(
            f"{item['condition']:<15}"
            f"{before_f1:>12.4f}"
            f"{after_f1:>12.4f}"
            f"{change:>+12.4f}"
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

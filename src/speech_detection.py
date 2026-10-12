
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import binary_closing, binary_opening

from filtering import filter_audio


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_MS = 25
HOP_MS = 10

# Hysteresis threshold settings
START_THRESHOLD_RATIO = 0.08
CONTINUE_THRESHOLD_RATIO = 0.035

NOISE_MULTIPLIER = 2.0
CONTINUE_NOISE_MULTIPLIER = 1.5

MIN_SPEECH_DURATION_MS = 100

# Morphological smoothing
CLOSING_FRAMES = 5
OPENING_FRAMES = 3


# ============================================================
# 2. SHORT-TIME RMS CALCULATION
# ============================================================

def calculate_rms(
    audio,
    sample_rate,
    frame_ms=FRAME_MS,
    hop_ms=HOP_MS
):
    """Calculate RMS energy for overlapping audio frames."""

    frame_length = int(sample_rate * frame_ms / 1000)
    hop_length = int(sample_rate * hop_ms / 1000)

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Invalid frame or hop duration.")

    if len(audio) < frame_length:
        raise ValueError("Audio is shorter than one analysis frame.")

    starts = np.arange(
        0,
        len(audio) - frame_length + 1,
        hop_length
    )

    frames = np.stack([
        audio[start:start + frame_length]
        for start in starts
    ])

    rms_values = np.sqrt(np.mean(frames ** 2, axis=1))

    # Center time of each analysis frame
    frame_times = (
        starts + frame_length / 2
    ) / sample_rate

    return rms_values, frame_times


# ============================================================
# 3. HYSTERESIS-BASED SPEECH DETECTION
# ============================================================

def detect_speech(audio, sample_rate):
    """
    Detect speech on the supplied audio.

    A higher threshold starts a speech region.
    A lower threshold keeps an existing region active
    through quieter frames.
    """

    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1:
        raise ValueError("Audio must be mono.")

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    rms_values, frame_times = calculate_rms(
        audio, sample_rate
    )

    # Estimate background energy from quieter frames
    noise_floor = np.percentile(rms_values, 20)
    peak_rms = np.max(rms_values)

    # Threshold required to start detecting speech
    start_threshold = max(
        noise_floor * NOISE_MULTIPLIER,
        peak_rms * START_THRESHOLD_RATIO
    )

    # Lower threshold for continuing speech
    continue_threshold = max(
        noise_floor * CONTINUE_NOISE_MULTIPLIER,
        peak_rms * CONTINUE_THRESHOLD_RATIO
    )

    # Ensure the continuation threshold does not exceed
    # the threshold required to start a speech region
    continue_threshold = min(
        continue_threshold,
        start_threshold
    )

    # Hysteresis state machine
    speech_mask = np.zeros(
        len(rms_values), dtype=bool
    )

    active = False

    for i, energy in enumerate(rms_values):

        if not active:
            if energy >= start_threshold:
                active = True
        else:
            if energy < continue_threshold:
                active = False

        speech_mask[i] = active

    # Fill short gaps in detected regions
    if len(speech_mask) >= CLOSING_FRAMES:
        speech_mask = binary_closing(
            speech_mask,
            structure=np.ones(CLOSING_FRAMES, dtype=bool)
        )

    # Remove isolated short detections
    if len(speech_mask) >= OPENING_FRAMES:
        speech_mask = binary_opening(
            speech_mask,
            structure=np.ones(OPENING_FRAMES, dtype=bool)
        )

    return (
        speech_mask,
        frame_times,
        rms_values,
        start_threshold,
        continue_threshold
    )


# ============================================================
# 4. SPEECH SEGMENT EXTRACTION
# ============================================================

def get_speech_segments(
    speech_mask,
    frame_times,
    hop_ms=HOP_MS,
    min_duration_ms=MIN_SPEECH_DURATION_MS
):
    """Extract speech intervals and reject very short segments."""

    if len(speech_mask) == 0:
        return []

    changes = np.diff(
        np.r_[False, speech_mask, False].astype(int)
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    hop_seconds = hop_ms / 1000
    min_duration = min_duration_ms / 1000

    segments = []

    for start, end in zip(starts, ends):

        start_time = float(frame_times[start])

        end_time = float(
            frame_times[end - 1] + hop_seconds
        )

        duration = end_time - start_time

        if duration >= min_duration:
            segments.append((
                max(0.0, start_time),
                end_time
            ))

    return segments


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_detection(
    original_audio,
    filtered_audio,
    sample_rate,
    frame_times,
    rms_values,
    start_threshold,
    continue_threshold,
    segments
):
    """Plot original audio, filtered audio, RMS and detections."""

    original_time = (
        np.arange(len(original_audio)) / sample_rate
    )

    filtered_time = (
        np.arange(len(filtered_audio)) / sample_rate
    )

    # Display only the accepted speech segments
    display_mask = np.zeros(
        len(frame_times), dtype=bool
    )

    for start, end in segments:
        display_mask |= (
            (frame_times >= start) &
            (frame_times < end)
        )

    fig, axes = plt.subplots(
        4, 1,
        figsize=(14, 11),
        sharex=True
    )

    # Original waveform
    axes[0].plot(
        original_time,
        original_audio,
        linewidth=0.6
    )

    axes[0].set_title("Original Audio Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Filtered waveform: actual detector input
    axes[1].plot(
        filtered_time,
        filtered_audio,
        linewidth=0.6
    )

    axes[1].set_title(
        "Filtered Audio Waveform (Detector Input)"
    )

    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    # RMS and hysteresis thresholds
    axes[2].plot(
        frame_times,
        rms_values,
        linewidth=0.8,
        label="RMS energy"
    )

    axes[2].axhline(
        start_threshold,
        color="red",
        linestyle="--",
        label=f"Start threshold = {start_threshold:.4f}"
    )

    axes[2].axhline(
        continue_threshold,
        color="green",
        linestyle="--",
        label=(
            "Continue threshold = "
            f"{continue_threshold:.4f}"
        )
    )

    axes[2].set_title("RMS Energy of Filtered Audio")
    axes[2].set_ylabel("RMS")
    axes[2].legend()
    axes[2].grid(True)

    # Final speech activity timeline
    axes[3].step(
        frame_times,
        display_mask.astype(int),
        where="mid"
    )

    axes[3].set_title("Speech Activity Detection")
    axes[3].set_ylabel("0 = No speech, 1 = Speech")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_yticks([0, 1])
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True)

    fig.tight_layout()

    plot_path = OUTPUT_DIR / "speech_detection.png"

    fig.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nPlot saved to: {plot_path}")


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

def main():

    # Load original audio
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    original_audio, sample_rate = sf.read(AUDIO_PATH)

    # Convert stereo audio to mono
    if original_audio.ndim == 2:
        original_audio = original_audio.mean(axis=1)

    original_audio = np.asarray(
        original_audio, dtype=np.float64
    )

    if len(original_audio) == 0:
        raise ValueError("Input audio is empty.")

    if not np.all(np.isfinite(original_audio)):
        raise ValueError("Audio contains invalid values.")

    # Filter original audio before detection
    filtered_audio = filter_audio(
        original_audio,
        sample_rate
    )

    # Save filtered audio for listening
    filtered_path = OUTPUT_DIR / "filtered_original.wav"

    sf.write(
        filtered_path,
        filtered_audio,
        sample_rate
    )

    # IMPORTANT: detector receives filtered_audio
    (
        speech_mask,
        frame_times,
        rms_values,
        start_threshold,
        continue_threshold
    ) = detect_speech(
        filtered_audio,
        sample_rate
    )

    # Extract accepted speech intervals
    segments = get_speech_segments(
        speech_mask,
        frame_times
    )

    duration = len(filtered_audio) / sample_rate

    # Print results
    print("\nFILTERED SPEECH DETECTION RESULTS")
    print("=" * 50)

    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Audio duration: {duration:.2f} seconds")
    print("Detector input: Band-pass-filtered audio")
    print("Filter: Butterworth, 100-4000 Hz")

    print(f"Start threshold: {start_threshold:.6f}")
    print(f"Continue threshold: {continue_threshold:.6f}")

    print(f"Analysis frames: {len(rms_values)}")
    print(f"Speech frames before duration filtering: "
          f"{np.sum(speech_mask)}")

    print(f"Accepted speech segments: {len(segments)}")

    if segments:
        print("\nDETECTED SPEECH INTERVALS")
        print("-" * 40)

        for i, (start, end) in enumerate(segments, 1):
            print(
                f"Segment {i}: {start:.2f}s - {end:.2f}s "
                f"(duration: {end - start:.2f}s)"
            )
    else:
        print("\nNo speech segments detected.")

    print(f"\nFiltered audio saved to: {filtered_path}")

    # Plot all results
    plot_detection(
        original_audio,
        filtered_audio,
        sample_rate,
        frame_times,
        rms_values,
        start_threshold,
        continue_threshold,
        segments
    )


if __name__ == "__main__":
    main()

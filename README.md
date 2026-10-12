# DSP-Based Speech Detection in Noisy Environments

A Python-based Digital Signal Processing (DSP) project that detects speech segments in audio recordings using RMS energy analysis, Butterworth bandpass filtering, and noise robustness evaluation.

## Overview

Speech detection identifies portions of an audio signal that contain speech while distinguishing them from silence and background noise. This project implements an energy-based detection system and evaluates how filtering affects its performance under different noise conditions.

## Features

- Audio loading and conversion from stereo to mono
- Butterworth bandpass filtering (100–4000 Hz)
- Short-time RMS energy calculation
- Adaptive thresholds with hysteresis for speech detection
- Minimum-duration filtering and segment smoothing
- Evaluation using precision, recall, F1-score, and accuracy
- Noise robustness testing at different signal-to-noise ratios (SNRs)
- Visualization and CSV export of evaluation results

## Methodology

1. **Audio preprocessing:** Load the input WAV file and convert stereo audio to mono when necessary.
2. **Bandpass filtering:** Apply a Butterworth bandpass filter to retain frequencies in the 100–4000 Hz range.
3. **Frame-based analysis:** Calculate RMS energy using 25 ms frames and a 10 ms hop.
4. **Speech detection:** Use separate start and continuation thresholds with hysteresis to reduce rapid switching between speech and silence.
5. **Post-processing:** Smooth the detected regions and reject segments shorter than the minimum duration.
6. **Performance evaluation:** Compare detected speech against reference intervals and calculate standard classification metrics.
7. **Noise evaluation:** Test performance on clean audio and audio corrupted with white Gaussian noise at different SNR levels.

## Experimental Results

The following results were obtained using the current reference intervals and evaluation configuration.

| Condition | F1 Before Filtering | F1 After Filtering |
|---|---:|---:|
| Clean audio | 77.91% | 86.90% |
| 20 dB SNR | 77.25% | 86.90% |
| 10 dB SNR | 66.22% | 84.81% |
| 5 dB SNR | 36.90% | 81.35% |

At 5 dB SNR, bandpass filtering improved the F1-score by 44.45 percentage points in this experiment.

**Evaluation note:** The reference speech intervals are provisional and should be verified against the original recording. Results depend on the reference labels, audio sample, and detection parameters.

## Project Structure

```text
DSP-Speech-Detection/
├── audio/
│   └── original.wav
├── src/
│   ├── filtering.py
│   ├── speech_detection.py
│   ├── evaluate_detection.py
│   └── noise_evaluation.py
├── output/
│   ├── noise_evaluation_summary.csv
│   ├── noise_evaluation_frames.csv
│   └── noise_evaluation_f1.png
├── .gitignore
└── README.md
```

*The output filenames shown above are expected artifacts from the evaluation script; actual files may vary depending on the scripts you run.*

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- Pandas

Install the dependencies in your virtual environment:

```bash
pip install numpy scipy matplotlib pandas
```

## How to Run

Run these commands from the project root directory.

### Run speech detection

```bash
python src/speech_detection.py
```

### Evaluate speech detection

```bash
python src/evaluate_detection.py
```

### Evaluate noise robustness

```bash
python src/noise_evaluation.py
```

Check the `output/` directory for generated plots and evaluation reports.

## Evaluation Metrics

- **Precision:** Proportion of detected speech frames that correspond to reference speech.
- **Recall:** Proportion of reference speech frames successfully detected.
- **F1-score:** Harmonic mean of precision and recall.
- **Accuracy:** Proportion of correctly classified speech and non-speech frames.

## Technologies Used

Python · NumPy · SciPy · Matplotlib · Pandas · Digital Signal Processing

## Future Improvements

- Validate reference speech intervals through manual annotation.
- Compare RMS-based detection with zero-crossing rate and other voice activity detection methods.
- Tune thresholds using a separate validation recording.
- Test on recordings with real-world background noise.
- Evaluate performance across multiple speakers and audio samples.

## Author

**Sasmit Bansod**

Developed as a Digital Signal Processing project exploring energy-based speech detection and noise robustness.


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from scipy.fft import rfft, rfftfreq

# Project paths
ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = ROOT / "audio" / "original.wav"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load audio
audio, sample_rate = sf.read(AUDIO_PATH)

# Convert stereo audio to mono if required
if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = audio.astype(np.float64)

if len(audio) == 0:
    raise ValueError("The audio file is empty.")

if not np.all(np.isfinite(audio)):
    raise ValueError("Audio contains invalid numerical values.")

duration = len(audio) / sample_rate
time = np.arange(len(audio)) / sample_rate

print("Audio loaded successfully")
print(f"Sampling rate: {sample_rate} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Number of samples: {len(audio)}")
print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")

# 2. Plot time-domain waveform
plt.figure(figsize=(12, 4))
plt.plot(time, audio, linewidth=0.7)
plt.title("Original Speech Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "original_waveform.png", dpi=200)
plt.show()

# 3. Calculate and plot frequency spectrum using FFT
spectrum = np.abs(rfft(audio))
frequencies = rfftfreq(len(audio), d=1 / sample_rate)

# Normalize spectrum for visualization
spectrum = spectrum / max(np.max(spectrum), 1e-12)

plt.figure(figsize=(12, 4))
plt.plot(frequencies, spectrum, linewidth=0.7)
plt.title("Frequency Spectrum of Speech")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.xlim(0, min(sample_rate / 2, 8000))
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "frequency_spectrum.png", dpi=200)
plt.show()

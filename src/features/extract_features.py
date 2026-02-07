import os
import sys
import numpy as np
import librosa
from collections import Counter

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Paths (relative to project root)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Audio settings
SR = 22050
WINDOW_DURATION = 3  # seconds
WINDOW_SAMPLES = SR * WINDOW_DURATION  # 66150
N_MFCC = 40

X = []
y = []

# Loop through each class directory
for class_name in os.listdir(RAW_DATA_DIR):
    class_path = os.path.join(RAW_DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)

        try:
            # Load full audio file
            signal, sr = librosa.load(file_path, sr=SR)

            # Calculate number of full 3s windows
            num_windows = len(signal) // WINDOW_SAMPLES

            for i in range(num_windows):
                start = i * WINDOW_SAMPLES
                end = start + WINDOW_SAMPLES
                window = signal[start:end]

                # Discard incomplete windows explicitly
                if len(window) < WINDOW_SAMPLES:
                    continue

                # Extract MFCCs
                mfcc = librosa.feature.mfcc(
                    y=window,
                    sr=sr,
                    n_mfcc=N_MFCC
                )

                X.append(mfcc)
                y.append(class_name)

        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Save processed features
np.save(os.path.join(PROCESSED_DATA_DIR, "X.npy"), X)
np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), y)

# Print stats
print("✅ Advanced feature extraction completed")
print("X shape:", X.shape)

class_counts = Counter(y)
print("Samples per class:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

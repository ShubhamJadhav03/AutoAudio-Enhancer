import os
import random
from pydub import AudioSegment

SOURCE = "LibriSpeech/dev-clean"
TARGET = "data/raw/speech"

os.makedirs(TARGET, exist_ok=True)

# Collect all FLAC files
flac_files = []

for root, _, files in os.walk(SOURCE):
    for file in files:
        if file.endswith(".flac"):
            flac_files.append(os.path.join(root, file))

print(f"Found {len(flac_files)} FLAC files")

# Select samples
NUM_FILES = 15  # good starting point
selected_files = random.sample(flac_files, NUM_FILES)

# Convert to WAV
for i, file_path in enumerate(selected_files):
    audio = AudioSegment.from_file(file_path, format="flac")

    # Normalize for ML
    audio = audio.set_frame_rate(22050).set_channels(1)

    output_path = os.path.join(TARGET, f"speech_{i}.wav")
    audio.export(output_path, format="wav")

print("✅ Speech dataset prepared from LibriSpeech")

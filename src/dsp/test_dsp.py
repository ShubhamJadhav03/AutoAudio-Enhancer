import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.dsp.audio_profiles import AudioProfiles
import librosa
import soundfile as sf
import numpy as np

# Load a test file (Use one of your music files)
# CHANGE THIS PATH to a real file you have!
# Get project root directory (two levels up from this file)
project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../'))
input_file = os.path.join(project_root, "data/raw/gaming/test.wav") 

try:
    print(f"🎧 Loading {input_file}...")
    audio, sr = librosa.load(input_file, sr=44100, duration=10.0) # Load 10 seconds

    processor = AudioProfiles(sample_rate=sr)

    print("🎛️ Applying 'Gaming' Profile (High compression)...")
    processed_audio = processor.apply_profile(audio, "gaming")

    output_file = os.path.join(project_root, "data/processed/test_gaming_output.wav")
    sf.write(output_file, processed_audio, sr)
    print(f"✅ Saved to {output_file}. Go listen to it!")

except Exception as e:
    print(f"❌ Error: {e}")
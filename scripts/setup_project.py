import os

# Project folder structure
folders = [
    "data/raw/music",
    "data/raw/speech",
    "data/raw/gaming",
    "data/raw/movie",
    "data/processed",
    "models",
    "src/features",
    "src/utils"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# requirements.txt content
requirements = [
    "numpy",
    "pandas",
    "librosa",
    "soundfile",
    "scipy",
    "pydub",
    "sounddevice",
    "torch",
    "pedalboard"
]

# Write requirements.txt
with open("requirements.txt", "w") as f:
    for req in requirements:
        f.write(req + "\n")

print("✅ Project Structure Created")

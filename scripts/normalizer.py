from pydub import AudioSegment
import os

def normalize_audio(folder):
    for file in os.listdir(folder):
        if file.endswith(".mp3") or file.endswith(".wav"):
            audio = AudioSegment.from_file(os.path.join(folder, file))
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(
                os.path.join(folder, file.replace(".mp3", ".wav")),
                format="wav"
            )

folders = [
    "data/raw/music",
    "data/raw/speech",
    "data/raw/gaming",
    "data/raw/movie"
]

for folder in folders:
    normalize_audio(folder)

print("✅ Audio normalized")

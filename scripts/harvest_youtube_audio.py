import subprocess
import os
import sys

def download_audio(urls, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        command = [
            sys.executable, "-m", "yt_dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", f"{output_dir}/%(title)s.%(ext)s",
            url
        ]
        subprocess.run(command)

# 🔫 Gaming (no commentary)
movie_urls = [
    

    "https://youtu.be/JUE2gIWLOGA?si=bg15s4td6eONxiy6"

]

# 🎬 Movies


download_audio(movie_urls, "data/raw/movie")

print("✅ YouTube audio harvested")

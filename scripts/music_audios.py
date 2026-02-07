import kagglehub
import os
import random
import shutil

# 1️⃣ Download & cache dataset (automatic)
dataset_path = kagglehub.dataset_download(
    "andradaolteanu/gtzan-dataset-music-genre-classification"
)

print("Dataset cached at:", dataset_path)

# 2️⃣ GTZAN audio path inside dataset
SOURCE = os.path.join(dataset_path, "Data", "genres_original")
if not os.path.exists(SOURCE):
    SOURCE = os.path.join(dataset_path, "genres_original")

if not os.path.exists(SOURCE):
    raise FileNotFoundError(f"Could not find 'genres_original' in {dataset_path}. Contents: {os.listdir(dataset_path)}")

TARGET = "data/raw/music"

os.makedirs(TARGET, exist_ok=True)

# 3️⃣ Choose genres you care about
genres = ["pop", "rock", "metal", "disco", "hiphop"]

FILES_PER_GENRE = 20  # adjust if needed

for genre in genres:
    genre_path = os.path.join(SOURCE, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]

    selected = random.sample(files, FILES_PER_GENRE)

    for file in selected:
        src = os.path.join(genre_path, file)
        dst = os.path.join(TARGET, f"{genre}_{file}")
        shutil.copy(src, dst)

print("✅ Music dataset prepared from Kaggle GTZAN")

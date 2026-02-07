## AutoAudio-Enhancer

Audio classification and enhancement project for gaming, music, speech, and movie audio.

## Project Structure

```
AutoAudio-Enhancer/
├── scripts/              # Data collection and setup scripts
│   ├── setup_project.py
│   ├── harvest_youtube_audio.py
│   ├── music_audios.py
│   ├── speech.py
│   └── normalizer.py
├── src/
│   ├── features/        # Feature extraction
│   │   └── extract_features.py
│   ├── models/          # Model training
│   │   └── train_model.py
│   └── utils/           # Utility functions
├── data/
│   ├── raw/            # Raw audio files
│   └── processed/      # Processed features
├── datasets/           # External datasets (e.g., dev-clean)
├── models/            # Trained model files
└── venv/              # Virtual environment
```

## Phase 1 – Project Setup & Data Collection

1. Create project structure:
    ```bash
    python scripts/setup_project.py
    ```

2. Prepare music dataset (GTZAN):
    ```bash
    python scripts/music_audios.py
    ```

3. Prepare speech dataset (LibriSpeech):
    ```bash
    python scripts/speech.py
    ```

4. Harvest YouTube audio:
    ```bash
    python scripts/harvest_youtube_audio.py
    ```

5. Normalize audio files:
    ```bash
    python scripts/normalizer.py
    ```

## Phase 2 – Feature Extraction & Training

1. Extract features:
    ```bash
    python src/features/extract_features.py
    ```

2. Train model:
    ```bash
    python src/models/train_model.py
    ```
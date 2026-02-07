# AutoAudio-Enhancer

An intelligent audio classification and real-time enhancement system that automatically detects audio content type (gaming, music, speech, movie) and applies optimized DSP profiles for the best listening experience.

## Features

- **Real-time Audio Classification**: Uses a CNN model to classify audio into 4 categories (gaming, music, speech, movie)
- **Automatic DSP Enhancement**: Applies optimized audio profiles based on detected content type
- **Rich TUI Interface**: Beautiful terminal UI using the `rich` library
- **Low Latency**: Real-time audio processing with minimal delay
- **Standalone Executable**: Can be built as a Windows `.exe` for easy distribution

## Project Structure

```
AutoAudio-Enhancer/
├── main.py              # Main application entry point
├── AutoAudio.spec       # PyInstaller specification file
├── requirements.txt     # Python dependencies
├── scripts/             # Data collection and setup scripts
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
│   ├── utils/           # Utility functions
│   │   └── audio_profiles.py
│   └── dsp/             # DSP processing
│       └── audio_profiles.py
├── data/
│   ├── raw/            # Raw audio files
│   └── processed/      # Processed features
├── datasets/           # External datasets (e.g., dev-clean)
├── models/            # Trained model files
│   ├── audio_cnn.pth
│   └── label_encoder.npy
└── dist/              # Built executables (after PyInstaller build)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Windows 10/11 (for executable build)
- Virtual audio cable software (e.g., VB-Audio Cable) for audio routing

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AutoAudio-Enhancer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install rich  # If not in requirements.txt
   ```

## Usage

### Running the Application

1. Ensure you have a virtual audio cable installed and configured
2. Run the main application:
   ```bash
   python main.py
   ```

3. When prompted, enter:
   - **Input ID**: Your virtual audio cable output device ID
   - **Output ID**: Your headphones/speakers device ID
   
   > **Note**: Both devices must use the same API (MME/DirectSound). Recommended: Input=4 (MME), Output=7 (MME)

4. The application will:
   - Display a real-time TUI showing the detected audio type
   - Automatically apply DSP enhancements based on classification
   - Show confidence scores and smoothing information

### Building a Standalone Executable

To create a Windows `.exe` file:

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Build the executable:
   ```bash
   python -m PyInstaller AutoAudio.spec
   ```

3. The executable will be created in `dist/AutoAudio.exe`

4. Run the executable:
   ```bash
   dist\AutoAudio.exe
   ```

**Note**: The executable includes all dependencies and model files, so it can run on any Windows machine without Python installed.

## Development

### Phase 1 – Project Setup & Data Collection

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

### Phase 2 – Feature Extraction & Training

1. Extract features:
   ```bash
   python src/features/extract_features.py
   ```

2. Train model:
   ```bash
   python src/models/train_model.py
   ```

## Dependencies

- **torch**: Deep learning framework for the CNN model
- **librosa**: Audio feature extraction (MFCC)
- **sounddevice**: Real-time audio I/O
- **pedalboard**: Audio DSP processing
- **rich**: Terminal UI framework
- **numpy**: Numerical computations
- **sklearn**: Label encoding and data utilities

## Technical Details

- **Model**: CNN-based audio classifier (AudioCNN_Pro)
- **Features**: 40 MFCC coefficients
- **Sample Rate**: 22050 Hz
- **Block Size**: 2048 samples
- **Prediction Interval**: 0.5 seconds
- **Smoothing Window**: 5 frames

## Troubleshooting

### Virtual Environment Issues

If you encounter path errors with the virtual environment:
```bash
# Use system Python directly
python -m pip install <package>
```

### Missing Model Files

Ensure `models/audio_cnn.pth` and `models/label_encoder.npy` exist before running the application.

### Audio Device Issues

- Verify both input and output devices use the same API
- Check device IDs using the list displayed at startup
- Ensure virtual audio cable is properly installed

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
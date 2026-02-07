import os
import sys
import numpy as np
import torch
import librosa
import sounddevice as sd
import threading
import time
from collections import deque, Counter

# --- UI IMPORTS ---
try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.align import Align
    from rich.text import Text
    from rich.console import Console
except ImportError:
    print("❌ MISSING LIBRARY: Run 'pip install rich' first.")
    sys.exit(1)

# --- PROJECT IMPORTS ---
try:
    from src.models.train_model import AudioCNN_Pro
    from src.utils.audio_profiles import AudioProfiles
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIG ---
# Handle PyInstaller bundled paths
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    base_path = sys._MEIPASS
else:
    # Running as script
    base_path = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base_path, "models", "audio_cnn.pth")
LABEL_ENCODER_PATH = os.path.join(base_path, "models", "label_encoder.npy")
SAMPLE_RATE = 22050 
BLOCK_SIZE = 2048 
PREDICTION_INTERVAL = 0.5
SMOOTHING_WINDOW = 5

# --- SHARED STATE ---
brain_buffer = np.zeros(int(SAMPLE_RATE * 3), dtype=np.float32) 
current_state = "waiting" 
confidence_score = 0.0
lock = threading.Lock()
prediction_history = deque(maxlen=SMOOTHING_WINDOW)

# --- SETUP ---
console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Brain
if not os.path.exists(LABEL_ENCODER_PATH): sys.exit("❌ Encoder missing.")
classes = np.load(LABEL_ENCODER_PATH)
le_classes = classes.tolist()

model = AudioCNN_Pro(num_classes=len(classes)).to(device)
if not os.path.exists(MODEL_PATH): sys.exit("❌ Model missing.")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load Voice
dsp = AudioProfiles(sample_rate=SAMPLE_RATE)

def callback(indata, outdata, frames, time, status):
    """Audio Passthrough & DSP Processing"""
    global brain_buffer
    
    # 1. Update Brain Buffer (Mono)
    if indata.shape[1] > 1: mono_input = np.mean(indata, axis=1)
    else: mono_input = indata[:, 0]

    with lock:
        shift = len(mono_input)
        brain_buffer = np.roll(brain_buffer, -shift)
        brain_buffer[-shift:] = mono_input

    # 2. Apply DSP (Stereo)
    try:
        # Transpose for pedalboard
        input_T = indata.T 
        # Apply current profile
        processed_T = dsp.apply_profile(input_T, current_state)
        # Transpose back
        outdata[:] = processed_T.T
    except:
        outdata[:] = indata # Fail-safe passthrough

def brain_loop():
    """AI Thread"""
    global current_state, confidence_score
    while True:
        time.sleep(PREDICTION_INTERVAL)
        
        with lock: snapshot = brain_buffer.copy()
        
        # Silence Check
        rms = np.sqrt(np.mean(snapshot**2))
        if rms < 0.005:
            current_state = "silence"
            continue

        # MFCC
        mfcc = librosa.feature.mfcc(y=snapshot, sr=SAMPLE_RATE, n_mfcc=40)
        if mfcc.shape[1] > 130: mfcc = mfcc[:, :130]
        else: mfcc = np.pad(mfcc, ((0,0), (0, 130 - mfcc.shape[1])))
            
        inp = torch.tensor(mfcc[np.newaxis, np.newaxis, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(inp)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            raw_pred = le_classes[pred.item()]
            
            prediction_history.append(raw_pred)
            
            # Vote
            counts = Counter(prediction_history)
            winner, count = counts.most_common(1)[0]
            
            if count >= 3:
                current_state = winner
                confidence_score = conf.item()

def make_layout():
    """Define the UI Grid"""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    return layout

def update_ui(layout):
    """Update UI components"""
    # Header
    layout["header"].update(
        Panel(Align.center("[bold cyan]AUTO AUDIO ENHANCER v1.0[/]"), style="blue")
    )
    
    # Main Display (Big State)
    color_map = {
        "music": "magenta",
        "gaming": "green",
        "movie": "yellow",
        "speech": "cyan",
        "silence": "dim white",
        "waiting": "white"
    }
    color = color_map.get(current_state, "white")
    
    state_text = Text(current_state.upper(), style=f"bold {color}", justify="center")
    state_text.stylize(f"bold {color}", 0, len(current_state))
    
    main_panel = Panel(
        Align.center(
            f"\n\n{state_text}\n\n[dim]Confidence: {confidence_score:.2f}[/]",
            vertical="middle"
        ),
        title="[b]Current Acoustic Environment[/]",
        border_style=color
    )
    layout["main"].update(main_panel)
    
    # Footer
    layout["footer"].update(
        Panel(Align.center(f"[dim]DSP Profile: Active | Smoothing: {SMOOTHING_WINDOW} frames[/]"), style="grey50")
    )

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print("[bold yellow]🎧 AUDIO CONFIGURATION[/]")
    print(sd.query_devices())
    
    print("\n[bold red]⚠️  CRITICAL: PAIR MUST MATCH API (MME/DirectSound)[/]")
    print("Recommended: Input=4 (MME), Output=7 (MME)")
    
    try:
        in_id = int(input("👉 Input ID (Cable Output): "))
        out_id = int(input("👉 Output ID (Headphones): "))
        
        # Start Brain
        t = threading.Thread(target=brain_loop, daemon=True)
        t.start()
        
        # Start Stream & UI
        stream = sd.Stream(device=(in_id, out_id),
                           samplerate=SAMPLE_RATE,
                           blocksize=BLOCK_SIZE,
                           channels=2,
                           callback=callback)
        
        with stream:
            layout = make_layout()
            with Live(layout, refresh_per_second=4, screen=True):
                while True:
                    update_ui(layout)
                    time.sleep(0.25)
                    
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
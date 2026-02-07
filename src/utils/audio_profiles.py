from pedalboard import Pedalboard, Compressor, Gain, HighpassFilter, LowShelfFilter, HighShelfFilter
import numpy as np

class AudioProfiles:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        
        # Music: Bass Boost + Treble Sparkle
        self.music_board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=6),  # Bass Boost
            HighShelfFilter(cutoff_frequency_hz=10000, gain_db=3), # Treble Boost
            Gain(gain_db=2)
        ])

        #  Speech: Remove rumble + Even out volume
        self.speech_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80), # Kill low rumble
            Compressor(threshold_db=-20, ratio=4),  # Make voice consistent
            Gain(gain_db=4)
        ])

        # Gaming: Hear everything (Heavy Compression)
        self.gaming_board = Pedalboard([
            HighShelfFilter(cutoff_frequency_hz=4000, gain_db=4), # Boost footsteps
            Compressor(threshold_db=-25, ratio=5, attack_ms=5),   # Aggressive compression
            Gain(gain_db=3)
        ])

        #  Movie: Dynamic + Loud
        self.movie_board = Pedalboard([
            Compressor(threshold_db=-15, ratio=2), # Light glue
            LowShelfFilter(cutoff_frequency_hz=200, gain_db=4), # Rumble
            Gain(gain_db=5)
        ])

    def apply_profile(self, audio_chunk, profile_name):
        """
        Input: audio_chunk (numpy array), profile_name (str)
        Output: processed_audio (numpy array)
        """
        # Ensure audio is float32 for Pedalboard
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        if profile_name == "music":
            return self.music_board(audio_chunk, self.sample_rate)
        elif profile_name == "speech":
            return self.speech_board(audio_chunk, self.sample_rate)
        elif profile_name == "gaming":
            return self.gaming_board(audio_chunk, self.sample_rate)
        elif profile_name == "movie":
            return self.movie_board(audio_chunk, self.sample_rate)
        else:
            return audio_chunk # Bypass
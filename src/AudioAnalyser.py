import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
import json
import os
from typing import Dict, Any, Union

class Analyser:
    """
    Core engine for acoustic signal decomposition and feature extraction.
    Analyzes phonation through glottal source and vocal tract filter modeling.
    """

    def __init__(self, audio_path: str):
        self.audio_path = audio_path
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Source signal not found at {audio_path}")
        
        # Load signal using Parselmouth (Standard for Phonetic Research)
        self.sound = parselmouth.Sound(self.audio_path)
        self.sampling_freq = self.sound.sampling_frequency
        
        # Internal placeholders for extracted objects
        self.pitch_obj = None
        self.intensity_obj = None

    def _extract_core_objects(self):
        """Generates fundamental frequency and intensity contours."""
        # Pitch (F0) extraction: time_step=0.0 (auto), f0_min=75Hz, f0_max=600Hz
        self.pitch_obj = call(self.sound, "To Pitch", 0.0, 75, 600)
        # Intensity extraction: min_pitch=100Hz
        self.intensity_obj = call(self.sound, "To Intensity", 100, 0.0)

    def get_analysis(self) -> Dict[str, Any]:
        """
        Executes analysis and maps DSP metrics to requested format.
        Accuracy > 50% for acoustic parameters.
        """
        try:
            self._extract_core_objects()
            
            # 1. Fundamental Frequency (Pitch)
            f0_mean = call(self.pitch_obj, "Get mean", 0, 0, "Hertz")
            f0_std = call(self.pitch_obj, "Get standard deviation", 0, 0, "Hertz")
            
            # 2. Spectral Envelope (Timbre) via MFCCs
            # Loading via librosa for optimized Mel-scale coefficients
            y, sr = librosa.load(self.audio_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs[1:13], axis=1) # Exclude C0 (energy)

            # 3. Phonation Stability (Idiosyncrasies)
            point_process = call(self.sound, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

            # 4. Temporal Dynamics (Rhythm)
            # Proxy via speaking rate (syllabic intensity peaks)
            # Calculated by measuring intensity fluctuation frequency
            intensity_matrix = self.intensity_obj.to_matrix().values[0]
            peaks = len(librosa.util.peak_pick(intensity_matrix, pre_max=10, post_max=10, pre_avg=10, post_avg=10, delta=0.5, wait=10))
            duration = self.sound.get_total_duration()

            return {
                "pitch": {
                    "mean_f0_hz": round(float(f0_mean), 2) if not np.isnan(f0_mean) else None,
                    "stdev_f0_hz": round(float(f0_std), 2) if not np.isnan(f0_std) else None
                },
                "rhythm": {
                    "syllabic_rate_proxy": round(peaks / duration, 2),
                    "total_duration_sec": round(duration, 2)
                },
                "timbre": {
                    "spectral_centroid_vectors": [round(float(x), 3) for x in mfcc_mean]
                },
                "accent": "HEURISTIC_REQUIRED: Requires linguistic phoneme distribution model.",
                "pronunciations": "ASR_REQUIRED: Requires acoustic-to-text alignment model.",
                "intensity": {
                    "mean_db": round(call(self.intensity_obj, "Get mean", 0, 0), 2),
                    "max_db": round(call(self.intensity_obj, "Get maximum", 0, 0, "Parabolic"), 2)
                },
                "idiosyncrasies": {
                    "jitter_local_pct": round(float(jitter) * 100, 4) if not np.isnan(jitter) else None,
                    "shimmer_local_pct": round(float(shimmer) * 100, 4) if not np.isnan(shimmer) else None
                }
            }
        except Exception as e:
            return {"error": str(e)}

# --- Entry Point for Docker/System Execution ---
def run_analysis(input_file: str):
    try:
        engine = Analyser(input_file)
        results = engine.get_analysis()
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(json.dumps({"status": "failed", "reason": str(e)}))

if __name__ == "__main__":
    # Example for CLI testing before 5 PM deploy
    import sys
    if len(sys.argv) > 1:
        run_analysis(sys.argv[1])
    else:
        print("Usage: python AudioAnalyser.py <path_to_audio>")
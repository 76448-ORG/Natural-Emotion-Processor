import numpy as np
import json
from collections import namedtuple as ntuple
from AudioAnalyser import Analyser as AudioAnalyser
from VideoAnalyser import Analyser as VideoAnalyser
from TextAnalyser import Analyser as TextAnalyser


# --- Core Data Structures ---
EToken = ntuple(
    'EToken', 
    ['logical_schema', 'emotion_schema', 'base_abstract', 'current_abstract']
)


class Abstracts:
    AudioAbstract = ntuple(
        'AudioAbstract',
        ["pitch", "rhythm", "timbre", "accent", "pronunciations", "intensity", "cords", "idiosyncrasies", "melody"],
        defaults=[None] * 9
    )

    VideoAbstract = ntuple(
        'VideoAbstract',
        [],
        defaults=[]
    )

    TextAbstract = ntuple(
        'TextAbstract',
        ["vocabulary-diversity", "word-preferences", "slang-rate", "Idiom-preferences", "Contraction-rate", 
        "capitalization-ratio", "stop-words-rate", "punctuation-frequency", "typos-rate", "emoji-rate", 
        "salutation-rate", "quoting-style", "abbreviation-rate"],
        defaults=[None] * 13
    )


class DifferentialEngine:
    """NEP Differential Engine for quantifying Affective Deviation (ΔA)."""
    
    @staticmethod
    def compute_delta(current, base):
        """Calculates scalar/vector deviation from established baseline."""
        if not current or not base: return {}
        delta = {}
        for field in current._fields:
            c_val, b_val = getattr(current, field), getattr(base, field)
            
            # Numeric Feature Deviation
            if isinstance(c_val, (int, float)) and isinstance(b_val, (int, float)):
                delta[field] = round(c_val - b_val, 4)
            
            # Recursive Analysis for Nested Biometric Objects (Intensity/Pitch/Timbre)
            elif isinstance(c_val, dict) and isinstance(b_val, dict):
                delta[field] = {k: round(c_val[k] - b_val[k], 4) 
                                for k in c_val if isinstance(c_val[k], (int, float)) and k in b_val}
        return delta


def extract_etoken(text=None, audio_path=None, video_path=None, base_profiles=None):
    """
    NEP Entry Point: Decouples Logical Content (S') and generates the Emotional Schema.
    """
    # 1. Baseline Provisioning
    base = base_profiles or {
        "text": Abstracts.TextAbstract(),
        "audio": Abstracts.AudioAbstract(),
        "video": Abstracts.VideoAbstract()
    }

    # 2. Real-Time Feature Extraction (Current Abstract)
    curr_text = Abstracts.TextAbstract(**TextAnalyser(text).get_analysis()) if text else None
    curr_audio = None
    if audio_path:
        raw_audio = AudioAnalyser(audio_path).get_analysis()
        filtered = {k: v for k, v in raw_audio.items() if k in Abstracts.AudioAbstract._fields}
        curr_audio = Abstracts.AudioAbstract(**filtered)

    # 3. Affective Deviation (ΔA) Calculation
    delta_a = {
        "text_deviation": DifferentialEngine.compute_delta(curr_text, base["text"]),
        "audio_deviation": DifferentialEngine.compute_delta(curr_audio, base["audio"])
    }

    # 4. Packaging for JAST-V Projection
    # Logical Schema (S'): Minimal filtered content for Rational Policy (πlog)
    logical_schema = text if text else ""
    
    # Emotional Schema: ΔA Vector prefix + Unfiltered content for XLM-R semantic analysis
    delta_prefix = f"[DELTA_A: {json.dumps(delta_a)}]"
    emotion_schema = f"{delta_prefix} {text}" if text else delta_prefix

    return EToken(
        logical_schema=logical_schema,
        emotion_schema=emotion_schema,
        base_abstract=base,
        current_abstract={"text": curr_text, "audio": curr_audio}
    )

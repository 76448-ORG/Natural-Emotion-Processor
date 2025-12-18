# NEP Core: Natural Emotion Processor

**Module of the GPR-Framework (Generative & Programmable Response Framework)**

### Overview

NEP is a high-precision diagnostic engine designed to quantify **Affective Deviation (\Delta A)** by decoupling raw input signals into **Logical Content (S')** and **Emotional Schemas**. It utilizes a **Dual-Translation Emotive Bridge** to map user states onto a 4D **JAST-V** bipolar space, providing the necessary data for **XLM-RoBERTa** semantic analysis and **HITL-Reinforcement Learning** loops.

### Core Architecture

* **Differential Engine**: Computes the Z-score/scalar variance between the `Current Abstract` (real-time biometric data) and the `Base Abstract` (user's neutral/habitual profile).
* **E-Token Generator**: Packages the logical schema and the emotional deviation vector into a dense token for Transformer-based policy decisions.
* **Multimodal Analyzers**:
* **TextAnalyser**: 14 stylometric factors (lexical diversity, punctuation frequency, contraction rates).
* **AudioAnalyser**: Acoustic signal decomposition using Parselmouth/Librosa (Pitch/F0, Jitter, Shimmer, Spectral Centroids).
* **VideoAnalyser**: (Placeholder) Future integration for micro-expression and facial action unit (FAU) tracking.

### File Structure

* `main.py`: Entry point; contains the `DifferentialEngine` and `EToken` packaging logic.
* `TextAnalyser.py`: Psycholinguistic diagnostic tool.
* `AudioAnalyser.py`: DSP engine for glottal source and vocal tract filter modeling.
* `WhiteList.dict`: FOSS/Technical terminology whitelist to prevent false-positive "typo" or "slang" detection in technical discourse.

### E-Token Specification

The `EToken` namedtuple consists of:

1. `logical_schema`: The raw semantic data (S').
2. `emotion_schema`: S' prefixed with the quantified \Delta A vector.
3. `base_abstract`: The reference neutral profile.
4. `current_abstract`: The momentary feature set.

### Usage

```python
from NEP import tokeniser

# Example: Process text and audio input
e_token = tokeniser(
    text="The Arch Linux kernel is performing within parameters, brother.",
    audio_path="input_signal.wav",
    base_profiles=my_stored_baseline
)

# Output for XLM-R / GPRF Decision Layer
print(e_token.emotion_schema)
```

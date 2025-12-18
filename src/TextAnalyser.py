import re
import json
import string
import os
from collections import Counter
from typing import Dict, Any, List

# --- Dependency Integration ---
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz

# --- NLTK Resource Provisioning ---
def _setup_nltk():
    resources = ['stopwords', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}' if res == 'stopwords' else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

_setup_nltk()

class Analyser:
    """
    Main execution class for psycholinguistic and stylometric text diagnostics.
    """
    _STOP_WORDS = set(stopwords.words('english'))
    _PUNC_CHARS = string.punctuation
    _SHORTHAND_FILE = "./WhiteList.dict"
    
    # Heuristic Lexical Databases
    _CONTRACTIONS = {"i'm", "you're", "he's", "she's", "it's", "we're", "they're", "isn't", "aren't", "don't", "doesn't", "can't", "won't"}
    _SLANG = {"lol", "omg", "imo", "tldr", "fomo", "yolo", "tbh", "ikr", "vibe", "btw"}
    _IDIOMS = ["piece of cake", "break a leg", "spill the beans", "bite the bullet", "cost an arm and a leg"]
    _PHONETIC_TARGETS = {"you", "are", "see", "night", "great", "later", "for", "be"}

    def __init__(self, text: str):
        self.raw = text
        self.text_lower = text.lower()
        self.whitelist = self._load_whitelist()
        
        # Tokenization via NLTK punkt
        all_tokens = word_tokenize(self.text_lower)
        self.words = [t for t in all_tokens if t.isalnum()]
        self.word_count = len(self.words)
        self.char_count = len(text)

    def _load_whitelist(self) -> set:
        if os.path.exists(self._SHORTHAND_FILE):
            with open(self._SHORTHAND_FILE, 'r') as f:
                return {line.strip().lower() for line in f if line.strip()}
        return set()

    def _get_diversity(self) -> float:
        return len(set(self.words)) / self.word_count if self.word_count > 0 else 0.0

    def _get_prefs(self) -> Dict[str, float]:
        meaningful = [w for w in self.words if w not in self._STOP_WORDS]
        counts = Counter(meaningful).most_common(3)
        return {w: c / self.word_count for w, c in counts} if self.word_count > 0 else {}

    def _get_slang_rate(self) -> float:
        return sum(1 for w in self.words if w in self._SLANG) / self.word_count if self.word_count > 0 else 0.0

    def _get_idiom_rate(self) -> float:
        count = sum(len(re.findall(re.escape(i), self.text_lower)) for i in self._IDIOMS)
        return count / self.word_count if self.word_count > 0 else 0.0

    def _get_contraction_rate(self) -> float:
        return sum(1 for w in self.words if w in self._CONTRACTIONS) / self.word_count if self.word_count > 0 else 0.0

    def _get_cap_ratio(self) -> float:
        return sum(1 for c in self.raw if c.isupper()) / self.char_count if self.char_count > 0 else 0.0

    def _get_stop_rate(self) -> float:
        return sum(1 for w in self.words if w in self._STOP_WORDS) / self.word_count if self.word_count > 0 else 0.0

    def _get_punc_freq(self) -> Dict[str, float]:
        counts = Counter(c for c in self.raw if c in self._PUNC_CHARS)
        return {p: c / self.char_count for p, c in counts.items()} if self.char_count > 0 else {}

    def _get_typos_rate(self) -> float:
        if self.word_count == 0: return 0.0
        spell = SpellChecker()
        unknown = spell.unknown(self.words)
        # Filter out whitelisted items from unknown words
        actual_typos = [w for w in unknown if w not in self.whitelist]
        return len(actual_typos) / self.word_count

    def _get_emoji_rate(self) -> float:
        emojis = re.findall(r'[^\x00-\x7F]+', self.raw) # Basic non-ASCII heuristic for emojis
        return len(emojis) / self.char_count if self.char_count > 0 else 0.0

    def _get_salutation_rate(self) -> float:
        salutations = {"hello", "hi", "dear", "regards", "hey"}
        if self.words and self.words[0] in salutations: return 1.0
        return 0.0

    def _get_quoting_style(self) -> str:
        s, d = self.raw.count("'"), self.raw.count('"')
        if s > d: return "single-quotes"
        if d > s: return "double-quotes"
        return "none/mixed"

    def _get_abbreviation_rate(self) -> float:
        if self.word_count == 0: return 0.0
        abbr_count = sum(1 for w in self.words if w in self.whitelist)
        
        # Phonetic detection for non-dictionary short tokens
        spell = SpellChecker()
        for w in self.words:
            if w not in self.whitelist and len(w) <= 3 and w not in spell:
                for target in self._PHONETIC_TARGETS:
                    if fuzz.ratio(w, target) > 70:
                        abbr_count += 1
                        break
        return abbr_count / self.word_count

    def get_diagnostics(self) -> Dict[str, Any]:
        """Consolidates all metrics into requested JSON schema."""
        if not self.words and not self.raw:
            return {k: None for k in ["vocabulary-diversity", "word-preferences", "slang-rate", 
                                      "Idiom-preferences", "Contraction-rate", "capitalization-ratio", 
                                      "stop-words-rate", "punctuation-frequency", "typos-rate", 
                                      "emoji-rate", "salutation-rate", "quoting-style", "abbreviation-rate"]}
        
        return {
            "vocabulary-diversity": self._get_diversity(),
            "word-preferences": self._get_prefs(),
            "slang-rate": self._get_slang_rate(),
            "Idiom-preferences": self._get_idiom_rate(),
            "Contraction-rate": self._get_contraction_rate(),
            "capitalization-ratio": self._get_cap_ratio(),
            "stop-words-rate": self._get_stop_rate(),
            "punctuation-frequency": self._get_punc_freq(),
            "typos-rate": self._get_typos_rate(),
            "emoji-rate": self._get_emoji_rate(),
            "salutation-rate": self._get_salutation_rate(),
            "quoting-style": self._get_quoting_style(),
            "abbreviation-rate": self._get_abbreviation_rate()
        }

if __name__ == "__main__":
    try:
        user_input = input("Enter a text to analyse:\n>>> ")
        analyzer = Analyser(user_input)
        print(json.dumps(analyzer.get_diagnostics(), indent=4))
    except EOFError:
        pass
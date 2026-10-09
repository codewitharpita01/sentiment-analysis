"""Text normalization that preserves negation cues such as not and never."""

import re
import unicodedata

from nltk.stem import SnowballStemmer

_STEMMER = SnowballStemmer("english")
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)


def clean_text(text: object) -> str:
    """Normalize a review and apply light stemming without removing negations."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text)).lower().strip()
    tokens = _TOKEN_RE.findall(value)
    # Keep stopwords: words such as "not", "never", and "no" change sentiment.
    return " ".join(_STEMMER.stem(token) if token.isalpha() else token for token in tokens)

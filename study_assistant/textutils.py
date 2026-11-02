"""Small NLP helpers used by the offline mode and by the deterministic review checks."""
from __future__ import annotations

import re
from collections import Counter

STOP = set(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just may me might more most
    must my no nor not of off on once only or other our out over own same she should so some such than
    that the their theirs them then there these they this those through to too under until up us very was
    we were what when where which while who whom why will with would you your often many one two use used
    using called known makes make made like different various include includes including every""".split()
)


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.replace("\r", " ")).strip()
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c(])', text)
    return [p.strip() for p in parts if p.strip()]


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']*", s)


def content_words(s: str) -> list[str]:
    return [w.lower() for w in words(s) if w.lower() not in STOP and len(w) > 2]


def word_freq(text: str) -> Counter:
    return Counter(content_words(text))


def usable_sentences(text: str, lo: int = 8, hi: int = 40) -> list[str]:
    return [s for s in split_sentences(text) if lo <= len(words(s)) <= hi]


def rank_sentences(text: str, lo: int = 8, hi: int = 40) -> list[str]:
    """Sentences ranked by importance (word-frequency score), best first."""
    freq = word_freq(text)
    scored = []
    for s in usable_sentences(text, lo, hi):
        cw = set(content_words(s))
        if not cw:
            continue
        scored.append((sum(freq[w] for w in cw) / (len(cw) ** 0.5), s))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def support_ratio(claim: str, document: str) -> float:
    """Best fraction of the claim's content words found in any single document sentence."""
    cw = set(content_words(claim))
    if not cw:
        return 1.0
    best = 0.0
    for s in split_sentences(document):
        best = max(best, len(cw & set(content_words(s))) / len(cw))
    return best

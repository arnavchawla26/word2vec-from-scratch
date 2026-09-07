"""Tokenization and vocabulary building.

Vocabulary construction follows the standard word2vec recipe: drop tokens below
a minimum count, then assign each surviving word a subsampling "keep
probability" so very frequent words (the, is, a, ...) are thinned out during
training (Mikolov et al. 2013, section 2.3).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z']+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens, dropping punctuation/digits.

    A lone apostrophe is stripped from tokens ("cats'" -> "cats") but an
    internal apostrophe is kept ("don't" -> "don't").
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t.strip("'") for t in tokens if t.strip("'")]


@dataclass
class Vocab:
    """A word <-> index mapping with frequencies and subsampling probabilities.

    Build with :meth:`Vocab.build` rather than the constructor directly.
    """

    word_to_index: dict[str, int] = field(default_factory=dict)
    index_to_word: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    total_tokens: int = 0
    subsample_threshold: float = 1e-3

    @classmethod
    def build(
        cls,
        tokenized_sentences: list[list[str]],
        min_count: int = 2,
        subsample_threshold: float = 1e-3,
    ) -> "Vocab":
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        counter: Counter[str] = Counter()
        for sentence in tokenized_sentences:
            counter.update(sentence)

        kept = [(w, c) for w, c in counter.items() if c >= min_count]
        # Deterministic ordering: most frequent first, ties broken alphabetically.
        kept.sort(key=lambda wc: (-wc[1], wc[0]))

        vocab = cls(subsample_threshold=subsample_threshold)
        for word, count in kept:
            vocab.word_to_index[word] = len(vocab.index_to_word)
            vocab.index_to_word.append(word)
            vocab.counts.append(count)
        vocab.total_tokens = sum(vocab.counts)
        return vocab

    def __len__(self) -> int:
        return len(self.index_to_word)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to indices, silently dropping out-of-vocabulary words."""
        return [self.word_to_index[t] for t in tokens if t in self.word_to_index]

    def encode_sentences(self, tokenized_sentences: list[list[str]]) -> list[list[int]]:
        encoded = [self.encode(s) for s in tokenized_sentences]
        return [s for s in encoded if len(s) >= 2]

    def frequency(self, index: int) -> float:
        """Unigram frequency (count / total_tokens) of the word at `index`."""
        return self.counts[index] / self.total_tokens

    def keep_probability(self, index: int) -> float:
        """Probability of keeping an occurrence of this word during subsampling.

        Implements the word2vec subsampling formula:
            P(keep) = (sqrt(f / t) + 1) * (t / f)
        clamped to [0, 1], where f is the word's relative frequency and t is
        `subsample_threshold`. Rarer words (f << t) are always kept; very
        common words are thinned out.
        """
        f = self.frequency(index)
        if f <= 0:
            return 1.0
        t = self.subsample_threshold
        p = ((f / t) ** 0.5 + 1.0) * (t / f)
        return min(1.0, p)

    def to_dict(self) -> dict:
        return {
            "index_to_word": self.index_to_word,
            "counts": self.counts,
            "total_tokens": self.total_tokens,
            "subsample_threshold": self.subsample_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        vocab = cls(
            index_to_word=list(data["index_to_word"]),
            counts=list(data["counts"]),
            total_tokens=int(data["total_tokens"]),
            subsample_threshold=float(data["subsample_threshold"]),
        )
        vocab.word_to_index = {w: i for i, w in enumerate(vocab.index_to_word)}
        return vocab

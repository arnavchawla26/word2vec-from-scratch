"""Training-pair generation and negative sampling.

Skip-gram pairs are (center_word, context_word) for every context word within
a window of the center. CBOW pairs are (context_words, target_word) -- the
inverse framing, predicting the center from the *average* of its context.

Negative sampling draws "noise" words from a smoothed unigram distribution
(raised to the 3/4 power, as in the original word2vec paper) so that a
handful of negative examples per positive pair can stand in for a full
softmax over the vocabulary.
"""

from __future__ import annotations

import random

import numpy as np

from word2vec.tokenizer import Vocab


class NegativeSampler:
    """Samples word indices from the smoothed unigram (^0.75) distribution."""

    def __init__(self, vocab: Vocab, power: float = 0.75, table_size: int = 1_000_000):
        if len(vocab) == 0:
            raise ValueError("cannot build a negative sampler from an empty vocabulary")
        weights = np.array([c**power for c in vocab.counts], dtype=np.float64)
        probs = weights / weights.sum()
        self._probs = probs
        self.vocab_size = len(vocab)
        # A precomputed alias-free table gives O(1) sampling via np.random.choice's
        # internal cumulative-sum search, which is plenty fast for this project's
        # corpus sizes; a true alias method would only matter at much larger scale.
        self._table_size = min(table_size, max(len(vocab) * 200, len(vocab)))

    def sample(self, k: int, rng: np.random.Generator, exclude: int | None = None) -> np.ndarray:
        """Draw `k` negative sample indices, retrying to avoid `exclude`."""
        samples = rng.choice(self.vocab_size, size=k, p=self._probs)
        if exclude is None:
            return samples
        # Rejection-resample any collisions with the true target (rare for a
        # reasonably sized vocabulary); bounded retries avoid an infinite loop
        # on a pathological single-word vocabulary.
        for _ in range(10):
            mask = samples == exclude
            if not mask.any():
                break
            samples[mask] = rng.choice(self.vocab_size, size=int(mask.sum()), p=self._probs)
        return samples


def _keep_mask(sentence: list[int], vocab: Vocab, rng: random.Random, subsample: bool) -> list[bool]:
    if not subsample:
        return [True] * len(sentence)
    return [rng.random() < vocab.keep_probability(idx) for idx in sentence]


def generate_skipgram_pairs(
    encoded_sentences: list[list[int]],
    vocab: Vocab,
    window_size: int = 2,
    subsample: bool = True,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Build (center, context) index pairs from encoded sentences.

    The window size around each center word is itself randomized between 1
    and `window_size` (as in the reference word2vec implementation), which
    gives closer context words relatively more training signal.
    """
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    for sentence in encoded_sentences:
        keep = _keep_mask(sentence, vocab, rng, subsample)
        kept_positions = [i for i, k in enumerate(keep) if k]
        for pos_i, i in enumerate(kept_positions):
            dynamic_window = rng.randint(1, window_size)
            lo = max(0, pos_i - dynamic_window)
            hi = min(len(kept_positions), pos_i + dynamic_window + 1)
            for pos_j in range(lo, hi):
                if pos_j == pos_i:
                    continue
                pairs.append((sentence[i], sentence[kept_positions[pos_j]]))
    return pairs


def generate_cbow_pairs(
    encoded_sentences: list[list[int]],
    vocab: Vocab,
    window_size: int = 2,
    subsample: bool = True,
    seed: int = 0,
) -> list[tuple[list[int], int]]:
    """Build (context_indices, target_index) pairs from encoded sentences."""
    rng = random.Random(seed)
    pairs: list[tuple[list[int], int]] = []
    for sentence in encoded_sentences:
        keep = _keep_mask(sentence, vocab, rng, subsample)
        kept_positions = [i for i, k in enumerate(keep) if k]
        for pos_i, i in enumerate(kept_positions):
            dynamic_window = rng.randint(1, window_size)
            lo = max(0, pos_i - dynamic_window)
            hi = min(len(kept_positions), pos_i + dynamic_window + 1)
            context = [
                sentence[kept_positions[pos_j]]
                for pos_j in range(lo, hi)
                if pos_j != pos_i
            ]
            if context:
                pairs.append((context, sentence[i]))
    return pairs

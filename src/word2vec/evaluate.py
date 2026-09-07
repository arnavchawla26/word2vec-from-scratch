"""Nearest-neighbor and analogy evaluation utilities.

Analogies are solved with the standard 3CosAdd method (Mikolov et al. 2013):
for "a is to b as c is to ?", rank every vocabulary word d by

    cos(vec(d), vec(b) - vec(a) + vec(c))

excluding a, b, and c themselves from the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from word2vec.model import Word2Vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalized_matrix(model: Word2Vec) -> np.ndarray:
    """Mean-center then L2-normalize every row of `model.W_in`.

    On a small corpus, word2vec embeddings tend to share a large common
    ("all words are somewhat similar to each other") component early in
    training -- the mean vector across the vocabulary can account for most
    of each word's norm before the model has converged. Subtracting the
    vocabulary mean before normalizing (the same idea behind "All But The
    Top", Mu & Viswanath 2018) removes that shared direction. Measured
    empirically in benchmark/ablation.py: it gives a real accuracy boost for
    an undertrained model (e.g. +13pp top-5 accuracy at 15 epochs on this
    project's analogy test set) but the gap mostly closes out once training
    has converged (40 epochs). It's kept on by default since it never hurt
    in any configuration tested and helps whenever training is cut short.
    """
    centered = model.W_in - model.W_in.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return centered / norms


def most_similar(model: Word2Vec, word: str, topn: int = 10) -> list[tuple[str, float]]:
    """Return the `topn` words most cosine-similar to `word`, excluding itself."""
    if word not in model:
        raise KeyError(f"'{word}' is not in the vocabulary")
    normed = _normalized_matrix(model)
    idx = model.vocab.word_to_index[word]
    query = normed[idx]
    scores = normed @ query
    scores[idx] = -np.inf
    top_indices = np.argsort(-scores)[:topn]
    return [(model.vocab.index_to_word[i], float(scores[i])) for i in top_indices]


def analogy(model: Word2Vec, a: str, b: str, c: str, topn: int = 5) -> list[tuple[str, float]]:
    """Solve "a is to b as c is to ?" via 3CosAdd, returning the top `topn` guesses.

    The b - a + c arithmetic is done in the same mean-centered, normalized
    space that candidates are ranked in (see `_normalized_matrix`), not on
    the raw embeddings -- mixing the two spaces would compare a query still
    carrying the dominant shared direction against candidates that have had
    it removed, understating accuracy.
    """
    for w in (a, b, c):
        if w not in model:
            raise KeyError(f"'{w}' is not in the vocabulary")
    normed = _normalized_matrix(model)
    idx_a, idx_b, idx_c = (model.vocab.word_to_index[w] for w in (a, b, c))
    target_vec = normed[idx_b] - normed[idx_a] + normed[idx_c]
    norm = np.linalg.norm(target_vec)
    if norm > 0:
        target_vec = target_vec / norm

    scores = normed @ target_vec
    exclude = {model.vocab.word_to_index[w] for w in (a, b, c)}
    for i in exclude:
        scores[i] = -np.inf
    top_indices = np.argsort(-scores)[:topn]
    return [(model.vocab.index_to_word[i], float(scores[i])) for i in top_indices]


@dataclass
class AnalogyResult:
    a: str
    b: str
    c: str
    expected: str
    predictions: list[tuple[str, float]]
    skipped: bool = False

    @property
    def correct(self) -> bool:
        return (not self.skipped) and any(word == self.expected for word, _ in self.predictions)


def evaluate_analogies(
    model: Word2Vec, test_set: list[tuple[str, str, str, str]], topn: int = 1
) -> tuple[float, list[AnalogyResult]]:
    """Run every (a, b, c, expected) quadruple through `analogy` and score accuracy.

    Quadruples whose words all fall outside the model's vocabulary are marked
    `skipped` and excluded from the accuracy denominator (there is nothing the
    model could have gotten right or wrong).
    """
    results: list[AnalogyResult] = []
    for a, b, c, expected in test_set:
        if any(w not in model for w in (a, b, c, expected)):
            results.append(AnalogyResult(a, b, c, expected, [], skipped=True))
            continue
        predictions = analogy(model, a, b, c, topn=topn)
        results.append(AnalogyResult(a, b, c, expected, predictions))

    scored = [r for r in results if not r.skipped]
    accuracy = sum(r.correct for r in scored) / len(scored) if scored else 0.0
    return accuracy, results

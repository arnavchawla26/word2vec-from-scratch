#!/usr/bin/env python3
"""Ablation: does mean-centering the embeddings before ranking actually help?

`evaluate._normalized_matrix` mean-centers the embedding matrix before
computing cosine similarity (see its docstring for why: a shared "all words
are somewhat similar" direction can account for a large fraction of each
word vector's norm on a small corpus). This script checks that claim
empirically rather than assuming it, by training the same model at three
different training budgets and scoring the analogy test set both ways.

Actual finding (see the printed table): centering gives a real accuracy
boost for an *undertrained* model, where the shared component dominates the
vectors -- but as training converges, the shared component shrinks relative
to each word's individual direction and the raw-vs-centered gap mostly
closes. Centering is left on by default because it never hurt in any run
here and meaningfully helps a model that hasn't fully converged, but it is
not the dramatic fix a smaller/quicker ablation might suggest -- most of
this project's accuracy comes from training long enough, not from the
centering step.

Run from the repo root after `pip install -e .`:

    python benchmark/ablation.py
"""

from __future__ import annotations

import time

import numpy as np

from word2vec.corpus import analogy_test_set, generate_corpus
from word2vec.model import Word2Vec, build_vocab_and_tokenize
from word2vec.tokenizer import Vocab


def _rank_analogies(W: np.ndarray, vocab: Vocab, test_set, topn: int = 1) -> float:
    """Score analogies by ranking cosine similarity directly against `W`'s rows.

    Deliberately independent of `word2vec.evaluate` (which always centers)
    so this script can compare "raw" and "centered" on equal footing.
    """
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = W / norms

    scored = 0
    correct = 0
    for a, b, c, expected in test_set:
        if any(w not in vocab.word_to_index for w in (a, b, c, expected)):
            continue
        scored += 1
        ia, ib, ic = (vocab.word_to_index[w] for w in (a, b, c))
        target = normed[ib] - normed[ia] + normed[ic]
        norm = np.linalg.norm(target)
        if norm > 0:
            target = target / norm
        scores = normed @ target
        for i in (ia, ib, ic):
            scores[i] = -np.inf
        top = np.argsort(-scores)[:topn]
        if any(vocab.index_to_word[i] == expected for i in top):
            correct += 1
    return correct / scored if scored else 0.0


def main() -> None:
    raw_sentences = generate_corpus(seed=0, repeats=8)
    vocab, tokenized = build_vocab_and_tokenize(raw_sentences, min_count=2)
    test_set = analogy_test_set()
    print(f"corpus: {len(raw_sentences)} sentences, {len(vocab)} vocab words\n")

    header = f"{'epochs':>6}  {'mean/avg norm':>13}  {'raw@1':>6}  {'cent@1':>6}  {'raw@5':>6}  {'cent@5':>6}"
    print(header)
    print("-" * len(header))

    for epochs in (5, 15, 40):
        model = Word2Vec(vocab, dim=50, architecture="skipgram", seed=0)
        start = time.time()
        model.fit(tokenized, epochs=epochs, window_size=3, negative_samples=8, initial_lr=0.03, verbose=False)
        elapsed = time.time() - start

        mean_norm = float(np.linalg.norm(model.W_in.mean(axis=0)))
        avg_norm = float(np.linalg.norm(model.W_in, axis=1).mean())
        centered = model.W_in - model.W_in.mean(axis=0)

        raw1 = _rank_analogies(model.W_in, vocab, test_set, topn=1)
        cen1 = _rank_analogies(centered, vocab, test_set, topn=1)
        raw5 = _rank_analogies(model.W_in, vocab, test_set, topn=5)
        cen5 = _rank_analogies(centered, vocab, test_set, topn=5)

        print(
            f"{epochs:>6}  {mean_norm / avg_norm:>12.1%}  "
            f"{raw1:>6.1%}  {cen1:>6.1%}  {raw5:>6.1%}  {cen5:>6.1%}"
            f"   ({elapsed:.1f}s)"
        )


if __name__ == "__main__":
    main()

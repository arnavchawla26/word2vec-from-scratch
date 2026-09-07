"""Word2Vec: skip-gram and CBOW with negative sampling, implemented in NumPy.

This is a from-scratch reimplementation of Mikolov et al. (2013), "Distributed
Representations of Words and Phrases and their Compositionality" -- no
autodiff, no ML framework. Every gradient below is derived and coded by hand.

Model
-----
Each word has two vectors: an "input" (center/context-source) vector in
`W_in` and an "output" vector in `W_out`. Negative sampling replaces the full
softmax over the vocabulary with a much cheaper binary classification: for a
true (input, output) pair, push sigmoid(in . out) toward 1; for `k` randomly
sampled negative outputs, push sigmoid(in . neg) toward 0.

For a single positive pair with negative samples n_1..n_k, the loss is:

    L = -log(sigmoid(v_in . v_out)) - sum_j log(sigmoid(-v_in . v_neg_j))

with gradients (letting s = sigmoid(v_in . v_out) and s_j = sigmoid(v_in . v_neg_j)):

    dL/d(v_out)    = (s - 1) * v_in
    dL/d(v_neg_j)  = s_j * v_in
    dL/d(v_in)     = (s - 1) * v_out + sum_j s_j * v_neg_j

CBOW is the same update rule with `v_in` replaced by the *average* of the
context vectors, and the resulting gradient split evenly back across each
context word's row in `W_in`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from word2vec.dataset import NegativeSampler, generate_cbow_pairs, generate_skipgram_pairs
from word2vec.tokenizer import Vocab, simple_tokenize

_MAX_EXP = 6.0  # clip range for numerically stable sigmoid


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, clipped to avoid overflow in exp()."""
    x = np.clip(x, -_MAX_EXP, _MAX_EXP)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TrainingHistory:
    epoch_losses: list[float] = field(default_factory=list)


class Word2Vec:
    """A trainable skip-gram/CBOW word embedding model."""

    def __init__(
        self,
        vocab: Vocab,
        dim: int = 50,
        architecture: str = "skipgram",
        seed: int = 0,
    ):
        if architecture not in ("skipgram", "cbow"):
            raise ValueError("architecture must be 'skipgram' or 'cbow'")
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.vocab = vocab
        self.dim = dim
        self.architecture = architecture
        self._rng = np.random.default_rng(seed)
        vocab_size = len(vocab)
        # Small random init, matching the original implementation's scale.
        self.W_in = (self._rng.random((vocab_size, dim)) - 0.5) / dim
        self.W_out = np.zeros((vocab_size, dim), dtype=np.float64)
        self.history = TrainingHistory()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _train_skipgram_pair(self, center: int, context: int, neg: np.ndarray, lr: float) -> float:
        v_in = self.W_in[center]

        pos_out = self.W_out[context]
        pos_score = sigmoid(np.dot(v_in, pos_out))
        pos_grad_coeff = pos_score - 1.0  # dL/d(dot product) for the positive pair

        neg_out = self.W_out[neg]  # (k, dim)
        neg_scores = sigmoid(neg_out @ v_in)  # (k,)

        # Accumulate the gradient w.r.t. v_in from the positive pair and all negatives
        # before touching W_out, since W_out rows factor into d(loss)/d(v_in).
        grad_in = pos_grad_coeff * pos_out + neg_scores @ neg_out

        self.W_out[context] -= lr * pos_grad_coeff * v_in
        # neg may contain duplicate indices; a plain fancy-index -= would silently
        # drop all but one update per duplicate, so accumulate with np.add.at.
        np.add.at(self.W_out, neg, -lr * neg_scores[:, None] * v_in[None, :])

        self.W_in[center] -= lr * grad_in

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1.0 - neg_scores + 1e-10))
        return float(loss)

    def _train_cbow_pair(self, context: list[int], target: int, neg: np.ndarray, lr: float) -> float:
        context_arr = np.array(context)
        v_in = self.W_in[context_arr].mean(axis=0)

        pos_out = self.W_out[target]
        pos_score = sigmoid(np.dot(v_in, pos_out))
        pos_grad_coeff = pos_score - 1.0

        neg_out = self.W_out[neg]
        neg_scores = sigmoid(neg_out @ v_in)

        grad_in = pos_grad_coeff * pos_out + neg_scores @ neg_out

        self.W_out[target] -= lr * pos_grad_coeff * v_in
        np.add.at(self.W_out, neg, -lr * neg_scores[:, None] * v_in[None, :])

        # The averaging in the forward pass means each context word's gradient
        # is the shared grad_in scaled by 1/len(context) (chain rule through a mean).
        shared_update = lr * grad_in / len(context)
        np.add.at(self.W_in, context_arr, -shared_update)

        loss = -np.log(pos_score + 1e-10) - np.sum(np.log(1.0 - neg_scores + 1e-10))
        return float(loss)

    def fit(
        self,
        tokenized_sentences: list[list[str]],
        epochs: int = 20,
        window_size: int = 2,
        negative_samples: int = 5,
        initial_lr: float = 0.025,
        min_lr: float = 0.0001,
        subsample: bool = True,
        verbose: bool = False,
    ) -> TrainingHistory:
        """Train in place on already-tokenized sentences (using self.vocab to encode)."""
        encoded = self.vocab.encode_sentences(tokenized_sentences)
        if not encoded:
            raise ValueError("no training pairs: every sentence was empty after encoding")
        sampler = NegativeSampler(self.vocab)

        if self.architecture == "skipgram":
            pairs = generate_skipgram_pairs(encoded, self.vocab, window_size, subsample, seed=0)
        else:
            pairs = generate_cbow_pairs(encoded, self.vocab, window_size, subsample, seed=0)
        if not pairs:
            raise ValueError("no training pairs were generated from the corpus")

        order = np.arange(len(pairs))
        for epoch in range(epochs):
            progress = epoch / max(1, epochs - 1)
            lr = initial_lr - (initial_lr - min_lr) * progress
            self._rng.shuffle(order)
            total_loss = 0.0
            for idx in order:
                if self.architecture == "skipgram":
                    center, context = pairs[idx]
                    neg = sampler.sample(negative_samples, self._rng, exclude=context)
                    total_loss += self._train_skipgram_pair(center, context, neg, lr)
                else:
                    context, target = pairs[idx]
                    neg = sampler.sample(negative_samples, self._rng, exclude=target)
                    total_loss += self._train_cbow_pair(context, target, neg, lr)
            mean_loss = total_loss / len(pairs)
            self.history.epoch_losses.append(mean_loss)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}  lr={lr:.5f}  mean_loss={mean_loss:.4f}")
        return self.history

    def vector(self, word: str) -> np.ndarray:
        """Return the (input) embedding for `word`. Raises KeyError if unknown."""
        if word not in self.vocab.word_to_index:
            raise KeyError(f"'{word}' is not in the vocabulary")
        return self.W_in[self.vocab.word_to_index[word]]

    def __contains__(self, word: str) -> bool:
        return word in self.vocab.word_to_index

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), W_in=self.W_in, W_out=self.W_out)
        meta = {
            "dim": self.dim,
            "architecture": self.architecture,
            "vocab": self.vocab.to_dict(),
            "epoch_losses": self.history.epoch_losses,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Word2Vec":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        vocab = Vocab.from_dict(meta["vocab"])
        model = cls(vocab, dim=meta["dim"], architecture=meta["architecture"])
        with np.load(path.with_suffix(".npz")) as data:
            model.W_in = data["W_in"]
            model.W_out = data["W_out"]
        model.history.epoch_losses = meta.get("epoch_losses", [])
        return model


def build_vocab_and_tokenize(
    raw_sentences: list[str], min_count: int = 2, subsample_threshold: float = 1e-3
) -> tuple[Vocab, list[list[str]]]:
    """Convenience helper: tokenize raw sentences and build a Vocab from them."""
    tokenized = [simple_tokenize(s) for s in raw_sentences]
    vocab = Vocab.build(tokenized, min_count=min_count, subsample_threshold=subsample_threshold)
    return vocab, tokenized

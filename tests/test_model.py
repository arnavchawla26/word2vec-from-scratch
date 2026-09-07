import numpy as np
import pytest

from word2vec.model import Word2Vec, build_vocab_and_tokenize, sigmoid
from word2vec.tokenizer import Vocab


def test_sigmoid_at_zero_is_half():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_monotonically_increasing():
    xs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    ys = sigmoid(xs)
    assert all(ys[i] < ys[i + 1] for i in range(len(ys) - 1))


def test_sigmoid_bounded_in_unit_interval():
    xs = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    ys = sigmoid(xs)
    assert np.all(ys > 0.0) and np.all(ys < 1.0)


def test_sigmoid_handles_extreme_values_without_overflow():
    # Without clipping, exp(1e6) overflows and would raise/produce inf/nan.
    ys = sigmoid(np.array([1e6, -1e6]))
    assert np.isfinite(ys).all()


def test_word2vec_rejects_bad_architecture():
    vocab = Vocab.build([["a", "b", "c"]], min_count=1)
    with pytest.raises(ValueError):
        Word2Vec(vocab, architecture="bogus")


def test_word2vec_rejects_bad_dim():
    vocab = Vocab.build([["a", "b", "c"]], min_count=1)
    with pytest.raises(ValueError):
        Word2Vec(vocab, dim=0)


def test_word2vec_vector_lookup():
    vocab = Vocab.build([["a", "b", "c"]], min_count=1)
    model = Word2Vec(vocab, dim=4, seed=0)
    v = model.vector("a")
    assert v.shape == (4,)
    with pytest.raises(KeyError):
        model.vector("nonexistent")


def test_word2vec_contains():
    vocab = Vocab.build([["a", "b", "c"]], min_count=1)
    model = Word2Vec(vocab, dim=4, seed=0)
    assert "a" in model
    assert "nonexistent" not in model


TINY_CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat and a dog played together",
    "the cat chased the dog around the yard",
] * 20


@pytest.mark.parametrize("architecture", ["skipgram", "cbow"])
def test_fit_reduces_loss_over_epochs(architecture):
    vocab, tokenized = build_vocab_and_tokenize(TINY_CORPUS, min_count=2)
    model = Word2Vec(vocab, dim=8, architecture=architecture, seed=0)
    history = model.fit(tokenized, epochs=8, window_size=2, negative_samples=3, initial_lr=0.05)
    assert len(history.epoch_losses) == 8
    # Losses need not decrease monotonically every epoch, but the final epoch
    # should be meaningfully better than the first.
    assert history.epoch_losses[-1] < history.epoch_losses[0]


def test_fit_changes_embeddings():
    vocab, tokenized = build_vocab_and_tokenize(TINY_CORPUS, min_count=2)
    model = Word2Vec(vocab, dim=8, seed=0)
    W_in_before = model.W_in.copy()
    model.fit(tokenized, epochs=3, window_size=2, negative_samples=3)
    assert not np.allclose(W_in_before, model.W_in)


def test_fit_is_deterministic_given_seed():
    vocab, tokenized = build_vocab_and_tokenize(TINY_CORPUS, min_count=2)
    model_a = Word2Vec(vocab, dim=8, seed=7)
    model_a.fit(tokenized, epochs=3, window_size=2, negative_samples=3)

    model_b = Word2Vec(vocab, dim=8, seed=7)
    model_b.fit(tokenized, epochs=3, window_size=2, negative_samples=3)

    assert np.array_equal(model_a.W_in, model_b.W_in)
    assert np.array_equal(model_a.W_out, model_b.W_out)


def test_fit_raises_on_empty_corpus():
    vocab, _ = build_vocab_and_tokenize(TINY_CORPUS, min_count=2)
    model = Word2Vec(vocab, dim=4, seed=0)
    with pytest.raises(ValueError):
        model.fit([[]], epochs=1)


def test_save_and_load_roundtrip(tmp_path):
    vocab, tokenized = build_vocab_and_tokenize(TINY_CORPUS, min_count=2)
    model = Word2Vec(vocab, dim=6, architecture="cbow", seed=1)
    model.fit(tokenized, epochs=2, window_size=2, negative_samples=3)

    out_path = tmp_path / "model"
    model.save(out_path)
    assert out_path.with_suffix(".npz").exists()
    assert out_path.with_suffix(".json").exists()

    loaded = Word2Vec.load(out_path)
    assert loaded.dim == model.dim
    assert loaded.architecture == model.architecture
    assert loaded.vocab.index_to_word == model.vocab.index_to_word
    assert np.array_equal(loaded.W_in, model.W_in)
    assert np.array_equal(loaded.W_out, model.W_out)
    assert loaded.history.epoch_losses == model.history.epoch_losses


def test_skipgram_single_pair_update_moves_vectors_toward_each_other():
    # A direct check of the hand-derived gradient: after one positive-pair
    # update (with zero negative samples), the center and context vectors'
    # dot product should have increased (pulled together).
    vocab = Vocab.build([["a", "b"]], min_count=1)
    model = Word2Vec(vocab, dim=4, seed=0)
    center, context = 0, 1
    # Give both non-zero, distinguishable starting vectors.
    model.W_in[center] = np.array([0.1, 0.2, 0.3, 0.4])
    model.W_out[context] = np.array([0.4, 0.3, 0.2, 0.1])
    dot_before = np.dot(model.W_in[center], model.W_out[context])

    model._train_skipgram_pair(center, context, neg=np.array([], dtype=int), lr=0.1)

    dot_after = np.dot(model.W_in[center], model.W_out[context])
    assert dot_after > dot_before

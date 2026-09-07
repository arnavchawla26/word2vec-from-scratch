from collections import Counter

import numpy as np
import pytest

from word2vec.dataset import NegativeSampler, generate_cbow_pairs, generate_skipgram_pairs
from word2vec.tokenizer import Vocab

SENTENCES = [["a", "b", "c", "d", "e"]]


def _vocab_no_subsampling():
    # min_count=1 keeps every word; each occurs once so counts are all equal.
    return Vocab.build(SENTENCES, min_count=1)


def test_skipgram_pairs_fixed_window_no_subsampling():
    vocab = _vocab_no_subsampling()
    encoded = vocab.encode_sentences(SENTENCES)
    # window_size=1 with dynamic window disabled would give exactly neighbors;
    # since the window is randomized in [1, window_size], use window_size=1 so
    # it's deterministic: every word paired with its immediate left/right neighbor.
    pairs = generate_skipgram_pairs(encoded, vocab, window_size=1, subsample=False, seed=0)
    pair_words = {(vocab.index_to_word[c], vocab.index_to_word[o]) for c, o in pairs}
    # "a" only has "b" as a neighbor; "c" has "b" and "d".
    assert ("a", "b") in pair_words
    assert ("c", "b") in pair_words
    assert ("c", "d") in pair_words
    assert ("a", "d") not in pair_words  # too far apart for window=1


def test_skipgram_pairs_are_symmetric():
    vocab = _vocab_no_subsampling()
    encoded = vocab.encode_sentences(SENTENCES)
    pairs = generate_skipgram_pairs(encoded, vocab, window_size=1, subsample=False, seed=0)
    pair_set = set(pairs)
    for center, context in pairs:
        assert (context, center) in pair_set


def test_skipgram_no_pairs_from_single_word_sentence():
    vocab = Vocab.build([["only"]], min_count=1)
    encoded = vocab.encode_sentences([["only"]])
    assert encoded == []  # dropped: needs >= 2 tokens
    pairs = generate_skipgram_pairs(encoded, vocab, window_size=2, subsample=False, seed=0)
    assert pairs == []


def test_cbow_pairs_context_excludes_target():
    vocab = _vocab_no_subsampling()
    encoded = vocab.encode_sentences(SENTENCES)
    pairs = generate_cbow_pairs(encoded, vocab, window_size=2, subsample=False, seed=0)
    for context, target in pairs:
        assert target not in context


def test_cbow_pairs_cover_every_position():
    vocab = _vocab_no_subsampling()
    encoded = vocab.encode_sentences(SENTENCES)
    pairs = generate_cbow_pairs(encoded, vocab, window_size=2, subsample=False, seed=0)
    targets = {vocab.index_to_word[t] for _, t in pairs}
    assert targets == {"a", "b", "c", "d", "e"}


def test_subsampling_reduces_pair_count_for_frequent_word():
    # "frequent" appears 500x, "rare" 500x too but in a long enough corpus that
    # subsampling of "frequent"... construct so "the" is very common relative to others.
    sentences = [["the", "cat", "sat"] for _ in range(500)] + [["a", "dog", "ran"]]
    vocab = Vocab.build(sentences, min_count=1, subsample_threshold=1e-3)
    encoded = vocab.encode_sentences(sentences)
    pairs_no_sub = generate_skipgram_pairs(encoded, vocab, window_size=2, subsample=False, seed=0)
    pairs_sub = generate_skipgram_pairs(encoded, vocab, window_size=2, subsample=True, seed=0)
    assert len(pairs_sub) < len(pairs_no_sub)


def test_negative_sampler_rejects_empty_vocab():
    vocab = Vocab.build([], min_count=1)
    with pytest.raises(ValueError):
        NegativeSampler(vocab)


def test_negative_sampler_excludes_target_when_requested():
    vocab = Vocab.build([["a", "b"]], min_count=1)
    sampler = NegativeSampler(vocab)
    rng = np.random.default_rng(0)
    target = vocab.word_to_index["a"]
    for _ in range(20):
        samples = sampler.sample(5, rng, exclude=target)
        assert target not in samples


def test_negative_sampler_matches_unigram_power_distribution():
    # Two words with counts 900 and 100 (ratio 9:1). Unigram^0.75 sampling
    # should favor the frequent word less than a raw-frequency sampler would,
    # but still noticeably more than uniform -- check it lands strictly
    # between the raw-frequency ratio and 50/50.
    sentences = [["frequent"]] * 900 + [["rare"]] * 100
    vocab = Vocab.build(sentences, min_count=1)
    sampler = NegativeSampler(vocab)
    rng = np.random.default_rng(0)
    samples = sampler.sample(20000, rng)
    counts = Counter(samples.tolist())
    frequent_idx = vocab.word_to_index["frequent"]
    frequent_share = counts[frequent_idx] / len(samples)
    # Raw frequency share would be 0.9; smoothing towards 0.75 power pulls it down.
    assert 0.5 < frequent_share < 0.9


def test_negative_sampler_deterministic_with_seeded_generator():
    vocab = Vocab.build([["a", "b", "c"]], min_count=1)
    sampler = NegativeSampler(vocab)
    s1 = sampler.sample(10, np.random.default_rng(42))
    s2 = sampler.sample(10, np.random.default_rng(42))
    assert np.array_equal(s1, s2)

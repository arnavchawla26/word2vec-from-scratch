import pytest

from word2vec.tokenizer import Vocab, simple_tokenize


def test_simple_tokenize_lowercases_and_splits():
    assert simple_tokenize("The Quick Brown Fox!") == ["the", "quick", "brown", "fox"]


def test_simple_tokenize_drops_punctuation_and_digits():
    assert simple_tokenize("hello, world -- 123 go!") == ["hello", "world", "go"]


def test_simple_tokenize_keeps_internal_apostrophe_but_strips_trailing():
    assert simple_tokenize("don't stop the cats' toys") == ["don't", "stop", "the", "cats", "toys"]


def test_simple_tokenize_empty_string():
    assert simple_tokenize("") == []


SENTENCES = [
    ["the", "cat", "sat"],
    ["the", "cat", "ran"],
    ["a", "dog", "sat"],
]


def test_vocab_build_drops_below_min_count():
    vocab = Vocab.build(SENTENCES, min_count=2)
    # "the" (2) and "cat" (2) and "sat" (2) survive; "ran", "a", "dog" (1 each) do not.
    assert set(vocab.index_to_word) == {"the", "cat", "sat"}


def test_vocab_build_min_count_one_keeps_everything():
    vocab = Vocab.build(SENTENCES, min_count=1)
    assert set(vocab.index_to_word) == {"the", "cat", "sat", "ran", "a", "dog"}


def test_vocab_sorted_by_frequency_descending():
    vocab = Vocab.build(SENTENCES, min_count=1)
    freqs = [vocab.counts[i] for i in range(len(vocab))]
    assert freqs == sorted(freqs, reverse=True)


def test_vocab_rejects_invalid_min_count():
    with pytest.raises(ValueError):
        Vocab.build(SENTENCES, min_count=0)


def test_vocab_encode_drops_oov_words():
    vocab = Vocab.build(SENTENCES, min_count=2)
    encoded = vocab.encode(["the", "cat", "unknown_word", "sat"])
    assert len(encoded) == 3
    assert vocab.index_to_word[encoded[0]] == "the"


def test_vocab_encode_sentences_drops_short_sentences():
    vocab = Vocab.build(SENTENCES, min_count=2)
    # "a dog sat" -> only "sat" survives min_count=2, so the encoded sentence
    # has length 1 and should be dropped (need >= 2 tokens for training pairs).
    encoded = vocab.encode_sentences([["a", "dog", "sat"], ["the", "cat", "sat"]])
    assert len(encoded) == 1
    assert len(encoded[0]) == 3


def test_vocab_len_and_roundtrip_dict():
    vocab = Vocab.build(SENTENCES, min_count=1)
    assert len(vocab) == 6
    restored = Vocab.from_dict(vocab.to_dict())
    assert restored.index_to_word == vocab.index_to_word
    assert restored.counts == vocab.counts
    assert restored.total_tokens == vocab.total_tokens
    assert restored.word_to_index == vocab.word_to_index


def test_keep_probability_rare_word_always_kept():
    # A word occurring once in a huge corpus has frequency far below threshold,
    # so its keep probability should clamp to 1.0.
    sentences = [["rare"]] + [["common"]] * 100000
    vocab = Vocab.build(sentences, min_count=1, subsample_threshold=1e-3)
    rare_idx = vocab.word_to_index["rare"]
    assert vocab.keep_probability(rare_idx) == 1.0


def test_keep_probability_frequent_word_is_thinned():
    sentences = [["rare"]] + [["common"]] * 100000
    vocab = Vocab.build(sentences, min_count=1, subsample_threshold=1e-3)
    common_idx = vocab.word_to_index["common"]
    rare_idx = vocab.word_to_index["rare"]
    assert vocab.keep_probability(common_idx) < vocab.keep_probability(rare_idx)
    assert 0.0 < vocab.keep_probability(common_idx) < 1.0


def test_keep_probability_monotonic_in_frequency():
    # Three words at three different frequencies: keep probability should
    # decrease as frequency increases.
    sentences = [["low"]] * 10 + [["mid"]] * 100 + [["high"]] * 1000
    vocab = Vocab.build(sentences, min_count=1, subsample_threshold=1e-3)
    p_low = vocab.keep_probability(vocab.word_to_index["low"])
    p_mid = vocab.keep_probability(vocab.word_to_index["mid"])
    p_high = vocab.keep_probability(vocab.word_to_index["high"])
    assert p_low > p_mid > p_high

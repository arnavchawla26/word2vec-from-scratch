from word2vec.corpus import (
    ADJECTIVE_COMPARATIVE_PAIRS,
    COUNTRY_CAPITAL_PAIRS,
    ROYALTY_GENDER_PAIRS,
    VERB_TENSE_PAIRS,
    analogy_test_set,
    generate_corpus,
)


def test_generate_corpus_deterministic_for_same_seed():
    a = generate_corpus(seed=0, repeats=3)
    b = generate_corpus(seed=0, repeats=3)
    assert a == b


def test_generate_corpus_different_seeds_differ():
    a = generate_corpus(seed=0, repeats=3)
    b = generate_corpus(seed=1, repeats=3)
    assert a != b


def test_generate_corpus_repeats_scales_sentence_count():
    small = generate_corpus(seed=0, repeats=2)
    large = generate_corpus(seed=0, repeats=4)
    assert len(large) > len(small)


def test_generate_corpus_contains_expected_vocabulary():
    sentences = " ".join(generate_corpus(seed=0, repeats=5))
    for word in ("king", "queen", "paris", "berlin", "walked", "bigger"):
        assert word in sentences


def test_generate_corpus_sentences_are_nonempty_strings():
    sentences = generate_corpus(seed=0, repeats=2)
    assert len(sentences) > 0
    assert all(isinstance(s, str) and s.strip() for s in sentences)


def test_analogy_test_set_words_all_appear_in_corpus_vocabulary():
    corpus_text = " ".join(generate_corpus(seed=0, repeats=10))
    corpus_words = set(corpus_text.split())
    for a, b, c, d in analogy_test_set():
        for word in (a, b, c, d):
            assert word in corpus_words, f"analogy word '{word}' never appears in the generated corpus"


def test_analogy_test_set_quadruples_have_distinct_words():
    for a, b, c, d in analogy_test_set():
        assert len({a, b, c, d}) == 4


def test_pair_lists_have_no_duplicate_words_within_family():
    for pairs in (ROYALTY_GENDER_PAIRS, COUNTRY_CAPITAL_PAIRS, VERB_TENSE_PAIRS, ADJECTIVE_COMPARATIVE_PAIRS):
        firsts = [p[0] for p in pairs]
        seconds = [p[1] for p in pairs]
        assert len(firsts) == len(set(firsts))
        assert len(seconds) == len(set(seconds))

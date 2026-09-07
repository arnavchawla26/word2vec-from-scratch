import numpy as np
import pytest

from word2vec.evaluate import AnalogyResult, analogy, cosine_similarity, evaluate_analogies, most_similar
from word2vec.model import Word2Vec
from word2vec.tokenizer import Vocab


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([-1.0, -1.0])) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_defined_as_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def _hand_built_model(vectors: dict[str, list[float]]) -> Word2Vec:
    """A model with W_in set exactly to the given vectors, for deterministic tests."""
    words = list(vectors)
    vocab = Vocab.build([words], min_count=1)
    dim = len(next(iter(vectors.values())))
    model = Word2Vec(vocab, dim=dim, seed=0)
    for word, vec in vectors.items():
        model.W_in[vocab.word_to_index[word]] = np.array(vec, dtype=np.float64)
    return model


def test_most_similar_ranks_by_cosine_distance():
    model = _hand_built_model(
        {
            "king": [1.0, 0.0],
            "queen": [0.9, 0.1],   # close to king
            "banana": [-1.0, 0.0],  # opposite of king
            "car": [0.0, 1.0],      # orthogonal to king
        }
    )
    results = most_similar(model, "king", topn=3)
    words = [w for w, _ in results]
    assert words[0] == "queen"
    assert words[-1] == "banana"


def test_most_similar_excludes_query_word():
    model = _hand_built_model(
        {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0], "d": [0.0, -1.0]}
    )
    results = most_similar(model, "a", topn=3)
    assert "a" not in [w for w, _ in results]


def test_most_similar_raises_on_unknown_word():
    model = _hand_built_model({"a": [1.0, 0.0]})
    with pytest.raises(KeyError):
        most_similar(model, "nonexistent")


def test_analogy_solves_hand_built_linear_relationship():
    # Construct vectors where queen - king == woman - man exactly, so
    # king + (woman - man) == queen, i.e. "man is to woman as king is to queen".
    model = _hand_built_model(
        {
            "man": [1.0, 0.0, 0.0],
            "woman": [1.0, 1.0, 0.0],
            "king": [0.0, 0.0, 1.0],
            "queen": [0.0, 1.0, 1.0],
            "car": [5.0, -5.0, 5.0],  # unrelated distractor
        }
    )
    results = analogy(model, "man", "woman", "king", topn=1)
    assert results[0][0] == "queen"


def test_analogy_excludes_query_words_from_results():
    model = _hand_built_model(
        {
            "man": [1.0, 0.0, 0.0],
            "woman": [1.0, 1.0, 0.0],
            "king": [0.0, 0.0, 1.0],
            "queen": [0.0, 1.0, 1.0],
            "car": [5.0, -5.0, 5.0],
        }
    )
    results = analogy(model, "man", "woman", "king", topn=2)
    words = {w for w, _ in results}
    assert not ({"man", "woman", "king"} & words)


def test_analogy_raises_on_unknown_word():
    model = _hand_built_model({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
    with pytest.raises(KeyError):
        analogy(model, "a", "b", "nonexistent")


def test_evaluate_analogies_scores_hand_built_relationship():
    model = _hand_built_model(
        {
            "man": [1.0, 0.0, 0.0],
            "woman": [1.0, 1.0, 0.0],
            "king": [0.0, 0.0, 1.0],
            "queen": [0.0, 1.0, 1.0],
        }
    )
    accuracy, results = evaluate_analogies(model, [("man", "woman", "king", "queen")], topn=1)
    assert accuracy == 1.0
    assert results[0].correct
    assert not results[0].skipped


def test_evaluate_analogies_skips_out_of_vocabulary_quadruples():
    model = _hand_built_model({"man": [1.0, 0.0], "woman": [0.0, 1.0]})
    accuracy, results = evaluate_analogies(
        model, [("man", "woman", "nonexistent", "also_missing")], topn=1
    )
    assert results[0].skipped
    assert accuracy == 0.0  # no scored quadruples, defined as 0.0 not NaN


def test_evaluate_analogies_mixed_scored_and_skipped():
    model = _hand_built_model(
        {
            "man": [1.0, 0.0, 0.0],
            "woman": [1.0, 1.0, 0.0],
            "king": [0.0, 0.0, 1.0],
            "queen": [0.0, 1.0, 1.0],
        }
    )
    test_set = [
        ("man", "woman", "king", "queen"),  # scored, correct
        ("man", "woman", "ghost", "phantom"),  # skipped, OOV
    ]
    accuracy, results = evaluate_analogies(model, test_set, topn=1)
    assert accuracy == 1.0  # only the scored one counts
    assert [r.skipped for r in results] == [False, True]


def test_analogy_result_correct_property_false_when_skipped():
    r = AnalogyResult("a", "b", "c", "d", predictions=[("d", 0.9)], skipped=True)
    assert r.correct is False

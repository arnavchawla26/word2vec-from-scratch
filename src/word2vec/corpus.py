"""A small, synthetic, reproducible corpus for training and demoing word2vec.

Word2vec needs a lot of repeated, structured co-occurrence to learn useful
directions in embedding space, and a truly small *natural* corpus (a
paragraph or two) isn't enough to make analogies like king - man + woman ~=
queen work. Real word2vec is trained on billions of tokens.

Instead of committing a large scraped text file, this module *generates* a
toy corpus from sentence templates filled in with word-pairs from four
relation families (royalty/gender, country-capital, verb tense, adjective
comparative). The relations repeat across many sentence shapes, which is
exactly the kind of regular co-occurrence structure that lets a small model
learn a linear analogy direction on a tiny amount of data. This is a
deliberate, documented shortcut for demonstration purposes, not a claim that
the resulting embeddings capture real-world semantics the way embeddings
trained on a real corpus would.

The corpus is fully deterministic for a given seed, so `generate_corpus(seed=0)`
always returns the same sentences.
"""

from __future__ import annotations

import random

# Each pair is (word_a, word_b) such that "a is to b" as "man is to woman"
# (word_a is conventionally the "male"/base/positive-sized form).
ROYALTY_GENDER_PAIRS = [
    ("king", "queen"),
    ("prince", "princess"),
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("husband", "wife"),
    ("uncle", "aunt"),
    ("actor", "actress"),
    ("son", "daughter"),
    ("brother", "sister"),
]

COUNTRY_CAPITAL_PAIRS = [
    ("france", "paris"),
    ("germany", "berlin"),
    ("italy", "rome"),
    ("spain", "madrid"),
    ("japan", "tokyo"),
    ("russia", "moscow"),
    ("england", "london"),
    ("greece", "athens"),
    ("egypt", "cairo"),
    ("norway", "oslo"),
]

# (base, past-tense)
VERB_TENSE_PAIRS = [
    ("walk", "walked"),
    ("play", "played"),
    ("jump", "jumped"),
    ("look", "looked"),
    ("talk", "talked"),
    ("cook", "cooked"),
    ("climb", "climbed"),
    ("clean", "cleaned"),
    ("paint", "painted"),
    ("visit", "visited"),
]

# (base, comparative)
ADJECTIVE_COMPARATIVE_PAIRS = [
    ("big", "bigger"),
    ("small", "smaller"),
    ("fast", "faster"),
    ("slow", "slower"),
    ("strong", "stronger"),
    ("tall", "taller"),
    ("cold", "colder"),
    ("warm", "warmer"),
    ("bright", "brighter"),
    ("dark", "darker"),
]

ROYALTY_TEMPLATES = [
    "the {a} ruled the kingdom while the {b} watched from the tower",
    "the old {a} and the wise {b} sat together at the long table",
    "every {a} in the story was brave and every {b} was clever",
    "the young {a} met the young {b} beside the river",
    "people said the {a} and the {b} would rule the land together",
    "the {a} wore a crown and the {b} wore one too",
    "in the tale the {a} loved the {b} very much",
]

COUNTRY_TEMPLATES = [
    "the capital of {country} is {capital}",
    "she flew from {capital} to visit another {country} city",
    "people in {country} often talk about {capital} with pride",
    "he studied the history of {country} and the streets of {capital}",
    "many tourists travel to {capital} to see the heart of {country}",
    "the government of {country} meets in {capital} every year",
]

VERB_TEMPLATES = [
    "yesterday she {past} in the park but today she will {base} again",
    "he {past} for hours and then {past} some more",
    "they always {base} on sunday and they {past} last sunday too",
    "we {past} together last week just like we usually {base}",
    "the children {past} happily and the teacher {past} along with them",
]

ADJ_TEMPLATES = [
    "the {a} tree grew {b} than the other one in the yard",
    "his new car is {b} than mine even though mine looks {a}",
    "the river seemed {a} at first but the lake was even {b}",
    "everyone agreed the second building was {b} than the {a} one before it",
    "she wanted something {b} not just something {a}",
]


def _combinations(pairs: list[tuple[str, str]], rng: random.Random) -> list[tuple[str, str]]:
    """All ordered pairs of *distinct* items from `pairs`, shuffled."""
    combos = [(a, b) for a in pairs for b in pairs if a != b]
    rng.shuffle(combos)
    return combos


def generate_corpus(seed: int = 0, repeats: int = 6) -> list[str]:
    """Generate a deterministic list of toy sentences.

    Args:
        seed: RNG seed; the same seed always produces the same corpus.
        repeats: how many times to cycle through each template family with a
            freshly shuffled set of word combinations. Higher values give
            word2vec more repeated co-occurrence to learn from, at the cost
            of a longer training run.

    Returns:
        A list of raw (untokenized) sentence strings.
    """
    rng = random.Random(seed)
    sentences: list[str] = []

    for _ in range(repeats):
        for (a, b) in ROYALTY_GENDER_PAIRS:
            template = rng.choice(ROYALTY_TEMPLATES)
            sentences.append(template.format(a=a, b=b))
            # Also generate a cross-pair sentence so e.g. "king"/"man" and
            # "queen"/"woman" show up in shared contexts.
            other_a, other_b = rng.choice(ROYALTY_GENDER_PAIRS)
            template2 = rng.choice(ROYALTY_TEMPLATES)
            sentences.append(template2.format(a=other_a, b=other_b))

        for (country, capital) in COUNTRY_CAPITAL_PAIRS:
            template = rng.choice(COUNTRY_TEMPLATES)
            sentences.append(template.format(country=country, capital=capital))

        for (base, past) in VERB_TENSE_PAIRS:
            template = rng.choice(VERB_TEMPLATES)
            sentences.append(template.format(base=base, past=past))

        for (base, comp) in ADJECTIVE_COMPARATIVE_PAIRS:
            template = rng.choice(ADJ_TEMPLATES)
            sentences.append(template.format(a=base, b=comp))

    rng.shuffle(sentences)
    return sentences


def analogy_test_set() -> list[tuple[str, str, str, str]]:
    """Hand-built analogy quadruples (a, b, c, expected_d) meaning a:b :: c:d.

    Drawn from the same four relation families the corpus is built from, so a
    model trained on `generate_corpus` has a fair shot at solving them, plus
    a couple of "diagonal" analogies that cross two royalty/gender pairs to
    check the model learned a shared gender direction rather than
    memorizing individual pairs.
    """
    return [
        ("man", "woman", "king", "queen"),
        ("man", "woman", "boy", "girl"),
        ("man", "woman", "father", "mother"),
        ("man", "woman", "actor", "actress"),
        ("king", "man", "queen", "woman"),
        ("france", "paris", "germany", "berlin"),
        ("france", "paris", "japan", "tokyo"),
        ("italy", "rome", "spain", "madrid"),
        ("walk", "walked", "play", "played"),
        ("walk", "walked", "look", "looked"),
        ("cook", "cooked", "clean", "cleaned"),
        ("big", "bigger", "small", "smaller"),
        ("big", "bigger", "fast", "faster"),
        ("cold", "colder", "warm", "warmer"),
        ("king", "queen", "prince", "princess"),
    ]

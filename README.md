# word2vec-from-scratch

Skip-gram and CBOW word embeddings with negative sampling, implemented from
scratch in NumPy — no PyTorch, no TensorFlow, no autodiff. Every forward pass
and every gradient in `model.py` is derived by hand from Mikolov et al.
(2013), "Distributed Representations of Words and Phrases and their
Compositionality."

Trained on a small, deterministically-generated synthetic corpus (see
[Why a synthetic corpus?](#why-a-synthetic-corpus) below), it learns
embeddings good enough to solve real analogies:

```
$ w2v analogy model man woman king
man is to woman as king is to ...
queen    0.7899
princess 0.5921
girl     0.5723
aunt     0.5573
wife     0.5427
```

## What it does

- **Two architectures**: skip-gram (predict context from center word) and
  CBOW (predict center word from averaged context), both trained with
  negative sampling instead of a full softmax.
- **From-scratch training loop**: hand-derived gradients, frequent-word
  subsampling, a dynamic context window, and linear learning-rate decay —
  the same recipe as the reference `word2vec.c` implementation, in readable
  NumPy.
- **A synthetic training corpus** (`corpus.py`) built from sentence
  templates covering four analogy families: royalty/gender (king/queen,
  man/woman), country/capital (france/paris), verb tense (walk/walked), and
  adjective comparatives (big/bigger). Fully deterministic given a seed —
  no large text file is committed to the repo.
- **Analogy solving** via 3CosAdd (`evaluate.py`): `vec(b) - vec(a) +
  vec(c)`, ranked against every word in the vocabulary by cosine similarity,
  after mean-centering the embedding matrix (see the docstring in
  `evaluate.py` and `benchmark/ablation.py` for why, and how much it
  actually helps).
- **A CLI** (`w2v`) to train a model and then explore it: nearest
  neighbors, analogy solving, and scoring against a hand-built analogy test
  set.

## Tech stack

Python 3.10+, NumPy for vectorized linear algebra (dot products, batched
negative-sample lookups). No ML framework, no GPU — training the default
configuration takes well under a minute on a laptop CPU. Tests use `pytest`;
`pyflakes` for linting.

## How to run

```bash
git clone https://github.com/arnavchawla26/word2vec-from-scratch.git
cd word2vec-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train on the built-in synthetic corpus (~40s on a laptop CPU)
w2v train --out model

# Explore the trained embeddings
w2v similar model king --topn 8
w2v analogy model man woman king
w2v analogy model france paris japan
w2v evaluate model --topn 5 --verbose
```

You can also train on your own text (one sentence per line):

```bash
w2v train --out model --corpus my_sentences.txt --min-count 3
```

Run the test suite:

```bash
pytest
```

Run the mean-centering ablation (trains three small models to show how much
each part of the pipeline actually contributes — takes ~30s):

```bash
python benchmark/ablation.py
```

### CLI reference

| Command | Purpose |
|---|---|
| `w2v train --out PATH [options]` | Train a model, saving `PATH.npz` (weights) and `PATH.json` (vocab + metadata). |
| `w2v similar PATH WORD [--topn N]` | Print the `N` nearest neighbors of `WORD` by cosine similarity. |
| `w2v analogy PATH A B C [--topn N]` | Solve "A is to B as C is to ?". |
| `w2v evaluate PATH [--topn N] [--verbose]` | Score the model against the built-in analogy test set; count a prediction correct if the answer is in the top `N`. |

Key `train` options: `--arch {skipgram,cbow}`, `--dim`, `--window`,
`--epochs`, `--negative-samples`, `--min-count`, `--lr`,
`--corpus-repeats` (synthetic corpus size), `--corpus-seed`, `--seed`. Run
`w2v train --help` for the full list.

## Why a synthetic corpus?

Word2vec's analogy behavior emerges from *statistics*, not meaning — it
needs a lot of repeated, structured co-occurrence to learn a clean linear
direction for a relation like gender or tense. Real word2vec is trained on
billions of tokens. A truly small natural corpus (a paragraph or two) isn't
enough data for `king - man + woman ≈ queen` to work at all.

Rather than committing a large scraped text file, `corpus.py` *generates* a
few thousand short sentences from templates filled in with word pairs from
four relation families, repeated across many sentence shapes and
combinations with a fixed seed. That regular structure is exactly what lets
a tiny model learn a usable analogy direction from a tiny amount of data.
This is a deliberate, documented shortcut for demonstrating the algorithm —
not a claim that these embeddings capture real-world semantics the way
embeddings trained on a real corpus would. The `analogy_test_set()` in the
same file is a hand-built set of 15 quadruples spanning all four relation
families, used by `w2v evaluate` and `tests/test_corpus.py`.

## Results

Training the default configuration (`w2v train --out model`, skip-gram,
dim=64, 40 epochs, ~750 sentences / 193-word vocabulary, ~40s on a laptop
CPU) and scoring against the built-in 15-quadruple analogy test set:

| Metric | Score |
|---|---|
| Analogy accuracy@1 | 66.7% (10/15) |
| Analogy accuracy@5 | 93.3% (14/15) |

The one analogy that misses even at top-5 (`king:queen :: prince:princess`)
still ranks `princess` 5th — the model has the right neighborhood, just not
a clean enough margin. Full per-analogy output is reproducible with
`w2v evaluate model --topn 5 --verbose`.

`benchmark/ablation.py` measures a design decision empirically instead of
assuming it: does mean-centering the embeddings before ranking actually
matter? At 40 epochs (converged), centered and raw ranking score
identically — the shared "all words are somewhat similar" component the
model picks up early in training shrinks relative to each word's own
direction as training continues. At 15 epochs (undertrained), centering
gives a real +13pp boost in top-5 accuracy (80.0% vs. 66.7%). Centering is
kept on by default since it never hurt in any configuration tested and
meaningfully helps whenever training is cut short.

## Project structure

```
src/word2vec/
    tokenizer.py   Tokenization + vocabulary building (min-count filtering, subsampling probabilities)
    corpus.py      Synthetic training corpus generator + hand-built analogy test set
    dataset.py     Skip-gram / CBOW pair generation, unigram^0.75 negative sampling
    model.py       Word2Vec: hand-derived skip-gram & CBOW training with negative sampling
    evaluate.py    Cosine similarity, nearest neighbors, 3CosAdd analogy solving
    cli.py         `w2v` command-line interface
tests/             67 tests covering every module above, plus an end-to-end CLI smoke test
benchmark/
    ablation.py    Empirical check of the mean-centering design decision
```

## Current status

Complete, tested v1. All four modules (tokenizer, corpus, dataset, model,
evaluate) have unit tests; `tests/test_cli.py` runs the full
train → similar → analogy → evaluate pipeline end to end. Both
architectures (skip-gram and CBOW) train and pass the same test suite.
Nothing planned beyond this is currently in progress.

Possible future extensions (not started): a hierarchical-softmax training
option as an alternative to negative sampling; phrase detection
(bigram/trigram collocations, as in the original paper's "Learning Phrases"
section); loading a real external corpus with streaming tokenization
instead of holding it fully in memory.

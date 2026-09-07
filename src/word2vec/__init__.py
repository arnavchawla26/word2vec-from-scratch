"""word2vec-from-scratch: skip-gram and CBOW word embeddings with negative sampling.

Pure NumPy implementation (no ML frameworks) of Mikolov et al.'s word2vec, trained
on a small synthetic corpus, with tools to explore nearest neighbors and solve
word analogies (king - man + woman ~= queen).
"""

from word2vec.model import Word2Vec
from word2vec.tokenizer import Vocab, simple_tokenize

__all__ = ["Word2Vec", "Vocab", "simple_tokenize"]
__version__ = "0.1.0"

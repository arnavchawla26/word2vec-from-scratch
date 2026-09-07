"""Command-line interface: `w2v train|similar|analogy|evaluate`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from word2vec.corpus import analogy_test_set, generate_corpus
from word2vec.evaluate import analogy, evaluate_analogies, most_similar
from word2vec.model import Word2Vec, build_vocab_and_tokenize


def _load_raw_sentences(corpus_path: str | None, corpus_seed: int, corpus_repeats: int) -> list[str]:
    if corpus_path:
        text = Path(corpus_path).read_text()
        # One sentence per non-empty line.
        return [line.strip() for line in text.splitlines() if line.strip()]
    return generate_corpus(seed=corpus_seed, repeats=corpus_repeats)


def cmd_train(args: argparse.Namespace) -> int:
    raw_sentences = _load_raw_sentences(args.corpus, args.corpus_seed, args.corpus_repeats)
    print(f"loaded {len(raw_sentences)} raw sentences", file=sys.stderr)

    vocab, tokenized = build_vocab_and_tokenize(
        raw_sentences, min_count=args.min_count, subsample_threshold=args.subsample_threshold
    )
    print(f"vocabulary size: {len(vocab)} words", file=sys.stderr)
    if len(vocab) < 4:
        print("error: vocabulary is too small to train on (need >= 4 words)", file=sys.stderr)
        return 1

    model = Word2Vec(vocab, dim=args.dim, architecture=args.arch, seed=args.seed)

    start = time.time()
    model.fit(
        tokenized,
        epochs=args.epochs,
        window_size=args.window,
        negative_samples=args.negative_samples,
        initial_lr=args.lr,
        subsample=not args.no_subsample,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start
    print(f"trained in {elapsed:.1f}s, final mean loss {model.history.epoch_losses[-1]:.4f}", file=sys.stderr)

    model.save(args.out)
    print(f"saved model to {args.out}.npz / {args.out}.json", file=sys.stderr)
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    model = Word2Vec.load(args.model)
    try:
        results = most_similar(model, args.word, topn=args.topn)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for word, score in results:
        print(f"{word}\t{score:.4f}")
    return 0


def cmd_analogy(args: argparse.Namespace) -> int:
    model = Word2Vec.load(args.model)
    try:
        results = analogy(model, args.a, args.b, args.c, topn=args.topn)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.a} is to {args.b} as {args.c} is to ...")
    for word, score in results:
        print(f"{word}\t{score:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = Word2Vec.load(args.model)
    test_set = analogy_test_set()
    accuracy, results = evaluate_analogies(model, test_set, topn=args.topn)

    skipped = sum(r.skipped for r in results)
    scored = len(results) - skipped
    print(f"accuracy@{args.topn}: {accuracy:.1%} ({scored} scored, {skipped} skipped - OOV)")
    if args.verbose:
        for r in results:
            if r.skipped:
                print(f"  SKIP  {r.a}:{r.b} :: {r.c}:{r.expected}  (out of vocabulary)")
                continue
            mark = "OK  " if r.correct else "MISS"
            guesses = ", ".join(f"{w} ({s:.3f})" for w, s in r.predictions)
            print(f"  {mark}  {r.a}:{r.b} :: {r.c}:{r.expected}  ->  {guesses}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="w2v", description="Skip-gram/CBOW word2vec from scratch.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model and save it to disk")
    p_train.add_argument("--out", default="model", help="output path prefix (writes <out>.npz and <out>.json)")
    p_train.add_argument("--corpus", default=None, help="path to a text file (one sentence per line); default: built-in synthetic corpus")
    p_train.add_argument("--corpus-seed", type=int, default=0, help="seed for the built-in synthetic corpus")
    p_train.add_argument("--corpus-repeats", type=int, default=15, help="repeat count for the built-in synthetic corpus")
    p_train.add_argument("--arch", choices=["skipgram", "cbow"], default="skipgram")
    p_train.add_argument("--dim", type=int, default=64, help="embedding dimensionality")
    p_train.add_argument("--window", type=int, default=3, help="max context window size")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--negative-samples", type=int, default=8)
    p_train.add_argument("--min-count", type=int, default=2, help="drop words occurring fewer than this many times")
    p_train.add_argument("--subsample-threshold", type=float, default=1e-3)
    p_train.add_argument("--no-subsample", action="store_true", help="disable frequent-word subsampling")
    p_train.add_argument("--lr", type=float, default=0.03, help="initial learning rate")
    p_train.add_argument("--seed", type=int, default=0, help="model init / training RNG seed")
    p_train.add_argument("--quiet", action="store_true", help="suppress per-epoch progress output")
    p_train.set_defaults(func=cmd_train)

    p_similar = sub.add_parser("similar", help="find nearest neighbors of a word")
    p_similar.add_argument("model", help="model path prefix (as passed to --out during training)")
    p_similar.add_argument("word")
    p_similar.add_argument("--topn", type=int, default=10)
    p_similar.set_defaults(func=cmd_similar)

    p_analogy = sub.add_parser("analogy", help="solve a is to b as c is to ?")
    p_analogy.add_argument("model")
    p_analogy.add_argument("a")
    p_analogy.add_argument("b")
    p_analogy.add_argument("c")
    p_analogy.add_argument("--topn", type=int, default=5)
    p_analogy.set_defaults(func=cmd_analogy)

    p_eval = sub.add_parser("evaluate", help="score the model against the built-in analogy test set")
    p_eval.add_argument("model")
    p_eval.add_argument("--topn", type=int, default=1, help="count a prediction correct if the answer is in the top N")
    p_eval.add_argument("--verbose", action="store_true", help="print every analogy's predictions")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

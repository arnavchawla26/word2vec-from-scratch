from word2vec.cli import main


def test_train_similar_analogy_evaluate_end_to_end(tmp_path, capsys):
    model_path = str(tmp_path / "tiny_model")

    rc = main(
        [
            "train",
            "--out",
            model_path,
            "--corpus-repeats",
            "3",
            "--epochs",
            "5",
            "--dim",
            "10",
            "--min-count",
            "2",
            "--quiet",
        ]
    )
    assert rc == 0
    assert (tmp_path / "tiny_model.npz").exists()
    assert (tmp_path / "tiny_model.json").exists()
    capsys.readouterr()

    rc = main(["similar", model_path, "king", "--topn", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [line for line in out.strip().splitlines() if line]
    assert len(lines) == 3
    for line in lines:
        word, score = line.split("\t")
        assert 0 <= float(score) <= 1.0001  # cosine similarity, small float slop

    rc = main(["analogy", model_path, "man", "woman", "king", "--topn", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "man is to woman as king is to" in out
    assert len(out.strip().splitlines()) == 4  # header + 3 guesses

    rc = main(["evaluate", model_path, "--topn", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "accuracy@5" in out


def test_similar_unknown_word_returns_error_exit_code(tmp_path, capsys):
    model_path = str(tmp_path / "tiny_model")
    main(["train", "--out", model_path, "--corpus-repeats", "2", "--epochs", "2", "--dim", "6", "--quiet"])
    capsys.readouterr()

    rc = main(["similar", model_path, "definitely_not_a_real_word_zzz"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not in the vocabulary" in err


def test_analogy_unknown_word_returns_error_exit_code(tmp_path, capsys):
    model_path = str(tmp_path / "tiny_model")
    main(["train", "--out", model_path, "--corpus-repeats", "2", "--epochs", "2", "--dim", "6", "--quiet"])
    capsys.readouterr()

    rc = main(["analogy", model_path, "king", "queen", "definitely_not_a_real_word_zzz"])
    assert rc == 1


def test_train_from_custom_corpus_file(tmp_path, capsys):
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        "the cat sat on the mat\n"
        "the dog sat on the rug\n"
        "the cat and the dog played together in the yard\n"
        "the cat chased the dog around the yard\n" * 5
    )
    model_path = str(tmp_path / "custom_model")
    rc = main(["train", "--out", model_path, "--corpus", str(corpus_file), "--epochs", "3", "--dim", "6", "--min-count", "2", "--quiet"])
    assert rc == 0
    assert (tmp_path / "custom_model.npz").exists()


def test_train_rejects_too_small_vocabulary(tmp_path, capsys):
    corpus_file = tmp_path / "tiny.txt"
    corpus_file.write_text("only one line here\n")
    model_path = str(tmp_path / "should_not_exist")
    rc = main(["train", "--out", model_path, "--corpus", str(corpus_file), "--min-count", "5", "--quiet"])
    assert rc == 1
    assert not (tmp_path / "should_not_exist.npz").exists()


def test_cli_requires_a_subcommand(capsys):
    try:
        main([])
        assert False, "expected SystemExit for missing subcommand"
    except SystemExit as exc:
        assert exc.code != 0

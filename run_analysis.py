"""
Headless analysis script — runs the full WEAT pipeline without Jupyter.
Results are saved to results/figures/ and results/csv/.

Usage:
    python run_analysis.py --fasttext-path models/cc.ko.300.bin
    python run_analysis.py --fasttext-path models/cc.ko.300.bin --word2vec-path models/ko.bin
    python run_analysis.py --fasttext-path models/cc.ko.300.bin --output-dir custom_results/
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Korean word embedding gender bias analysis (WEAT + hard debiasing)"
    )
    parser.add_argument(
        "--fasttext-path", type=Path, default=Path("models/cc.ko.300.bin"),
        help="Path to the Korean FastText binary (cc.ko.300.bin)"
    )
    parser.add_argument(
        "--word2vec-path", type=Path, default=None,
        help="(Optional) Path to a Korean Word2Vec binary"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for output figures and CSVs"
    )
    parser.add_argument(
        "--n-permutations", type=int, default=10_000,
        help="Number of permutation test iterations (default: 10000)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip saving figures (only output CSVs)"
    )
    args = parser.parse_args()

    figures_dir = args.output_dir / "figures"
    csv_dir = args.output_dir / "csv"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Imports after arg parsing so --help is fast
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless

    from src.word_sets import get_korean_word_sets, get_all_occupation_words
    from src.load_embeddings import (
        load_fasttext_korean,
        load_word2vec_namuwiki,
        verify_model_words,
        print_coverage_report,
        ModelNotFoundError,
    )
    from src.weat import run_all_occupation_tests
    from src.debiasing import build_debiased_lookup
    from src.visualize import (
        plot_weat_bar_chart,
        plot_pca_scatter,
        plot_cosine_heatmap,
        plot_debiasing_comparison,
        save_results_csv,
        results_to_dataframe,
    )
    import pandas as pd

    word_sets = get_korean_word_sets()

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("\n=== Loading models ===")
    ft_model = load_fasttext_korean(args.fasttext_path)
    print(f"FastText loaded (dim={ft_model.dim})")

    w2v_model = None
    if args.word2vec_path:
        try:
            w2v_model = load_word2vec_namuwiki(args.word2vec_path)
            print(f"Word2Vec loaded (dim={w2v_model.dim})")
        except ModelNotFoundError as e:
            print(f"[WARN] {e}")

    # ------------------------------------------------------------------
    # Coverage check
    # ------------------------------------------------------------------
    print("\n=== Vocabulary coverage ===")
    ft_report = verify_model_words(ft_model, word_sets)
    print_coverage_report(ft_report, model_name="FastText")

    if w2v_model:
        w2v_report = verify_model_words(w2v_model, word_sets)
        print_coverage_report(w2v_report, model_name="Word2Vec")

    # ------------------------------------------------------------------
    # WEAT — before debiasing
    # ------------------------------------------------------------------
    print("\n=== WEAT (before debiasing) ===")
    ft_before = run_all_occupation_tests(ft_model, word_sets, n_permutations=args.n_permutations)
    for r in ft_before:
        print(f"  [FastText] {r}")

    w2v_before = []
    if w2v_model:
        w2v_before = run_all_occupation_tests(w2v_model, word_sets, n_permutations=args.n_permutations)
        for r in w2v_before:
            print(f"  [Word2Vec] {r}")

    # ------------------------------------------------------------------
    # Visualizations (before debiasing)
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n=== Generating visualizations ===")

        all_before = [results_to_dataframe(ft_before, "FastText", "before")]
        if w2v_before:
            all_before.append(results_to_dataframe(w2v_before, "Word2Vec", "before"))
        combined = pd.concat(all_before, ignore_index=True)

        plot_weat_bar_chart(combined, output_path=figures_dir / "effect_size_bar_before.png")
        plot_pca_scatter(ft_model, word_sets, output_path=figures_dir / "pca_scatter_fasttext.png")
        plot_cosine_heatmap(ft_model, word_sets, output_path=figures_dir / "cosine_heatmap_fasttext.png")

    # ------------------------------------------------------------------
    # Hard debiasing
    # ------------------------------------------------------------------
    print("\n=== Hard debiasing ===")
    occ_words = get_all_occupation_words(word_sets)
    ft_lookup = build_debiased_lookup(ft_model, occ_words, word_sets.male_attrs, word_sets.female_attrs)
    print(f"  Debiased {len(ft_lookup)} words")

    # ------------------------------------------------------------------
    # WEAT — after debiasing
    # ------------------------------------------------------------------
    print("\n=== WEAT (after debiasing) ===")
    ft_after = run_all_occupation_tests(
        ft_model, word_sets,
        n_permutations=args.n_permutations,
        vector_override=ft_lookup,
    )
    for r in ft_after:
        print(f"  [FastText] {r}")

    # Comparison
    print("\n전후 비교 (FastText):")
    print(f"  {'테스트':<25} {'전':>8} {'후':>8} {'변화':>8}")
    print("  " + "-" * 52)
    for b, a in zip(ft_before, ft_after):
        delta = a.effect_size - b.effect_size
        print(f"  {b.test_name:<25} {b.effect_size:>+8.3f} {a.effect_size:>+8.3f} {delta:>+8.3f}")

    # ------------------------------------------------------------------
    # Debiasing comparison plot
    # ------------------------------------------------------------------
    if not args.no_plots:
        labels = [r.test_name for r in ft_before]
        plot_debiasing_comparison(
            ft_before, ft_after, labels,
            output_path=figures_dir / "debiasing_comparison_fasttext.png",
        )

        all_combined = pd.concat([
            results_to_dataframe(ft_before, "FastText", "before"),
            results_to_dataframe(ft_after, "FastText", "after"),
        ] + ([results_to_dataframe(w2v_before, "Word2Vec", "before")] if w2v_before else []),
        ignore_index=True)
        plot_weat_bar_chart(all_combined, output_path=figures_dir / "effect_size_bar_all.png")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    print("\n=== Saving results ===")
    save_results_csv(ft_before, "fasttext", csv_dir / "weat_results_fasttext_before.csv", phase="before")
    save_results_csv(ft_after,  "fasttext", csv_dir / "weat_results_fasttext_after.csv",  phase="after")
    if w2v_before:
        save_results_csv(w2v_before, "word2vec", csv_dir / "weat_results_word2vec_before.csv", phase="before")

    print("\nDone.")


if __name__ == "__main__":
    main()

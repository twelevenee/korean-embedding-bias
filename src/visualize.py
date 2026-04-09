"""
Visualization utilities for Korean word embedding gender bias analysis.

IMPORTANT: Korean Hangul requires a Hangul-capable font. Matplotlib's default
font renders Korean characters as empty boxes ("tofu"). _set_korean_font()
detects the OS and configures the correct font automatically. It is called at
module import time so all plots in this module inherit it.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Font setup (must run before any plot is created)
# ---------------------------------------------------------------------------

def _set_korean_font() -> None:
    """Detect OS and configure a Hangul-capable font for Matplotlib."""
    font_map = {
        "Darwin": "AppleGothic",
        "Windows": "Malgun Gothic",
        "Linux": "NanumGothic",
    }
    font = font_map.get(platform.system(), "NanumGothic")
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False  # prevent minus sign from rendering as box


_set_korean_font()


# ---------------------------------------------------------------------------
# WEAT effect size bar chart
# ---------------------------------------------------------------------------

def plot_weat_bar_chart(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "WEAT Effect Size by Occupation Category",
) -> plt.Figure:
    """
    Grouped bar chart of WEAT effect sizes.

    Expected columns in results_df:
      test_name, effect_size, p_value, model_name, phase
    where phase is 'before' or 'after' (debiasing).

    Bars solid = significant (p < 0.05), hatched = not significant.
    Horizontal line at d = 0.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    models = results_df["model_name"].unique()
    phases = results_df["phase"].unique() if "phase" in results_df.columns else ["before"]
    tests = results_df["test_name"].unique()

    n_groups = len(tests)
    n_bars = len(models) * len(phases)
    bar_width = 0.7 / n_bars
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_bars))

    x = np.arange(n_groups)

    # Build (model, phase) combos that actually have data, ordered before → after
    phase_order = {"before": 0, "after": 1}
    combos = results_df.groupby(["model_name", "phase"]).size().reset_index()[["model_name", "phase"]]
    combos = sorted(
        combos.itertuples(index=False, name=None),
        key=lambda t: (t[0], phase_order.get(t[1], 99))
    )

    n_bars = len(combos)
    bar_width = 0.7 / n_bars
    colors = plt.cm.Set2(np.linspace(0, 0.8, max(n_bars, 1)))

    bar_idx = 0
    legend_handles = []

    for model_name, phase in combos:
        subset = results_df[
            (results_df["model_name"] == model_name)
            & (results_df["phase"] == phase)
        ]

        offset = (bar_idx - n_bars / 2 + 0.5) * bar_width
        color = colors[bar_idx]
        label = f"{model_name} ({phase})" if len(phases) > 1 else model_name

        for i, test in enumerate(tests):
            row = subset[subset["test_name"] == test]
            if row.empty:
                continue
            d = row["effect_size"].values[0]
            p = row["p_value"].values[0]
            hatch = "" if p < 0.05 else "///"
            ax.bar(
                x[i] + offset, d,
                width=bar_width * 0.9,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=0.8,
            )

        patch = mpatches.Patch(facecolor=color, edgecolor="black", label=label)
        legend_handles.append(patch)
        bar_idx += 1

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=11)
    ax.set_ylabel("Effect Size (d)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(handles=legend_handles, fontsize=10)

    # Significance note
    hatch_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                  hatch="///", label="p ≥ 0.05 (not significant)")
    solid_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="p < 0.05 (significant)")
    ax.legend(handles=legend_handles + [solid_patch, hatch_patch], fontsize=9, loc="upper right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# PCA scatter plot
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    model,
    word_sets,
    title: str = "PCA of Korean Gender + Occupation Word Vectors",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    PCA 2D scatter of all gender attribute words + occupation words.

    Color coding:
      Blue  = male (attrs and occupations)
      Red   = female (attrs and occupations)
      Green = neutral occupations

    Markers:
      Circle (o)   = gender attribute words
      Square (s)   = occupation words
    """
    from src.word_sets import get_all_words

    all_words = get_all_words(word_sets)
    vecs, words_found = [], []
    for w in all_words:
        v = model.get_vector_safe(w)
        if v is not None:
            vecs.append(v)
            words_found.append(w)

    if len(vecs) < 3:
        raise ValueError("Need at least 3 words with vectors for PCA scatter.")

    matrix = np.array(vecs)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)

    # Category assignment for color + marker
    male_attr_set = set(word_sets.male_attrs)
    female_attr_set = set(word_sets.female_attrs)
    male_occ_set = set(word_sets.male_occupations)
    female_occ_set = set(word_sets.female_occupations)
    neutral_occ_set = set(word_sets.neutral_occupations)

    def _style(word):
        if word in male_attr_set:
            return "royalblue", "o", 80
        if word in female_attr_set:
            return "crimson", "o", 80
        if word in male_occ_set:
            return "royalblue", "s", 70
        if word in female_occ_set:
            return "crimson", "s", 70
        if word in neutral_occ_set:
            return "forestgreen", "D", 70
        return "gray", "x", 50

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, word in enumerate(words_found):
        color, marker, size = _style(word)
        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                   s=size, alpha=0.85, zorder=3)
        ax.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    # Legend
    legend_elements = [
        plt.scatter([], [], c="royalblue", marker="o", s=80, label="남성 속성어"),
        plt.scatter([], [], c="crimson", marker="o", s=80, label="여성 속성어"),
        plt.scatter([], [], c="royalblue", marker="s", s=70, label="전통 남성 직종"),
        plt.scatter([], [], c="crimson", marker="s", s=70, label="전통 여성 직종"),
        plt.scatter([], [], c="forestgreen", marker="D", s=70, label="중립/전문직"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="best")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Cosine similarity heatmap
# ---------------------------------------------------------------------------

def plot_cosine_heatmap(
    model,
    word_sets,
    title: str = "코사인 유사도: 직업어 × 젠더 속성어",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Seaborn heatmap of cosine similarities: occupations (rows) × gender attrs (cols).

    Color: RdBu_r diverging colormap centered at 0.
    Rows sorted by average association to male attribute words (most male at top).
    Column annotations distinguish male vs female attribute groups.
    """
    from src.word_sets import get_all_occupation_words

    gender_words = word_sets.male_attrs + word_sets.female_attrs
    occ_words = get_all_occupation_words(word_sets)

    # Build filtered lists (only words in model)
    gender_vecs, gender_found = [], []
    for w in gender_words:
        v = model.get_vector_safe(w)
        if v is not None:
            gender_vecs.append(v)
            gender_found.append(w)

    occ_vecs, occ_found = [], []
    for w in occ_words:
        v = model.get_vector_safe(w)
        if v is not None:
            occ_vecs.append(v)
            occ_found.append(w)

    if not gender_found or not occ_found:
        raise ValueError("Not enough words found in model for heatmap.")

    # Cosine sim matrix: shape (n_occ, n_gender)
    G = np.array(gender_vecs)
    O = np.array(occ_vecs)
    sim_matrix = O @ G.T  # pre-normalized → dot = cosine

    # Sort rows by mean similarity to male attrs
    n_male = sum(1 for w in gender_found if w in word_sets.male_attrs)
    male_mean = sim_matrix[:, :n_male].mean(axis=1)
    sort_idx = np.argsort(-male_mean)
    sim_matrix = sim_matrix[sort_idx]
    occ_found_sorted = [occ_found[i] for i in sort_idx]

    df_heat = pd.DataFrame(sim_matrix, index=occ_found_sorted, columns=gender_found)

    # Column color annotations
    col_colors = ["#4472C4" if w in word_sets.male_attrs else "#C0504D"
                  for w in gender_found]

    fig, ax = plt.subplots(figsize=(max(10, len(gender_found) * 0.9), max(6, len(occ_found) * 0.5)))
    sns.heatmap(
        df_heat, ax=ax, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.4, linecolor="lightgray",
    )

    # Color-code column labels
    for tick, color in zip(ax.get_xticklabels(), col_colors):
        tick.set_color(color)
        tick.set_fontsize(9)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("젠더 속성어  (파랑=남성, 빨강=여성)", fontsize=10)
    ax.set_ylabel("직업어", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Per-word association score bar chart
# ---------------------------------------------------------------------------

def plot_per_word_scores(
    results: list,
    word_sets,
    title: str = "직업어별 젠더 연관 점수 s(w, 남성, 여성)",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of individual s(w, A, B) scores for every occupation word.

    Shows how much each word leans male (positive) or female (negative).
    Bars colored by occupation category; sorted by score within each test.
    One subplot per WEAT test result.

    Args:
        results: list of WEATResult (from run_all_occupation_tests)
        word_sets: WEATWordSets (for category lookup)
    """
    from src.word_sets import occupation_category, CATEGORY_LABELS

    category_colors = {
        "male_occupations":   "#4472C4",   # blue
        "female_occupations": "#C0504D",   # red
        "neutral_occupations": "#70AD47",  # green
        "unknown": "gray",
    }

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, max(5, 0.45 * max(
        len(r.x_words) + len(r.y_words) for r in results
    ))), squeeze=False)

    for ax, result in zip(axes[0], results):
        words = result.x_words + result.y_words
        scores = list(result.x_scores) + list(result.y_scores)

        # Sort by score descending
        order = sorted(range(len(words)), key=lambda i: scores[i], reverse=True)
        sorted_words = [words[i] for i in order]
        sorted_scores = [scores[i] for i in order]
        colors = [
            category_colors[occupation_category(w, word_sets)]
            for w in sorted_words
        ]

        bars = ax.barh(
            range(len(sorted_words)), sorted_scores,
            color=colors, edgecolor="white", linewidth=0.5, height=0.7,
        )

        # Value labels
        for i, (score, bar) in enumerate(zip(sorted_scores, bars)):
            x_pos = score + (0.003 if score >= 0 else -0.003)
            ha = "left" if score >= 0 else "right"
            ax.text(x_pos, i, f"{score:+.3f}", va="center", ha=ha, fontsize=7.5)

        ax.set_yticks(range(len(sorted_words)))
        ax.set_yticklabels(sorted_words, fontsize=9)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax.set_xlabel("s(w, 남성속성, 여성속성)", fontsize=9)
        ax.set_title(result.test_name, fontsize=10, pad=8)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()

    # Legend
    legend_handles = [
        mpatches.Patch(color=category_colors["male_occupations"],   label="전통 남성직종"),
        mpatches.Patch(color=category_colors["female_occupations"],  label="전통 여성직종"),
        mpatches.Patch(color=category_colors["neutral_occupations"], label="중립/전문직"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Debiasing before/after comparison
# ---------------------------------------------------------------------------

def plot_debiasing_comparison(
    before_results: list,
    after_results: list,
    test_labels: List[str],
    title: str = "편향 완화 전후 WEAT Effect Size 비교",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Side-by-side bar chart showing WEAT effect sizes before and after debiasing.
    Arrow annotations show the direction and magnitude of change per test.
    """
    n = len(test_labels)
    x = np.arange(n)
    bar_width = 0.35

    before_d = [r.effect_size for r in before_results]
    after_d = [r.effect_size for r in after_results]

    fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 6))

    bars_b = ax.bar(x - bar_width / 2, before_d, bar_width,
                    label="debiasing 전", color="#4472C4", alpha=0.85, edgecolor="black")
    bars_a = ax.bar(x + bar_width / 2, after_d, bar_width,
                    label="debiasing 후", color="#ED7D31", alpha=0.85, edgecolor="black")

    # Arrow annotations showing change
    for i in range(n):
        y_start = before_d[i]
        y_end = after_d[i]
        delta = y_end - y_start
        if abs(delta) > 0.05:
            ax.annotate(
                f"Δ{delta:+.2f}",
                xy=(x[i] + bar_width / 2, y_end),
                xytext=(x[i], max(abs(y_start), abs(y_end)) * 1.15),
                fontsize=8, ha="center", color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
            )

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(test_labels, fontsize=10)
    ax.set_ylabel("Effect Size (d)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(
    results: list,
    model_name: str,
    output_path: Path,
    phase: str = "before",
) -> pd.DataFrame:
    """
    Serialize a list of WEATResult objects to CSV.

    Columns: model, phase, test_name, effect_size, p_value, significant,
             mean_x, mean_y, std_all, n_x_words, n_y_words
    """
    rows = []
    for r in results:
        rows.append({
            "model": model_name,
            "phase": phase,
            "test_name": r.test_name,
            "effect_size": round(r.effect_size, 4),
            "p_value": round(r.p_value, 4),
            "significant": r.significant,
            "mean_x": round(r.mean_x, 4),
            "mean_y": round(r.mean_y, 4),
            "std_all": round(r.std_all, 4),
            "n_x_words": len(r.x_words),
            "n_y_words": len(r.y_words),
            "x_words": "|".join(r.x_words),
            "y_words": "|".join(r.y_words),
        })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")
    return df


def results_to_dataframe(
    results: list,
    model_name: str,
    phase: str = "before",
) -> pd.DataFrame:
    """Convert a list of WEATResult to a DataFrame (without saving)."""
    rows = [{
        "model_name": model_name,
        "phase": phase,
        "test_name": r.test_name,
        "effect_size": r.effect_size,
        "p_value": r.p_value,
        "significant": r.significant,
        "mean_x": r.mean_x,
        "mean_y": r.mean_y,
    } for r in results]
    return pd.DataFrame(rows)


def plot_alpha_tradeoff(
    alpha_results: dict,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot WEAT effect size vs debiasing strength (alpha) for each test.

    Visualizes the tradeoff between bias reduction and semantic distortion
    introduced by partial debiasing (alpha parameter in neutralize()).

    Args:
        alpha_results: dict mapping alpha (float) → list of WEATResult
                       e.g. {0.0: [...], 0.1: [...], ..., 1.0: [...]}
        output_path:   if provided, saves the figure to this path

    Returns:
        matplotlib Figure
    """
    alphas = sorted(alpha_results.keys())
    # Collect all test names in order from the first alpha entry
    test_names = [r.test_name for r in alpha_results[alphas[0]]]

    # Build {test_name: [d at each alpha]} mapping
    effect_by_test: Dict[str, List[float]] = {name: [] for name in test_names}
    sig_by_test: Dict[str, List[bool]] = {name: [] for name in test_names}
    for a in alphas:
        results_at_a = {r.test_name: r for r in alpha_results[a]}
        for name in test_names:
            r = results_at_a.get(name)
            effect_by_test[name].append(r.effect_size if r else float("nan"))
            sig_by_test[name].append(r.significant if r else False)

    colors = ["#2166ac", "#d6604d", "#4dac26"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for (name, d_values), color, marker in zip(effect_by_test.items(), colors, markers):
        ax.plot(alphas, d_values, color=color, marker=marker,
                linewidth=2, markersize=6, label=name)
        # Mark significant points with filled markers, non-significant with open
        for a, d, sig in zip(alphas, d_values, sig_by_test[name]):
            if sig:
                ax.plot(a, d, marker=marker, color=color,
                        markersize=9, markerfacecolor=color, zorder=5)
            else:
                ax.plot(a, d, marker=marker, color=color,
                        markersize=9, markerfacecolor="white",
                        markeredgewidth=1.5, zorder=5)

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
               label="α=1.0 (hard debiasing)")

    # Shade overcorrection region (where neutral-vs-female flips sign)
    nf_values = effect_by_test.get("중립직종 vs 여성직종", [])
    if nf_values:
        sign_flip_alpha = None
        for i in range(len(alphas) - 1):
            if nf_values[i] > 0 and nf_values[i + 1] <= 0:
                # Linear interpolation
                sign_flip_alpha = alphas[i] + (alphas[i + 1] - alphas[i]) * (
                    nf_values[i] / (nf_values[i] - nf_values[i + 1])
                )
                break
        if sign_flip_alpha is not None:
            ax.axvspan(sign_flip_alpha, max(alphas), alpha=0.06, color="red",
                       label=f"과교정 영역 (α > {sign_flip_alpha:.2f})")
            ax.axvline(sign_flip_alpha, color="red", linewidth=1.2,
                       linestyle="--", alpha=0.6)

    ax.set_xlabel("Debiasing 강도 (α)", fontsize=12)
    ax.set_ylabel("Effect Size (d)", fontsize=12)
    ax.set_title("부분 디바이어싱 트레이드오프: α별 WEAT Effect Size\n"
                 "(채운 마커 = p < 0.05 유의)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig

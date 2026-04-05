"""
WEAT (Word Embedding Association Test) implementation.
Caliskan et al. (2017) "Semantics derived automatically from language corpora
contain human-like biases." Science 356(6334):183-186.

All vector inputs are expected to be L2-normalized (unit vectors).
Under this assumption cos(u, v) = u · v, so np.dot replaces all cosine calls.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.word_sets import WEATWordSets, WEAT_TESTS


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WEATResult:
    test_name: str
    effect_size: float        # Cohen's d analogue
    p_value: float
    mean_x: float             # mean s(x, A, B) for target set X
    mean_y: float             # mean s(y, A, B) for target set Y
    std_all: float            # std over X ∪ Y scores
    n_permutations: int
    x_words: List[str]
    y_words: List[str]
    x_scores: np.ndarray      # per-word s() scores for X
    y_scores: np.ndarray      # per-word s() scores for Y

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        sig = "*" if self.significant else ""
        return (
            f"{self.test_name}: d={self.effect_size:+.3f}, "
            f"p={self.p_value:.4f}{sig}"
        )


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def association_score(
    word_vec: np.ndarray,
    attr_a_vecs: np.ndarray,
    attr_b_vecs: np.ndarray,
) -> float:
    """
    s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b)

    Means are computed independently for A and B so that unequal set sizes
    (which can arise after OOV filtering) are handled correctly.
    All inputs must be L2-normalized; cosine similarity = dot product.

    Args:
        word_vec:   shape (d,)
        attr_a_vecs: shape (|A|, d)
        attr_b_vecs: shape (|B|, d)

    Returns:
        scalar association score
    """
    mean_a = np.dot(attr_a_vecs, word_vec).mean()
    mean_b = np.dot(attr_b_vecs, word_vec).mean()
    return float(mean_a - mean_b)


def weat_effect_size(
    target_x_vecs: np.ndarray,
    target_y_vecs: np.ndarray,
    attr_a_vecs: np.ndarray,
    attr_b_vecs: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute WEAT effect size d.

    d = (mean_{x in X} s(x,A,B) - mean_{y in Y} s(y,A,B))
        / std_{z in X∪Y} s(z,A,B)

    Args:
        target_x_vecs: shape (|X|, d)
        target_y_vecs: shape (|Y|, d)
        attr_a_vecs:   shape (|A|, d)
        attr_b_vecs:   shape (|B|, d)

    Returns:
        (effect_size, x_scores, y_scores)
    """
    x_scores = np.array([
        association_score(v, attr_a_vecs, attr_b_vecs)
        for v in target_x_vecs
    ])
    y_scores = np.array([
        association_score(v, attr_a_vecs, attr_b_vecs)
        for v in target_y_vecs
    ])

    all_scores = np.concatenate([x_scores, y_scores])
    std = all_scores.std()

    if std < 1e-10:
        # degenerate case: all scores identical
        return 0.0, x_scores, y_scores

    d = (x_scores.mean() - y_scores.mean()) / std
    return float(d), x_scores, y_scores


def permutation_test(
    target_x_vecs: np.ndarray,
    target_y_vecs: np.ndarray,
    attr_a_vecs: np.ndarray,
    attr_b_vecs: np.ndarray,
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> float:
    """
    One-sided permutation test on the WEAT test statistic.

    Null hypothesis: random partition of X ∪ Y into two same-size sets.
    p-value = fraction of permutations where test statistic >= observed.

    Pre-computes all association scores once, then shuffles the index array
    for each permutation (avoids recomputing cosine sims per permutation).
    Runtime is O(n_permutations * (|X| + |Y|)) — well under 2s for typical sets.
    """
    n_x = len(target_x_vecs)
    n_y = len(target_y_vecs)
    n_total = n_x + n_y

    # Pre-compute all association scores
    all_vecs = np.concatenate([target_x_vecs, target_y_vecs], axis=0)
    all_scores = np.array([
        association_score(v, attr_a_vecs, attr_b_vecs)
        for v in all_vecs
    ])

    observed_stat = all_scores[:n_x].mean() - all_scores[n_x:].mean()

    rng = np.random.default_rng(random_state)
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_x = all_scores[perm[:n_x]]
        perm_y = all_scores[perm[n_x:]]
        stat = perm_x.mean() - perm_y.mean()
        if stat >= observed_stat:
            count_ge += 1

    return count_ge / n_permutations


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def run_weat(
    model,
    target_x: List[str],
    target_y: List[str],
    attr_a: List[str],
    attr_b: List[str],
    test_name: str = "",
    n_permutations: int = 10_000,
    vector_override: Optional[Dict[str, np.ndarray]] = None,
) -> WEATResult:
    """
    Run a single WEAT test.

    Args:
        model: EmbeddingModel instance (from load_embeddings.py)
        target_x, target_y: occupation word groups
        attr_a, attr_b: gender attribute word groups (male, female)
        test_name: label for this test
        n_permutations: permutation test iterations
        vector_override: dict {word: vector} to substitute over model vectors,
                         used for post-debiasing analysis without modifying model

    Raises:
        ValueError if any word set has fewer than 5 words after OOV filtering.
    """

    from src.word_sets import WORD_ALIASES

    def _get_vec(word: str) -> Tuple[str, Optional[np.ndarray]]:
        """Return (resolved_word, vector). Tries aliases if primary form is OOV."""
        if vector_override and word in vector_override:
            return word, vector_override[word]
        v = model.get_vector_safe(word)
        if v is not None:
            return word, v
        # Try aliases
        for alt in WORD_ALIASES.get(word, []):
            v = model.get_vector_safe(alt)
            if v is not None:
                return alt, v
        return word, None

    def _filter(words: List[str], label: str) -> Tuple[List[str], np.ndarray]:
        found_words = []
        vecs = []
        for w in words:
            resolved, v = _get_vec(w)
            if v is not None:
                found_words.append(resolved)
                vecs.append(v)
        if len(found_words) < 5:
            raise ValueError(
                f"WEAT '{test_name}': word set '{label}' has only "
                f"{len(found_words)}/{len(words)} words in vocabulary. "
                f"Missing: {[w for w in words if _get_vec(w)[1] is None]}. "
                f"Need at least 5 words for a valid WEAT test."
            )
        return found_words, np.array(vecs)

    x_words, x_vecs = _filter(target_x, "target_x")
    y_words, y_vecs = _filter(target_y, "target_y")
    a_words, a_vecs = _filter(attr_a, "attr_a")
    b_words, b_vecs = _filter(attr_b, "attr_b")

    d, x_scores, y_scores = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)
    p = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations)

    return WEATResult(
        test_name=test_name,
        effect_size=d,
        p_value=p,
        mean_x=float(x_scores.mean()),
        mean_y=float(y_scores.mean()),
        std_all=float(np.concatenate([x_scores, y_scores]).std()),
        n_permutations=n_permutations,
        x_words=x_words,
        y_words=y_words,
        x_scores=x_scores,
        y_scores=y_scores,
    )


def run_all_occupation_tests(
    model,
    word_sets: WEATWordSets,
    n_permutations: int = 10_000,
    vector_override: Optional[Dict[str, np.ndarray]] = None,
) -> List[WEATResult]:
    """
    Run all three occupation WEAT tests defined in WEAT_TESTS.

    Tests:
      1. 남성직종 vs 여성직종  (male vs female occupations)
      2. 중립직종 vs 남성직종  (neutral vs male occupations)
      3. 중립직종 vs 여성직종  (neutral vs female occupations)

    All tests use male_attrs as A and female_attrs as B.
    """
    target_sets = {
        "male_occupations": word_sets.male_occupations,
        "female_occupations": word_sets.female_occupations,
        "neutral_occupations": word_sets.neutral_occupations,
    }

    results = []
    for x_key, y_key, name in WEAT_TESTS:
        try:
            result = run_weat(
                model=model,
                target_x=target_sets[x_key],
                target_y=target_sets[y_key],
                attr_a=word_sets.male_attrs,
                attr_b=word_sets.female_attrs,
                test_name=name,
                n_permutations=n_permutations,
                vector_override=vector_override,
            )
            results.append(result)
        except ValueError as e:
            print(f"[WEAT] Skipping '{name}': {e}")

    return results

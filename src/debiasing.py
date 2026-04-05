"""
Hard debiasing for word embeddings.
Bolukbasi et al. (2016) "Man is to Computer Programmer as Woman is to Homemaker?
Debiasing Word Embeddings." NeurIPS 2016.

Pipeline:
  1. compute_gender_direction  — PCA on difference vectors → unit vector g
  2. neutralize                — remove gender component from occupation words
  3. equalize                  — make gender word pairs equidistant from midpoint
  4. hard_debias               — full pipeline
  5. build_debiased_lookup     — returns {word: debiased_vec} for WEAT override

Design: the original EmbeddingModel is never mutated. All functions return
plain np.ndarray objects. Debiased vectors are fed into run_weat() via its
vector_override parameter for clean before/after comparison.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Gender direction
# ---------------------------------------------------------------------------

def compute_gender_direction(
    male_vecs: np.ndarray,
    female_vecs: np.ndarray,
    method: str = "pca",
) -> np.ndarray:
    """
    Identify the gender subspace direction.

    PCA method (Bolukbasi 2016 default):
      1. For each (male_i, female_i) pair, compute the difference vector.
      2. Stack all difference vectors and run PCA.
      3. First principal component (max variance direction) = gender direction g.
      4. L2-normalize g to unit length — required for correct projection math.

    Mean-diff fallback:
      g = normalize(mean(male_vecs) - mean(female_vecs))

    Args:
        male_vecs:   shape (n, d), L2-normalized input vectors
        female_vecs: shape (n, d), L2-normalized input vectors — must match male_vecs
        method: "pca" (recommended) or "mean_diff"

    Returns:
        Unit vector g of shape (d,) pointing in the gender direction.
    """
    if len(male_vecs) != len(female_vecs):
        raise ValueError(
            f"male_vecs and female_vecs must have the same length for PCA method, "
            f"got {len(male_vecs)} vs {len(female_vecs)}"
        )

    if method == "pca":
        diffs = male_vecs - female_vecs   # shape (n, d)
        pca = PCA(n_components=1)
        pca.fit(diffs)
        g = pca.components_[0]           # first PC: direction of max variance

    elif method == "mean_diff":
        g = male_vecs.mean(axis=0) - female_vecs.mean(axis=0)

    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'pca' or 'mean_diff'.")

    # L2-normalize to unit length — critical for correct projection math
    norm = np.linalg.norm(g)
    if norm < 1e-10:
        raise ValueError("Gender direction vector is nearly zero — check input vectors.")
    return g / norm


# ---------------------------------------------------------------------------
# Neutralize
# ---------------------------------------------------------------------------

def neutralize(
    word_vecs: np.ndarray,
    gender_direction: np.ndarray,
) -> np.ndarray:
    """
    Remove the gender component from each word vector.

    v_debiased = v - (v · g) * g
    Then L2-renormalize to keep unit norm.

    Args:
        word_vecs:        shape (n, d), L2-normalized
        gender_direction: shape (d,), unit vector g

    Returns:
        Debiased vectors of shape (n, d), L2-normalized.
    """
    g = gender_direction
    # projections: shape (n,)
    projections = word_vecs @ g
    # remove gender component: shape (n, d)
    debiased = word_vecs - np.outer(projections, g)

    # L2-renormalize
    norms = np.linalg.norm(debiased, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # avoid division by zero
    return debiased / norms


# ---------------------------------------------------------------------------
# Equalize
# ---------------------------------------------------------------------------

def equalize(
    male_vecs: np.ndarray,
    female_vecs: np.ndarray,
    gender_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equalize gender word pairs so they are equidistant from the gender axis midpoint.

    For each pair (e+, e-):
      mean_vec    = (e+ + e-) / 2
      neutralized = mean_vec - (mean_vec · g) * g
      scalar      = sqrt(max(0, 1 - ||neutralized||²))
      e+_new      = normalize(neutralized + scalar * g)
      e-_new      = normalize(neutralized - scalar * g)

    This ensures cos(e+_new, g) == -cos(e-_new, g) for each pair.

    Args:
        male_vecs:        shape (n, d), L2-normalized
        female_vecs:      shape (n, d), L2-normalized
        gender_direction: shape (d,), unit vector g

    Returns:
        (male_equalized, female_equalized), each shape (n, d), L2-normalized.
    """
    g = gender_direction
    male_eq = np.zeros_like(male_vecs)
    female_eq = np.zeros_like(female_vecs)

    for i, (e_plus, e_minus) in enumerate(zip(male_vecs, female_vecs)):
        mean_vec = (e_plus + e_minus) / 2.0
        # Remove gender component from mean
        mean_proj = np.dot(mean_vec, g)
        mu = mean_vec - mean_proj * g

        mu_norm_sq = np.dot(mu, mu)
        scalar = np.sqrt(max(0.0, 1.0 - mu_norm_sq))

        e_plus_new = mu + scalar * g
        e_minus_new = mu - scalar * g

        # L2-normalize
        norm_p = np.linalg.norm(e_plus_new)
        norm_m = np.linalg.norm(e_minus_new)
        male_eq[i] = e_plus_new / norm_p if norm_p > 1e-10 else e_plus_new
        female_eq[i] = e_minus_new / norm_m if norm_m > 1e-10 else e_minus_new

    return male_eq, female_eq


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def hard_debias(
    model,
    occupation_words: List[str],
    male_words: List[str],
    female_words: List[str],
    gender_direction_method: str = "pca",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Full Bolukbasi hard debiasing pipeline.

    Steps:
      1. Look up vectors from model (OOV words are silently skipped)
      2. Compute gender direction g from matched male/female pairs
      3. Neutralize occupation word vectors
      4. Equalize gender word pairs

    Returns:
        (occ_debiased, male_eq, female_eq, g, occ_words_found, male_found, female_found)
    """
    # Collect vectors, skipping OOV
    occ_vecs, occ_found = _lookup(model, occupation_words)
    male_vecs, male_found = _lookup(model, male_words)
    female_vecs, female_found = _lookup(model, female_words)

    # Pair male/female (use min length to align pairs)
    n_pairs = min(len(male_vecs), len(female_vecs))
    if n_pairs < 2:
        raise ValueError(
            f"Need at least 2 gender word pairs to compute gender direction, "
            f"got {n_pairs}."
        )
    male_vecs_paired = male_vecs[:n_pairs]
    female_vecs_paired = female_vecs[:n_pairs]
    male_found_paired = male_found[:n_pairs]
    female_found_paired = female_found[:n_pairs]

    g = compute_gender_direction(male_vecs_paired, female_vecs_paired, method=gender_direction_method)
    occ_debiased = neutralize(occ_vecs, g)
    male_eq, female_eq = equalize(male_vecs_paired, female_vecs_paired, g)

    return occ_debiased, male_eq, female_eq, g, occ_found, male_found_paired, female_found_paired


def build_debiased_lookup(
    model,
    occupation_words: List[str],
    male_words: List[str],
    female_words: List[str],
    gender_direction_method: str = "pca",
) -> Dict[str, np.ndarray]:
    """
    Run hard_debias and return a {word: debiased_vector} dict.

    This dict is passed as vector_override to run_weat() so WEAT uses
    debiased vectors without mutating the original model.

    Returns:
        dict mapping each word (occupation + gender attrs) to its debiased vector.
    """
    occ_debiased, male_eq, female_eq, g, occ_found, male_found, female_found = hard_debias(
        model, occupation_words, male_words, female_words, gender_direction_method
    )

    lookup: Dict[str, np.ndarray] = {}
    for word, vec in zip(occ_found, occ_debiased):
        lookup[word] = vec
    for word, vec in zip(male_found, male_eq):
        lookup[word] = vec
    for word, vec in zip(female_found, female_eq):
        lookup[word] = vec

    return lookup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lookup(model, words: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Fetch L2-normalized vectors for words present in model. Skip OOV."""
    found_words = []
    vecs = []
    for w in words:
        v = model.get_vector_safe(w)
        if v is not None:
            found_words.append(w)
            vecs.append(v)
    if not found_words:
        raise ValueError(f"None of the words {words} are in the model vocabulary.")
    return np.array(vecs), found_words

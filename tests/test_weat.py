"""
Unit tests for src/weat.py using synthetic toy embeddings.
No model files are required.
"""

import numpy as np
import pytest
from src.weat import (
    association_score,
    weat_effect_size,
    permutation_test,
    WEATResult,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_vecs(*vecs):
    return np.array([_normalize(np.array(v, dtype=float)) for v in vecs])


# ---------------------------------------------------------------------------
# association_score
# ---------------------------------------------------------------------------

class TestAssociationScore:
    def test_positive_association(self):
        # word points toward A cluster, not B
        word = _normalize(np.array([1.0, 0.0, 0.0]))
        a_vecs = _make_vecs([1.0, 0.1, 0.0], [0.9, 0.2, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0, 0.0], [-0.9, -0.1, 0.0])
        score = association_score(word, a_vecs, b_vecs)
        assert score > 0

    def test_negative_association(self):
        word = _normalize(np.array([-1.0, 0.0, 0.0]))
        a_vecs = _make_vecs([1.0, 0.0, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0, 0.0])
        score = association_score(word, a_vecs, b_vecs)
        assert score < 0

    def test_zero_association(self):
        # word orthogonal to both A and B
        word = _normalize(np.array([0.0, 1.0, 0.0]))
        a_vecs = _make_vecs([1.0, 0.0, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0, 0.0])
        score = association_score(word, a_vecs, b_vecs)
        assert abs(score) < 1e-6

    def test_unequal_set_sizes(self):
        # A has 3 words, B has 1 — means computed independently
        word = _normalize(np.array([1.0, 0.0]))
        a_vecs = _make_vecs([1.0, 0.0], [0.9, 0.1], [0.8, 0.2])
        b_vecs = _make_vecs([-1.0, 0.0])
        score = association_score(word, a_vecs, b_vecs)
        # mean of dot products with A minus dot with B — should be positive
        assert score > 0

    def test_manual_calculation(self):
        # Verify against hand-computed value
        word = np.array([1.0, 0.0])
        a = np.array([[1.0, 0.0], [0.0, 1.0]])  # already unit norm
        b = np.array([[-1.0, 0.0]])
        # mean cos(word, A) = (1 + 0) / 2 = 0.5
        # mean cos(word, B) = -1
        # expected = 0.5 - (-1) = 1.5
        score = association_score(word, a, b)
        assert abs(score - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# weat_effect_size
# ---------------------------------------------------------------------------

class TestWEATEffectSize:
    def _biased_setup(self):
        """X vectors strongly aligned with A; Y vectors aligned with B."""
        a_vecs = _make_vecs([1.0, 0.0, 0.0], [0.9, 0.1, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0, 0.0], [-0.9, -0.1, 0.0])
        x_vecs = _make_vecs([0.95, 0.05, 0.0], [0.85, 0.1, 0.05],
                             [0.9, 0.0, 0.1], [0.8, 0.2, 0.0], [0.88, 0.12, 0.0])
        y_vecs = _make_vecs([-0.95, -0.05, 0.0], [-0.85, -0.1, 0.05],
                             [-0.9, 0.0, 0.1], [-0.8, -0.2, 0.0], [-0.88, -0.12, 0.0])
        return x_vecs, y_vecs, a_vecs, b_vecs

    def test_large_positive_effect_size(self):
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        d, _, _ = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)
        assert d > 1.0

    def test_zero_effect_size(self):
        # Symmetric: X and Y are mirror images, so d ≈ 0
        a_vecs = _make_vecs([1.0, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0])
        x_vecs = _make_vecs([0.5, 0.5], [-0.5, 0.5], [0.0, 1.0], [0.3, 0.7], [-0.3, 0.7])
        y_vecs = x_vecs.copy()  # identical sets → difference = 0
        d, _, _ = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)
        assert abs(d) < 1e-6

    def test_returns_score_arrays(self):
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        d, x_scores, y_scores = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)
        assert len(x_scores) == len(x_vecs)
        assert len(y_scores) == len(y_vecs)

    def test_negative_effect_size(self):
        # Swap X and Y roles → d flips sign
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        d_pos, _, _ = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)
        d_neg, _, _ = weat_effect_size(y_vecs, x_vecs, a_vecs, b_vecs)
        assert d_neg < -1.0
        assert abs(d_pos + d_neg) < 1e-6  # perfect antisymmetry


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def _biased_setup(self):
        a_vecs = _make_vecs([1.0, 0.0, 0.0], [0.9, 0.1, 0.0],
                             [0.95, 0.05, 0.0], [0.85, 0.15, 0.0], [0.8, 0.2, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0, 0.0], [-0.9, -0.1, 0.0],
                             [-0.95, -0.05, 0.0], [-0.85, -0.15, 0.0], [-0.8, -0.2, 0.0])
        x_vecs = _make_vecs([0.95, 0.05, 0.0], [0.85, 0.1, 0.05],
                             [0.9, 0.0, 0.1], [0.88, 0.12, 0.0], [0.92, 0.08, 0.0])
        y_vecs = _make_vecs([-0.95, -0.05, 0.0], [-0.85, -0.1, 0.05],
                             [-0.9, 0.0, 0.1], [-0.88, -0.12, 0.0], [-0.92, -0.08, 0.0])
        return x_vecs, y_vecs, a_vecs, b_vecs

    def test_significant_for_biased_embedding(self):
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        p = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations=5000, random_state=0)
        assert p < 0.05

    def test_not_significant_for_zero_bias(self):
        a_vecs = _make_vecs([1.0, 0.0])
        b_vecs = _make_vecs([-1.0, 0.0])
        # symmetric word vectors
        x_vecs = _make_vecs([0.5, 0.5], [-0.5, 0.5], [0.0, 1.0], [0.3, 0.7], [-0.3, 0.7])
        y_vecs = x_vecs.copy()
        p = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations=1000, random_state=0)
        # p should be >> 0.05 for symmetric case; at minimum not clearly significant
        assert p > 0.2

    def test_p_value_in_range(self):
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        p = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations=1000, random_state=99)
        assert 0.0 <= p <= 1.0

    def test_reproducible_with_same_seed(self):
        x_vecs, y_vecs, a_vecs, b_vecs = self._biased_setup()
        p1 = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations=500, random_state=7)
        p2 = permutation_test(x_vecs, y_vecs, a_vecs, b_vecs, n_permutations=500, random_state=7)
        assert p1 == p2

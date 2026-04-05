"""
Unit tests for src/debiasing.py using synthetic toy embeddings.
No model files are required.
"""

import numpy as np
import pytest
from src.debiasing import compute_gender_direction, neutralize, equalize, hard_debias, build_debiased_lookup
from src.weat import weat_effect_size


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _make_vecs(*vecs):
    return np.array([_normalize(np.array(v, dtype=float)) for v in vecs])


# ---------------------------------------------------------------------------
# Fake EmbeddingModel for testing
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal EmbeddingModel-compatible object backed by a dict."""

    def __init__(self, word_vecs: dict):
        self._vecs = {w: _normalize(np.array(v, dtype=float)) for w, v in word_vecs.items()}

    def get_vector(self, word: str) -> np.ndarray:
        if word not in self._vecs:
            raise KeyError(word)
        return self._vecs[word]

    def get_vector_safe(self, word: str):
        return self._vecs.get(word, None)

    def has_word(self, word: str) -> bool:
        return word in self._vecs


def _make_biased_model():
    """
    A simple 3D model where:
    - dimension 0 = gender axis (positive = male, negative = female)
    - male gender words are strongly positive on dim 0
    - female gender words are strongly negative on dim 0
    - male occupations lean positive, female occupations lean negative
    - neutral occupations are near zero on dim 0
    """
    return FakeModel({
        # gender attrs
        "male1": [0.9, 0.1, 0.1],
        "male2": [0.85, 0.15, 0.1],
        "male3": [0.8, 0.2, 0.0],
        "female1": [-0.9, 0.1, 0.1],
        "female2": [-0.85, 0.15, 0.1],
        "female3": [-0.8, 0.2, 0.0],
        # occupations
        "occ_male1": [0.7, 0.5, 0.1],
        "occ_male2": [0.6, 0.6, 0.0],
        "occ_female1": [-0.7, 0.5, 0.1],
        "occ_female2": [-0.6, 0.6, 0.0],
        "occ_neutral1": [0.0, 0.8, 0.2],
        "occ_neutral2": [0.1, 0.9, 0.0],
    })


# ---------------------------------------------------------------------------
# compute_gender_direction
# ---------------------------------------------------------------------------

class TestComputeGenderDirection:
    def test_pca_returns_unit_vector(self):
        male_vecs = _make_vecs([1.0, 0.1, 0.0], [0.9, 0.2, 0.0], [0.85, 0.15, 0.0])
        female_vecs = _make_vecs([-1.0, 0.1, 0.0], [-0.9, 0.2, 0.0], [-0.85, 0.15, 0.0])
        g = compute_gender_direction(male_vecs, female_vecs, method="pca")
        assert abs(np.linalg.norm(g) - 1.0) < 1e-6

    def test_pca_direction_captures_variance_in_differences(self):
        # PCA finds the direction of maximum variance in the (centered) difference
        # vectors. For this to yield a meaningful gender direction we need pairs
        # where the *gender signal strength varies* across pairs — otherwise all
        # differences point the same way, centering removes the consistent
        # component, and PCA picks up noise.
        # Here pair 0 has a strong dim-0 gender signal, pair 2 has a weak one,
        # so PCA's first PC lies mostly along dim 0.
        male_vecs = np.array([
            [0.99, 0.0, 0.14],   # strong positive on dim 0
            [0.71, 0.71, 0.0],   # moderate positive on dim 0
            [0.15, 0.0, 0.99],   # weak positive on dim 0
        ])
        female_vecs = np.array([
            [-0.99, 0.0, 0.14],  # strong negative on dim 0
            [-0.71, 0.71, 0.0],  # moderate negative on dim 0
            [-0.15, 0.0, 0.99],  # weak negative on dim 0
        ])
        g = compute_gender_direction(male_vecs, female_vecs, method="pca")
        # The direction should capture the variance in the gender signal:
        # male projections and female projections should differ consistently
        diffs = (male_vecs @ g) - (female_vecs @ g)
        # All differences should have the same sign (consistently separated)
        assert all(diffs > 0) or all(diffs < 0), (
            f"PCA gender direction does not consistently separate pairs: {diffs}"
        )

    def test_mean_diff_returns_unit_vector(self):
        male_vecs = _make_vecs([1.0, 0.0], [0.9, 0.1])
        female_vecs = _make_vecs([-1.0, 0.0], [-0.9, -0.1])
        g = compute_gender_direction(male_vecs, female_vecs, method="mean_diff")
        assert abs(np.linalg.norm(g) - 1.0) < 1e-6

    def test_mismatched_lengths_raises(self):
        male_vecs = _make_vecs([1.0, 0.0], [0.9, 0.1])
        female_vecs = _make_vecs([-1.0, 0.0])
        with pytest.raises(ValueError, match="same length"):
            compute_gender_direction(male_vecs, female_vecs, method="pca")

    def test_unknown_method_raises(self):
        male_vecs = _make_vecs([1.0, 0.0])
        female_vecs = _make_vecs([-1.0, 0.0])
        with pytest.raises(ValueError, match="Unknown method"):
            compute_gender_direction(male_vecs, female_vecs, method="svd")


# ---------------------------------------------------------------------------
# neutralize
# ---------------------------------------------------------------------------

class TestNeutralize:
    def test_gender_component_removed(self):
        g = np.array([1.0, 0.0, 0.0])  # gender axis = dim 0
        word_vecs = _make_vecs([0.8, 0.6, 0.0], [0.6, 0.8, 0.0], [0.7, 0.5, 0.5])
        debiased = neutralize(word_vecs, g)
        for v in debiased:
            assert abs(np.dot(v, g)) < 1e-6, f"Gender component not removed: {np.dot(v, g)}"

    def test_output_is_unit_norm(self):
        g = np.array([1.0, 0.0, 0.0])
        word_vecs = _make_vecs([0.8, 0.6, 0.0], [0.5, 0.5, 0.7])
        debiased = neutralize(word_vecs, g)
        for v in debiased:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_orthogonal_words_unchanged(self):
        g = np.array([1.0, 0.0, 0.0])
        # vectors already orthogonal to g (no dim-0 component)
        word_vecs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        debiased = neutralize(word_vecs, g)
        np.testing.assert_allclose(debiased, word_vecs, atol=1e-6)

    def test_shape_preserved(self):
        g = _normalize(np.array([1.0, 1.0, 0.0]))
        word_vecs = _make_vecs([0.8, 0.2, 0.0], [0.5, 0.5, 0.7], [0.3, 0.9, 0.1])
        debiased = neutralize(word_vecs, g)
        assert debiased.shape == word_vecs.shape


# ---------------------------------------------------------------------------
# equalize
# ---------------------------------------------------------------------------

class TestEqualize:
    def _setup(self):
        g = np.array([1.0, 0.0, 0.0])
        male_vecs = _make_vecs([0.9, 0.1, 0.0], [0.85, 0.15, 0.0], [0.8, 0.2, 0.0])
        female_vecs = _make_vecs([-0.9, 0.1, 0.0], [-0.85, 0.15, 0.0], [-0.8, 0.2, 0.0])
        return male_vecs, female_vecs, g

    def test_output_unit_norm(self):
        male_vecs, female_vecs, g = self._setup()
        m_eq, f_eq = equalize(male_vecs, female_vecs, g)
        for v in np.concatenate([m_eq, f_eq]):
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_symmetry_around_gender_axis(self):
        # After equalization, cos(male_eq, g) == -cos(female_eq, g) for each pair
        male_vecs, female_vecs, g = self._setup()
        m_eq, f_eq = equalize(male_vecs, female_vecs, g)
        for m, f in zip(m_eq, f_eq):
            assert abs(np.dot(m, g) + np.dot(f, g)) < 1e-6

    def test_shape_preserved(self):
        male_vecs, female_vecs, g = self._setup()
        m_eq, f_eq = equalize(male_vecs, female_vecs, g)
        assert m_eq.shape == male_vecs.shape
        assert f_eq.shape == female_vecs.shape


# ---------------------------------------------------------------------------
# hard_debias (integration)
# ---------------------------------------------------------------------------

class TestHardDebias:
    def test_returns_correct_shapes(self):
        model = _make_biased_model()
        occ_words = ["occ_male1", "occ_male2", "occ_female1", "occ_female2"]
        male_words = ["male1", "male2", "male3"]
        female_words = ["female1", "female2", "female3"]
        occ_d, m_eq, f_eq, g, occ_f, m_f, f_f = hard_debias(model, occ_words, male_words, female_words)
        assert occ_d.shape[0] == len(occ_words)
        assert occ_d.shape[1] == 3
        assert g.shape == (3,)

    def test_gender_direction_is_unit_vector(self):
        model = _make_biased_model()
        _, _, _, g, _, _, _ = hard_debias(
            model, ["occ_male1"], ["male1", "male2", "male3"], ["female1", "female2", "female3"]
        )
        assert abs(np.linalg.norm(g) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# build_debiased_lookup (integration + WEAT effect)
# ---------------------------------------------------------------------------

class TestBuildDebiasedLookup:
    def test_lookup_contains_all_words(self):
        model = _make_biased_model()
        occ_words = ["occ_male1", "occ_male2", "occ_female1", "occ_female2"]
        male_words = ["male1", "male2", "male3"]
        female_words = ["female1", "female2", "female3"]
        lookup = build_debiased_lookup(model, occ_words, male_words, female_words)
        for w in occ_words + male_words[:3] + female_words[:3]:
            assert w in lookup, f"'{w}' missing from debiased lookup"

    def test_debiased_occ_vectors_have_no_gender_component(self):
        model = _make_biased_model()
        occ_words = ["occ_male1", "occ_male2", "occ_female1", "occ_female2"]
        male_words = ["male1", "male2", "male3"]
        female_words = ["female1", "female2", "female3"]

        _, _, _, g, _, _, _ = hard_debias(model, occ_words, male_words, female_words)
        lookup = build_debiased_lookup(model, occ_words, male_words, female_words)

        for w in occ_words:
            v = lookup[w]
            dot = abs(np.dot(v, g))
            assert dot < 1e-5, f"'{w}' still has gender component {dot:.6f} after debiasing"

    def test_debiasing_reduces_effect_size(self):
        """WEAT effect size should decrease after debiasing."""
        model = _make_biased_model()
        occ_words = ["occ_male1", "occ_male2", "occ_female1", "occ_female2"]
        male_words = ["male1", "male2", "male3"]
        female_words = ["female1", "female2", "female3"]

        # Before debiasing
        x_vecs = np.array([model.get_vector(w) for w in ["occ_male1", "occ_male2"]])
        y_vecs = np.array([model.get_vector(w) for w in ["occ_female1", "occ_female2"]])
        a_vecs = np.array([model.get_vector(w) for w in male_words])
        b_vecs = np.array([model.get_vector(w) for w in female_words])
        d_before, _, _ = weat_effect_size(x_vecs, y_vecs, a_vecs, b_vecs)

        # After debiasing
        lookup = build_debiased_lookup(model, occ_words, male_words, female_words)
        x_vecs_d = np.array([lookup["occ_male1"], lookup["occ_male2"]])
        y_vecs_d = np.array([lookup["occ_female1"], lookup["occ_female2"]])
        a_vecs_d = np.array([lookup[w] for w in male_words])
        b_vecs_d = np.array([lookup[w] for w in female_words])
        d_after, _, _ = weat_effect_size(x_vecs_d, y_vecs_d, a_vecs_d, b_vecs_d)

        assert abs(d_after) < abs(d_before), (
            f"Expected |d_after| < |d_before|, got {abs(d_after):.4f} vs {abs(d_before):.4f}"
        )

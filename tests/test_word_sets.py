"""
Unit tests for src/word_sets.py — no model files required.
"""

from src.word_sets import (
    get_korean_word_sets,
    get_all_occupation_words,
    get_all_words,
    occupation_category,
    WORD_ALIASES,
    WEAT_TESTS,
)


class TestWordSetsValidity:
    def setup_method(self):
        self.ws = get_korean_word_sets()

    def test_all_lists_nonempty(self):
        assert len(self.ws.male_attrs) >= 3
        assert len(self.ws.female_attrs) >= 3
        assert len(self.ws.male_occupations) >= 3
        assert len(self.ws.female_occupations) >= 3
        assert len(self.ws.neutral_occupations) >= 3

    def test_no_word_in_both_gender_attr_sets(self):
        overlap = set(self.ws.male_attrs) & set(self.ws.female_attrs)
        assert not overlap, f"Words in both male and female attrs: {overlap}"

    def test_no_occupation_in_multiple_categories(self):
        male_occ = set(self.ws.male_occupations)
        female_occ = set(self.ws.female_occupations)
        neutral_occ = set(self.ws.neutral_occupations)
        assert not (male_occ & female_occ), f"Overlap male/female occ: {male_occ & female_occ}"
        assert not (male_occ & neutral_occ), f"Overlap male/neutral occ: {male_occ & neutral_occ}"
        assert not (female_occ & neutral_occ), f"Overlap female/neutral occ: {female_occ & neutral_occ}"

    def test_no_attr_word_in_occupations(self):
        all_attrs = set(self.ws.male_attrs) | set(self.ws.female_attrs)
        all_occ = set(get_all_occupation_words(self.ws))
        overlap = all_attrs & all_occ
        assert not overlap, f"Word in both attrs and occupations: {overlap}"

    def test_get_all_occupation_words_deduplicated(self):
        occ = get_all_occupation_words(self.ws)
        assert len(occ) == len(set(occ)), "Duplicate occupation words"

    def test_get_all_words_deduplicated(self):
        all_words = get_all_words(self.ws)
        assert len(all_words) == len(set(all_words)), "Duplicate words in get_all_words()"

    def test_get_all_words_contains_all_sets(self):
        all_words = set(get_all_words(self.ws))
        for w in self.ws.male_attrs + self.ws.female_attrs + get_all_occupation_words(self.ws):
            assert w in all_words, f"'{w}' missing from get_all_words()"

    def test_known_canonical_words_present(self):
        # Spot-check a few expected words from the research proposal
        assert "군인" in self.ws.male_occupations
        assert "간호사" in self.ws.female_occupations
        assert "의사" in self.ws.neutral_occupations
        assert "아버지" in self.ws.male_attrs
        assert "어머니" in self.ws.female_attrs


class TestOccupationCategory:
    def setup_method(self):
        self.ws = get_korean_word_sets()

    def test_male_occupation(self):
        assert occupation_category("군인", self.ws) == "male_occupations"

    def test_female_occupation(self):
        assert occupation_category("간호사", self.ws) == "female_occupations"

    def test_neutral_occupation(self):
        assert occupation_category("의사", self.ws) == "neutral_occupations"

    def test_unknown_word(self):
        assert occupation_category("unknown_word_xyz", self.ws) == "unknown"


class TestWordAliases:
    def test_ceo_has_alias(self):
        assert "CEO" in WORD_ALIASES
        assert len(WORD_ALIASES["CEO"]) >= 1

    def test_aliases_are_strings(self):
        for word, alts in WORD_ALIASES.items():
            assert isinstance(word, str)
            for alt in alts:
                assert isinstance(alt, str)


class TestWEATTests:
    def setup_method(self):
        self.ws = get_korean_word_sets()
        self.valid_keys = {
            "male_occupations", "female_occupations", "neutral_occupations"
        }

    def test_weat_tests_nonempty(self):
        assert len(WEAT_TESTS) >= 1

    def test_weat_test_structure(self):
        for x_key, y_key, name in WEAT_TESTS:
            assert x_key in self.valid_keys, f"Invalid x_key: {x_key}"
            assert y_key in self.valid_keys, f"Invalid y_key: {y_key}"
            assert isinstance(name, str) and len(name) > 0

    def test_weat_tests_have_distinct_keys(self):
        for x_key, y_key, _ in WEAT_TESTS:
            assert x_key != y_key, "x_key and y_key should differ"

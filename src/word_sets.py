"""
Korean WEAT word sets for gender-occupation bias analysis.

Word list design notes:
- Gender attribute words use high-frequency Korean gender markers (kinship terms,
  pronouns, gendered nouns). English uses he/she; Korean expresses gender via
  family role terms and the pronouns 그(he)/그녀(she).
- Occupation words are selected based on 통계청(Statistics Korea) 직업별 성별
  종사자 비율 (2024 1/2 survey). Inclusion threshold: ≥80% one gender.
  Source: 경제활동인구조사 직업별 취업자 (2024.1/2).

Gender ratio reference (통계청 2024):
  Male-coded (≥80% male):
    군인       ~97% M (국방부 별도 통계)
    소방관     90.9% M (소방·방재 기술자 150/165)
    기관사     ~100% M (철도·전동차 기관사 11/12)
    엔지니어   87.2% M (공학 전문가 967/1109)
    CEO        95.0% M (기업 고위 임원 19/20)
    경찰관     84.6% M (경찰·소방·교도 종사자 181/214)
    운전기사   98.4% M (자동차 운전원 974/990)
  Female-coded (≥80% female):
    간호사     92.6% F (299/323)
    보육교사   93.2% F (돌봄 서비스 672/721)
    유치원교사 98.4% F (62/63)
    영양사     97.7% F (42/43)
    미용사     85.2% F (미용 서비스 283/332)
    사회복지사 87.0% F (사회복지 종사자 400/460)
    가사도우미 98.3% F (가사·육아 도우미 113/115)
  Excluded (threshold not met):
    판사/검사  77.5% M — removed (법률 전문가 31/40)
    비서       75.9% F — removed (비서·사무보조 205/270)
    승무원     60.0% F — removed (운송 서비스 15/25; 통계 이종 집계)

Polysemy caveats (documented as limitations):
- 그: primarily "he" as a pronoun, but also means "that" (demonstrative) with
  very high corpus frequency. Its static vector may be pulled toward the
  demonstrative sense. If cosine similarity to other male attrs is anomalously
  low, consider removing it.
- 의사: primarily 醫師 (doctor) in this context, but 意思 (intent/will) and
  義士 (martyr) are also common in text corpora. Static embeddings average all
  senses into one vector. Document as a static embedding limitation.
- CEO: likely OOV in Word2Vec models; alias 최고경영자 is tried automatically.
- 가사도우미: compound noun; likely OOV in older Word2Vec; alias 도우미 tried.
- 운전기사: alias 운전원 tried for Word2Vec OOV.
- 사회복지사: compound noun; alias 복지사 tried for Word2Vec OOV.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class WEATWordSets:
    male_attrs: List[str]
    female_attrs: List[str]
    male_occupations: List[str]
    female_occupations: List[str]
    neutral_occupations: List[str]


def get_korean_word_sets() -> WEATWordSets:
    """Return the canonical Korean gender-occupation WEAT word sets."""
    return WEATWordSets(
        male_attrs=["아버지", "아들", "남자", "남성", "형", "오빠", "남편", "그"],
        female_attrs=["어머니", "딸", "여자", "여성", "언니", "누나", "아내", "그녀"],
        male_occupations=["군인", "소방관", "기관사", "엔지니어", "CEO", "경찰관", "운전기사"],
        female_occupations=["간호사", "보육교사", "유치원교사", "영양사", "미용사", "사회복지사", "가사도우미"],
        neutral_occupations=["의사", "교수", "변호사", "연구원", "작가"],
    )


# OOV aliases: if the primary form is missing from a model's vocabulary,
# these alternatives are tried in order.
WORD_ALIASES: Dict[str, List[str]] = {
    # CEO: Latin-script token often OOV in older Word2Vec models
    "CEO": ["최고경영자", "대표이사", "회장"],
    # Compound nouns missing from older/smaller Word2Vec vocabularies
    "유치원교사": ["유치원 교사", "선생님"],
    "보육교사": ["보육 교사", "교사"],
    "사회복지사": ["복지사", "사회복지"],
    "가사도우미": ["도우미", "가정부"],
    "운전기사": ["운전원", "기사"],
}


def get_all_occupation_words(word_sets: WEATWordSets) -> List[str]:
    """Flatten all three occupation lists into one deduplicated list."""
    seen = set()
    result = []
    for word in (
        word_sets.male_occupations
        + word_sets.female_occupations
        + word_sets.neutral_occupations
    ):
        if word not in seen:
            seen.add(word)
            result.append(word)
    return result


def get_all_words(word_sets: WEATWordSets) -> List[str]:
    """All words across all sets (attrs + occupations), deduplicated."""
    return list(
        dict.fromkeys(
            word_sets.male_attrs
            + word_sets.female_attrs
            + get_all_occupation_words(word_sets)
        )
    )


def occupation_category(word: str, word_sets: WEATWordSets) -> str:
    """Return the occupation category label for a word, or 'unknown'."""
    if word in word_sets.male_occupations:
        return "male_occupations"
    if word in word_sets.female_occupations:
        return "female_occupations"
    if word in word_sets.neutral_occupations:
        return "neutral_occupations"
    return "unknown"


# Human-readable labels for use in plots
CATEGORY_LABELS: Dict[str, str] = {
    "male_occupations": "전통 남성 직종",
    "female_occupations": "전통 여성 직종",
    "neutral_occupations": "중립/전문직",
}

# WEAT test definitions: (target_x_key, target_y_key, name)
# Each test compares two occupation groups against male vs female attrs.
WEAT_TESTS: List[Tuple[str, str, str]] = [
    ("male_occupations", "female_occupations", "남성직종 vs 여성직종"),
    ("neutral_occupations", "male_occupations", "중립직종 vs 남성직종"),
    ("neutral_occupations", "female_occupations", "중립직종 vs 여성직종"),
]

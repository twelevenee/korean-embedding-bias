# Korean Word Embedding Gender Bias — WEAT Analysis

> **한국어 단어 임베딩의 젠더 편향 측정 및 완화**  
> Measuring and mitigating gender bias in Korean pre-trained word embeddings via WEAT

---

## Overview

This project quantifies gender-occupational bias in Korean pre-trained word embeddings (FastText, Word2Vec) using the **Word Embedding Association Test (WEAT)** framework (Caliskan et al., 2017), then applies **hard debiasing** (Bolukbasi et al., 2016) and measures its effect.

Most prior WEAT research targets English embeddings. Korean's agglutinative morphology and distinct socio-cultural gender norms require independent analysis. This project provides:

- A Korean-specific gender–occupation stimulus word set grounded in **통계청 성별 종사자 비율** (≥80% threshold)
- A full WEAT pipeline with permutation testing (10,000 iterations)
- Before/after hard debiasing comparison with effect size and significance analysis
- **Partial debiasing (α sweep)**: quantifies the tradeoff between bias reduction and semantic distortion; identifies optimal debiasing strength α* = 0.95
- Word2Vec cross-validation for robustness check
- Per-word association score visualization revealing which occupations are most gender-coded
- Visualizations: PCA scatter, cosine heatmap, effect size bar charts, debiasing comparison, α tradeoff curve

---

## Key Results

### FastText (cc.ko.300) — Primary Analysis

| Test | Effect Size (d) | p-value | Significant |
|---|---|---|---|
| 전통 남성직종 vs 여성직종 | **+1.42** | 0.002 | ✅ |
| 중립/전문직 vs 여성직종 | **+1.29** | 0.007 | ✅ |
| 중립/전문직 vs 남성직종 | −0.59 | 0.811 | ❌ |

> d > 0 = target set X more associated with male attributes. |d| > 0.8 = large effect (Cohen's convention).

**After hard debiasing (α = 1.0):**

| Test | Before | After | Δ |
|---|---|---|---|
| 전통 남성직종 vs 여성직종 | +1.42 ✅ | +0.73 ❌ | −0.69 |
| 중립/전문직 vs 남성직종 | −0.59 ❌ | −1.03 ❌ | −0.44 |
| 중립/전문직 vs 여성직종 | +1.29 ✅ | **−0.81** ❌ | **−2.10** |

Full hard debiasing removes statistical significance but the neutral-vs-female test **reverses sign** (overcorrection).

### Partial Debiasing — α Sweep

To address the overcorrection problem, the debiasing strength α ∈ [0, 1] was swept across both the neutralize and equalize steps:

| α | 남vs여 | 중vs여 | 비고 |
|---|---|---|---|
| 0.00 | +1.42 ✅ | +1.29 ✅ | baseline |
| 0.50 | +0.91 ❌ | +0.78 ❌ | 유의성 소실 시작 |
| **0.95** | **+0.34 ❌** | **+0.19 ❌** | **최적점: 유의성 소실 + 부호 유지** |
| 1.00 | +0.73 ❌ | −0.81 ❌ | 부호 역전 (overcorrection) |

**α* = 0.95** achieves full removal of statistical significance across all three tests while preserving the sign of all effect sizes — no semantic distortion. The effective safe range is α ∈ [0.50, 0.95].

### Word2Vec (Kyubyong/wordvectors) — Cross-Validation

| Test | Effect Size (d) | p-value | Significant |
|---|---|---|---|
| 중립/전문직 vs 남성직종 | −0.20 | 0.606 | ❌ |

Word2Vec cross-validation is limited to the 중립직종 vs 남성직종 test: the revised female occupation set (영양사, 미용사, 사회복지사) falls outside the 2016 Wikipedia/Namuwiki vocabulary, leaving fewer than 5 resolvable words for tests involving female occupations. This vocabulary gap reflects the encyclopedic corpus domain rather than word novelty, and is itself evidence that **corpus selection shapes what semantic associations a model can encode**.

### Key Findings

1. **Gender-occupational bias confirmed in FastText**: Statistics-grounded male-coded occupations (군인, 소방관, 경찰관, 운전기사…) are strongly associated with male attributes (d = 1.42, p = 0.002).
2. **Male default for expertise**: Neutral/professional occupations (의사, 교수, 변호사…) are more male-associated than female-coded occupations (d = 1.29, p = 0.007) — suggesting a structural male default for expertise in Korean web-crawled text.
3. **Hard debiasing overcorrects at α = 1.0**: Statistical significance is removed but the neutral-vs-female test reverses sign (Δ = −2.10), indicating semantic distortion.
4. **Optimal partial debiasing at α = 0.95**: Sweeping the debiasing strength identifies α* = 0.95 as the point where all biases become statistically non-significant without sign reversal — bias removed, semantic structure preserved.
5. **Corpus matters more than model architecture**: The FastText/Word2Vec divergence suggests training data social context is the primary driver of bias strength over model architecture.

---

## Figures

### WEAT Effect Size (FastText — before debiasing)
![Effect Size Before](results/figures/effect_size_bar_before.png)

### FastText vs Word2Vec Cross-Model Comparison
![Cross-Model](results/figures/effect_size_bar_crossmodel.png)

### FastText Before/After Debiasing + Word2Vec
![All Results](results/figures/effect_size_bar_all.png)

### Hard Debiasing Before vs. After
![Debiasing Comparison](results/figures/debiasing_comparison_fasttext.png)

### Partial Debiasing α Tradeoff Curve
![Alpha Tradeoff](results/figures/alpha_tradeoff.png)

### Per-Word Association Scores — FastText
![Per-Word FastText](results/figures/per_word_scores_fasttext.png)

### Per-Word Association Scores — Word2Vec
![Per-Word Word2Vec](results/figures/per_word_scores_word2vec.png)

### PCA: Gender + Occupation Word Distribution
![PCA Scatter](results/figures/pca_scatter_fasttext.png)

### Cosine Similarity Heatmap (Occupations × Gender Attributes)
![Cosine Heatmap](results/figures/cosine_heatmap_fasttext.png)

---

## Method

### Word Sets

**Gender attributes:**
- Male: 아버지, 아들, 남자, 남성, 형, 오빠, 남편, 그
- Female: 어머니, 딸, 여자, 여성, 언니, 누나, 아내, 그녀

**Occupation targets — selected by 통계청 성별 종사자 비율 ≥ 80% threshold (경제활동인구조사 2024 1/2):**

| Category | Words (7) | Gender % range |
|---|---|---|
| Male-coded | 군인, 소방관, 기관사, 엔지니어, CEO, 경찰관, 운전기사 | 85–100% male |
| Female-coded | 간호사, 보육교사, 유치원교사, 영양사, 미용사, 사회복지사, 가사도우미 | 85–98% female |
| Neutral/professional | 의사, 교수, 변호사, 연구원, 작가 | 53–77% male |

Excluded words that failed the 80% threshold: 판사/검사 (77.5% M), 비서 (75.9% F), 승무원 (60.0% F).
Neutral occupations retain a slight male skew (의사 74% M, 변호사 77.5% M), consistent with the "male default for expertise" hypothesis under test.

### WEAT Effect Size

```
s(w, A, B) = (1/|A|) Σ cos(w,a) - (1/|B|) Σ cos(w,b)
d = (mean_{x∈X} s(x,A,B) - mean_{y∈Y} s(y,A,B)) / std_{z∈X∪Y} s(z,A,B)
```

Means are computed independently for A and B to handle unequal set sizes after OOV filtering.  
Statistical significance via one-sided permutation test (10,000 iterations).

### Hard Debiasing (Bolukbasi et al., 2016) with Partial Debiasing Extension

1. Compute gender direction **g** via PCA on male–female difference vectors (L2-normalized)
2. **Neutralize**: partially remove gender component from occupation word vectors: v′ = v − α(v·g)g
3. **Equalize**: interpolate gender word pairs α-fraction toward fully equalized position

α = 1.0 recovers the original Bolukbasi hard debiasing. α ∈ (0, 1) implements partial debiasing. The optimal α* = 0.95 is identified by finding the largest α at which all WEAT effect sizes remain positive (no sign reversal).

---

## Embedding Models

| Model | Source | Dim | Training Corpus | Notes |
|---|---|---|---|---|
| Korean FastText | [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html) | 300 | CC-100 web crawl | Subword OOV handling; primary analysis |
| Korean Word2Vec | [Kyubyong/wordvectors](https://github.com/Kyubyong/wordvectors) | 200 | Wikipedia + Namuwiki | Cross-validation; OOV words resolved via aliases |

Download FastText (≈4.2 GB):
```bash
bash setup.sh
```

---

## Limitations

- **Static embeddings**: Polysemous words (`그` = he/that; `의사` = doctor/intent/martyr) have a single averaged vector — contextual sense disambiguation is lost.
- **Word2Vec cross-validation scope**: The 2016 Wikipedia/Namuwiki model lacks 영양사, 미용사, 사회복지사 in its vocabulary, limiting cross-validation to tests not involving the female occupation set. The vocabulary gap is itself informative about corpus domain effects.
- **Stimulus word selection**: Occupation lists are grounded in 통계청 2024 employment statistics (≥80% gender threshold), but survey-based validation of psychological salience was not performed.
- **Hard debiasing overcorrection**: Sign reversal at α = 1.0 (Δ = −2.10) is mitigated by partial debiasing (α* = 0.95), but the residual effect sizes at α* are small rather than zero — complete debiasing without distortion remains an open problem. INLP (Ravfogel et al., 2020) may offer further improvement.
- **Scope**: Static (non-contextual) embeddings only. Contextual bias in KoBERT, EXAONE, etc. is out of scope.

---

## Repo Structure

```
├── src/
│   ├── word_sets.py        # Korean WEAT stimulus word sets + OOV aliases
│   ├── weat.py             # WEAT calculation + permutation test
│   ├── load_embeddings.py  # Model loading (FastText + Word2Vec)
│   ├── debiasing.py        # Hard debiasing with α scaling (neutralize + equalize)
│   └── visualize.py        # All visualizations + CSV export
├── notebooks/
│   └── analysis.ipynb      # Full narrative analysis (executed)
├── tests/                  # 47 unit tests — no model files required
├── results/
│   ├── figures/            # 9 PNG outputs
│   └── csv/                # WEAT result tables (FastText before/after, Word2Vec)
├── run_analysis.py         # Headless script entrypoint
├── setup.sh                # Dependency install + model download
└── conftest.py             # pytest path setup
```

## Quickstart

```bash
# 1. Install dependencies + download FastText model (~4.2 GB)
bash setup.sh

# 2. (Optional) Place Korean Word2Vec at models/ko.bin
#    https://github.com/Kyubyong/wordvectors

# 3. Run notebook
jupyter notebook notebooks/analysis.ipynb

# 4. Or headless
python run_analysis.py --fasttext-path models/cc.ko.300.bin
python run_analysis.py --fasttext-path models/cc.ko.300.bin --word2vec-path models/ko.bin
```

```bash
# Unit tests (no model required)
pytest tests/ -v   # 47 tests
```

---

## References

- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS*.
- Ravfogel, S. et al. (2020). Null it out: Guarding protected attributes by iterative nullspace projection. *ACL*.
- Ko, W. et al. (2023). KoBBQ: Korean Bias Benchmark for Question Answering. *ACL Findings*.
- Bender, E. M. et al. (2021). On the dangers of stochastic parrots. *FAccT*.

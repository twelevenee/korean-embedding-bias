# Korean Word Embedding Gender Bias — WEAT Analysis

> **한국어 단어 임베딩의 젠더 편향 측정 및 완화**  
> Measuring and mitigating gender bias in Korean pre-trained word embeddings via WEAT

---

## Overview

This project quantifies gender-occupational bias in Korean pre-trained word embeddings (FastText) using the **Word Embedding Association Test (WEAT)** framework (Caliskan et al., 2017), then applies **hard debiasing** (Bolukbasi et al., 2016) and measures its effect.

Most prior WEAT research targets English embeddings. Korean's agglutinative morphology and distinct socio-cultural gender norms require independent analysis. This project provides:

- A Korean-specific gender–occupation stimulus word set grounded in Korean social context
- A full WEAT pipeline with permutation testing
- Before/after hard debiasing comparison with effect size analysis
- Visualizations: PCA scatter, cosine heatmap, effect size bar chart

---

## Key Results

| Test | Effect Size (d) | p-value | Significant |
|---|---|---|---|
| 전통 남성직종 vs 여성직종 | **+1.46** | 0.003 | ✅ |
| 중립/전문직 vs 여성직종 | **+1.18** | 0.022 | ✅ |
| 중립/전문직 vs 남성직종 | -0.85 | 0.925 | ❌ |

> d > 0 indicates association with male attributes. |d| > 0.8 = large effect (Cohen's convention).

**Findings:**
- Traditional male-coded occupations (군인, 소방관, 판사, 엔지니어…) are strongly associated with male attributes (d = 1.46, p = 0.003).
- Neutral/professional occupations (의사, 교수, 변호사…) are also more male-associated than female-coded occupations (d = 1.18, p = 0.022) — suggesting a **male default for expertise** in the Korean embedding space.
- Hard debiasing removes statistical significance across all three tests, but the "neutral vs. female occupations" test shows a **sign reversal** (d: +1.18 → -1.01, Δ = -2.18), demonstrating the **overcorrection problem** inherent to hard debiasing: the algorithm distorts the semantic structure of occupation words beyond just removing gender association.

### Figures

| | |
|---|---|
| ![Effect Size Bar Chart](results/figures/effect_size_bar_before.png) | ![Debiasing Comparison](results/figures/debiasing_comparison_fasttext.png) |
| WEAT effect sizes (before debiasing) | Before vs. after hard debiasing |
| ![PCA Scatter](results/figures/pca_scatter_fasttext.png) | ![Cosine Heatmap](results/figures/cosine_heatmap_fasttext.png) |
| PCA: gender + occupation word distribution | Cosine similarity heatmap |

---

## Method

### Word Sets

**Gender attributes:**
- Male: 아버지, 아들, 남자, 남성, 형, 오빠, 남편, 그
- Female: 어머니, 딸, 여자, 여성, 언니, 누나, 아내, 그녀

**Occupation targets (Korean-context design):**
- Traditional male: 군인, 소방관, 기관사, 판사, 검사, 엔지니어, CEO
- Traditional female: 간호사, 보육교사, 유치원교사, 비서, 승무원
- Neutral/professional: 의사, 교수, 변호사, 연구원, 작가

### WEAT Effect Size

```
s(w, A, B) = (1/|A|) Σ cos(w,a) - (1/|B|) Σ cos(w,b)
d = (mean_{x∈X} s(x,A,B) - mean_{y∈Y} s(y,A,B)) / std_{z∈X∪Y} s(z,A,B)
```

Statistical significance via one-sided permutation test (10,000 iterations).

### Hard Debiasing (Bolukbasi et al., 2016)

1. Compute gender direction **g** via PCA on male–female difference vectors
2. **Neutralize**: remove gender component from occupation word vectors
3. **Equalize**: make gender word pairs equidistant from the gender axis midpoint

---

## Embedding Model

| Model | Source | Notes |
|---|---|---|
| Korean FastText | [fasttext.cc](https://fasttext.cc/docs/en/crawl-vectors.html) | cc.ko.300.bin, 300-dim, subword OOV handling |

Download (≈4.2 GB):
```bash
bash setup.sh
```

---

## Limitations

- **Single model**: Word2Vec cross-validation was not completed (model unavailable). Results reflect FastText only.
- **Static embeddings**: Polysemous words (e.g., `그` = he/that; `의사` = doctor/intent/martyr) have a single averaged vector — contextual nuance is lost.
- **Stimulus word selection bias**: The word lists reflect researcher judgment; survey-based validation was not performed.
- **Overcorrection**: Hard debiasing reverses the sign of the neutral-vs-female-occupation effect (d: +1.18 → -1.01), suggesting semantic distortion beyond the intended correction. Softer alternatives (INLP, Ravfogel et al., 2020) may be preferable.
- **Scope**: Static (non-contextual) embeddings only. Contextual bias in KoBERT, EXAONE, etc. is out of scope.

---

## Repo Structure

```
├── src/
│   ├── word_sets.py        # Korean WEAT stimulus word sets
│   ├── weat.py             # WEAT calculation + permutation test
│   ├── load_embeddings.py  # Model loading utilities
│   ├── debiasing.py        # Hard debiasing (neutralize + equalize)
│   └── visualize.py        # All visualizations + CSV export
├── notebooks/
│   └── analysis.ipynb      # Full narrative analysis
├── tests/                  # 47 unit tests (no model files required)
├── results/
│   ├── figures/            # PNG outputs
│   └── csv/                # WEAT result tables
├── run_analysis.py         # Headless script entrypoint
└── setup.sh                # Dependency install + model download
```

## Quickstart

```bash
# 1. Install dependencies + download FastText model (~4.2 GB)
bash setup.sh

# 2. Run notebook
jupyter notebook notebooks/analysis.ipynb

# 3. Or headless
python run_analysis.py --fasttext-path models/cc.ko.300.bin
```

```bash
# Unit tests (no model required)
pytest tests/ -v
```

---

## References

- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.
- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *NeurIPS*.
- Ko, W. et al. (2023). KoBBQ: Korean Bias Benchmark for Question Answering. *ACL Findings*.
- Bender, E. M. et al. (2021). On the dangers of stochastic parrots. *FAccT*.

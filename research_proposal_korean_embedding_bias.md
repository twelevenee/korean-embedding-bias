# 한국어 단어 임베딩의 젠더 편향 측정 및 분석

> **Korean Word Embedding Gender Bias Measurement via WEAT**  
> 머신러닝과 데이터마이닝 과목 프로젝트 | 2025

---

## 개요

본 프로젝트는 한국어 사전학습 단어 임베딩(Word2Vec / FastText)에 내재된 젠더 편향을 WEAT(Word Embedding Association Test) 방법론을 통해 정량적으로 측정하고 시각화한다. 기존 WEAT 연구의 대부분이 영어 임베딩을 대상으로 수행된 반면, 한국어는 교착어적 특성과 사회문화적 맥락이 상이하여 독립적인 분석이 필요하다. 직업 단어와 젠더 단어 간의 연관 강도를 코사인 유사도 기반으로 측정하고, 편향 완화(debiasing) 기법 적용 전후를 비교함으로써 한국어 임베딩 편향의 구조적 특성을 탐구한다.

---

## 연구 배경 및 동기

단어 임베딩은 대규모 텍스트 코퍼스로부터 학습된 단어의 분산 표현으로, 자연어처리 파이프라인 전반에서 광범위하게 활용된다. Bolukbasi et al.(2016)은 Google News 코퍼스로 학습된 Word2Vec 임베딩이 "man is to computer programmer as woman is to homemaker"와 같은 성별 고정관념을 재현한다는 사실을 보였고, Caliskan et al.(2017)은 WEAT를 통해 임베딩 내 편향을 심리학적 IAT(Implicit Association Test)에 대응하는 방식으로 측정하는 프레임워크를 제안했다.

그러나 한국어 임베딩에 대한 동등한 수준의 분석은 아직 충분히 이루어지지 않았다. 한국 사회의 젠더 규범은 직업, 역할, 감정 표현 등 다양한 영역에서 특수한 문화적 맥락을 가지며, 이는 영어권 연구 결과와 상이한 편향 패턴을 생성할 가능성이 있다. KoBBQ(Ko et al., 2023)와 같은 최근 연구가 LLM 수준의 한국어 사회적 편향을 다루기 시작했으나, 임베딩 레벨에서의 체계적 측정은 여전히 공백으로 남아 있다.

본 프로젝트는 이 공백을 직접 겨냥한다.

---

## 연구 목표

1. 한국어 맥락에 적합한 WEAT 자극 단어 목록(직업어, 젠더 속성어)을 설계한다.
2. 사전학습 한국어 임베딩에서 WEAT effect size를 계산하여 젠더-직업 편향을 정량화한다.
3. 직업 카테고리별 편향 패턴을 시각화하고, 영어 WEAT 선행 연구와 비교한다.
4. Hard debiasing(Bolukbasi et al., 2016) 적용 전후 effect size 변화를 측정한다.
5. 결과를 한국 사회의 젠더 권력 구조와 연결하여 비판적으로 해석한다.

---

## 방법론

### 1. 임베딩 모델

직접 학습 대신 공개된 사전학습 한국어 임베딩을 사용하여 구현 부담을 최소화하고 분석의 재현 가능성을 높인다.

| 모델 | 출처 | 특징 |
|---|---|---|
| Korean-fasttext | Facebook Research | 형태소 단위 서브워드 임베딩, OOV 강건성 |
| Word2Vec (나무위키) | 공개 사전학습 모델 | 문서 맥락 반영 |

두 모델을 모두 사용하여 모델 간 편향 패턴 일관성을 교차 검증한다.

### 2. 자극 단어 설계 (Stimulus Word Sets)

WEAT는 네 가지 단어 집합을 필요로 한다: 목표 개념 A, B (예: 남성 단어 / 여성 단어)와 속성 집합 X, Y (예: 직업 그룹 1 / 직업 그룹 2).

**젠더 속성어 (Gender attribute words)**

한국어 고빈도 젠더 지시어를 사용한다. 영어의 he/she 대신 한국어는 주로 직업 접사, 대명사, 친족어로 젠더를 표현한다.

- 남성 연상 단어: 아버지, 아들, 남자, 남성, 형, 오빠, 남편, 그
- 여성 연상 단어: 어머니, 딸, 여자, 여성, 언니, 누나, 아내, 그녀

**직업 목표어 (Occupation target words) — 한국 맥락 설계**

단순히 영어 목록을 번역하는 것이 아니라, 한국 사회에서 젠더 고정관념이 명확하게 형성된 직업군을 우선 선정한다.

| 카테고리 | 단어 예시 | 예상 편향 방향 |
|---|---|---|
| 전통적 남성 직종 | 군인, 소방관, 기관사, 판사, 검사, 엔지니어, CEO | 남성 연상 |
| 전통적 여성 직종 | 간호사, 보육교사, 유치원교사, 비서, 승무원 | 여성 연상 |
| 중립/전문직 | 의사, 교수, 변호사, 연구원, 작가 | 측정 대상 |

이 설계 자체가 연구의 기여 중 하나다 — 한국 사회 맥락에서 어떤 직업이 얼마나 강하게 젠더화되어 있는지를 임베딩 공간에서 확인하는 것이 목표이기 때문이다.

### 3. WEAT 계산

WEAT effect size $d$는 다음과 같이 정의된다:

$$d = \frac{\mu_A - \mu_B}{\sigma}$$

여기서 $\mu_A$는 목표 단어 집합 A(예: 남성 단어)와 두 속성 집합의 평균 코사인 유사도 차이의 평균, $\sigma$는 전체 목표 단어 집합에 대한 해당 차이의 표준편차다.

구체적으로:

$$s(w, A, B) = \text{mean}_{a \in A} \cos(w, a) - \text{mean}_{b \in B} \cos(w, b)$$

$$d = \frac{\text{mean}_{x \in X} s(x,A,B) - \text{mean}_{y \in Y} s(y,A,B)}{\text{std}_{z \in X \cup Y} s(z,A,B)}$$

$d > 0$이면 X가 A(남성)에, $d < 0$이면 Y가 A에 더 강하게 연관됨을 의미한다.

통계적 유의성은 순열 검정(permutation test, $p < 0.05$)으로 검증한다.

### 4. 시각화

- 직업 카테고리별 WEAT effect size bar chart
- 젠더 단어 및 직업 단어의 PCA 2D 투영 산점도
- 코사인 유사도 히트맵 (직업어 × 젠더 속성어)

### 5. 디바이어싱 비교

Bolukbasi et al.(2016)의 Hard debiasing 적용 후 동일한 WEAT 측정을 반복하여 편향 완화 전후 effect size 변화를 정량적으로 비교한다. 디바이어싱이 의도하지 않은 의미 정보까지 손실시키는지(overcorrection)도 함께 확인한다.

---

## 구현 계획

### 기술 스택

```
Python 3.10+
gensim          # 임베딩 모델 로드 및 유사도 계산
numpy           # WEAT 수치 계산
scikit-learn    # PCA, 통계 검정
matplotlib / seaborn  # 시각화
pandas          # 결과 정리
```

### 단계별 일정

| 단계 | 내용 | 예상 소요 |
|---|---|---|
| 환경 설정 및 모델 로드 | gensim으로 사전학습 모델 로드, 단어 확인 | 2~3시간 |
| Seed set 확정 | 단어 목록 설계, 모델 내 존재 여부 검증 | 반나절 |
| WEAT 구현 | 코사인 유사도, effect size, permutation test | 1~2일 |
| 시각화 | bar chart, PCA, heatmap | 반나절 |
| 디바이어싱 | Hard debiasing 구현 및 전후 비교 | 1일 |
| 결과 해석 및 정리 | README, 보고서 작성 | 1~2일 |

---

## 예상 결과 및 기여

**기술적 기여**  
한국어 임베딩에 적용 가능한 WEAT 파이프라인과, 한국 사회 맥락에 맞게 설계된 젠더-직업 자극 단어 목록을 공개 리소스로 제공한다.

**연구적 기여**  
영어 중심 편향 연구의 결과가 한국어에도 재현되는지, 한국 특수적 패턴(예: 군인, 승무원, 보육교사)이 어느 정도 강도로 나타나는지를 최초 측정한다.

**비판적 기여**  
임베딩이 학습 데이터의 사회적 불평등을 그대로 흡수하고 재생산한다는 사실을, 한국어 데이터로 실증함으로써 Responsible AI 담론의 한국어 맥락 근거를 제공한다.

---

## 선행 연구

| 연구 | 기여 |
|---|---|
| Bolukbasi et al. (2016). *Man is to Computer Programmer as Woman is to Homemaker?* | 영어 Word2Vec 젠더 편향 최초 실증, Hard debiasing 제안 |
| Caliskan et al. (2017). *Semantics derived automatically from language corpora contain human-like biases.* | WEAT 방법론 확립 |
| Bender et al. (2021). *On the Dangers of Stochastic Parrots.* | LLM의 사회적 편향 재생산 문제 제기 |
| Ko et al. (2023). *KoBBQ: Korean Bias Benchmark for Question Answering.* | 한국어 LLM 사회적 편향 측정 최초 시도 |
| Blodgett et al. (2020). *Language (Technology) is Power.* | NLP 편향 연구의 사회적 맥락 필요성 |

---

## 한계 및 향후 연구

본 프로젝트는 단어 임베딩 레벨에 한정되며, BERT 계열 문맥적 임베딩(contextual embedding)에서의 편향은 다루지 않는다. 또한 자극 단어 목록은 연구자의 판단에 기반하므로 선택 편향(selection bias)이 개입할 수 있다. 향후 연구에서는 KoBERT, EXAONE 등 한국어 LLM의 문맥적 편향 측정으로 확장하고, 자극 단어 목록의 타당성 검증을 위한 설문 기반 노멀라이제이션을 추가할 수 있다.

---

## 참고 리소스

- Korean-fasttext: https://fasttext.cc/docs/en/crawl-vectors.html
- KoBBQ dataset: https://github.com/naver-ai/KoBBQ
- WEAT 원논문 구현 참고: https://github.com/wordbias/WEAT

---

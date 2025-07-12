---
title: "Attention"
last_modified_at: 2023-01-03

tags:
  - attention
  - transformer
toc: true
toc_sticky: true

categories: anytopic
published: true
---

## 1. 어텐션이란?

어텐션은 문장에서 어떤 단어가 현재 단어와 **얼마나 관련이 깊은지(유사도)**를 계산한 뒤,  
그 **중요도에 따라 정보(Value)를 가중합**해서 문맥 정보를 반영하는 기법

---

## 2. 예시 문장

문장:  
**The cat sat on the mat**

총 6개의 단어가 있으며, 각각 토큰으로 나누어 처리된다.

우리는 현재 단어 **"sat"**에 대해 어텐션을 계산할 것이다.  
즉, "sat"이 문장에서 다른 단어들로부터 어떤 정보를 얼마큼 받아야 하는지를 계산한다.

---

## 3. Q, K, V 정의 및 예시

각 단어는 모델에 의해 일정한 차원의 벡터로 변환된다 (예: d_model = 4).  
각 단어 벡터로부터 다음 3가지가 생성된다:

- Q: 현재 단어의 query (정보를 찾는 기준)
- K: 각 단어의 key (정보를 제공할 수 있는 열쇠)
- V: 각 단어의 value (제공할 정보)

### 예시 (차원: d_model = 4)

| Token | Q (1x4)            | K (1x4)            | V (1x4)            |
|-------|--------------------|--------------------|--------------------|
| the₁  | [1, 0, 1, 0]       | [1, 0, 1, 0]       | [0.1, 0.0, 0.1, 0.0]|
| cat   | [0, 1, 0, 1]       | [0, 1, 0, 1]       | [0.0, 0.2, 0.0, 0.2]|
| sat   | [1, 1, 1, 1]       | [1, 1, 1, 1]       | [0.3, 0.3, 0.3, 0.3]|
| on    | [0, 0, 1, 1]       | [0, 0, 1, 1]       | [0.0, 0.0, 0.1, 0.1]|
| the₂  | [1, 0, 0, 1]       | [1, 0, 0, 1]       | [0.1, 0.0, 0.0, 0.1]|
| mat   | [1, 1, 0, 0]       | [1, 1, 0, 0]       | [0.2, 0.2, 0.0, 0.0]|

---

## 4. 어텐션 계산 (Query = "sat")

### 1단계: 유사도 계산 (Q · Kᵀ)

현재 Query = [1, 1, 1, 1]  
각 단어의 Key와 내적(dot product) 수행:

```
Q · Kᵢ = sum(Q[i] * Kᵢ[i])
```

| Token | Q·Kᵢ = 유사도 점수 |
|-------|------------------|
| the₁  | 1+0+1+0 = 2      |
| cat   | 0+1+0+1 = 2      |
| sat   | 1+1+1+1 = 4      |
| on    | 0+0+1+1 = 2      |
| the₂  | 1+0+0+1 = 2      |
| mat   | 1+1+0+0 = 2      |

→ 결과 shape: (1 x 6)

---

### 2단계: 스케일 조정

모델 차원(dk) = 4 → √dk = 2

→ 점수를 2로 나눈다:

| Token | scaled score = Q·Kᵢ / 2 |
|-------|------------------------|
| all   | 2/2 = 1.0 (sat만 4/2=2.0) |

→ scaled scores = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0]

---

### 3단계: softmax 적용

softmax(xᵢ) = exp(xᵢ) / sum(exp(xⱼ))

```
exp(1.0) ≈ 2.718, exp(2.0) ≈ 7.389

softmax = [2.718, 2.718, 7.389, 2.718, 2.718, 2.718]
sum ≈ 20.979

attention_weights = 각 값 / 20.979
```

| Token | softmax 비율 (approx.) |
|-------|------------------------|
| all (except sat) | ≈ 0.1296     |
| sat             | ≈ 0.352      |

→ attention_weights shape: (1 x 6)

---

### 4단계: 가중합 계산 (attention weights × V)

각 단어의 Value 벡터에 softmax 가중치를 곱하고 더한다:

예시:

```
context_vector = 
0.13*[0.1, 0.0, 0.1, 0.0] +
0.13*[0.0, 0.2, 0.0, 0.2] +
0.35*[0.3, 0.3, 0.3, 0.3] +
0.13*[0.0, 0.0, 0.1, 0.1] +
0.13*[0.1, 0.0, 0.0, 0.1] +
0.13*[0.2, 0.2, 0.0, 0.0]
```

→ 결과는 shape (1 x 4)의 context vector

---

## 5. 전체 과정 요약 (shape 포함)

| 단계 | 연산 | shape |
|------|------|--------|
| Q · Kᵀ | (1 x 4) × (6 x 4)ᵀ → (1 x 6) | 유사도 점수 |
| 나누기 √dk | (1 x 6) | 스케일 조정 |
| softmax 적용 | (1 x 6) | 중요도 확률 |
| softmax × V | (1 x 6) × (6 x 4) → (1 x 4) | context vector 생성 |

---

## 6. 핵심 정리

- **Q**는 "지금 이 단어가 어떤 정보를 원하나?"를 나타냄
- **K**는 "내가 가진 정보의 특징은 이것이다"
- **Q·Kᵀ**는 "너와 나의 관련도"
- **softmax**는 중요도를 정규화
- **V**는 실제 정보를 담고, softmax로 가중합
- **출력은 context vector**이며, 현재 단어의 문맥 정보를 반영한 표현


## [핵심] 7.실습

    LLaMA 기반 모델에서 문장 입력 후 attention을 추출

    마지막 레이어, 첫 번째 헤드의 attention map을 출력

    문장을 토큰 단위로 나누고, attention이 어떻게 흐르는지 확인

```python
# 1. Hugging Face transformers 설치 (Colab or 로컬 only)
# !pip install transformers
```

→ 이 코드는 Colab 또는 로컬 환경에서 한 번만 실행하면 된다.

---

```python
# 2. LLaMA 모델 로드 (TinyLlama: 가장 가벼운 버전)
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
```

→ TinyLLaMA는 Hugging Face에서 공개한 가장 작은 LLaMA 변형으로, 로컬에서 실행하기 쉬움  
→ `output_attentions=True`로 어텐션 가중치도 함께 반환받는다.

---

```python
# 3. 입력 문장 정의 및 토크나이징
sentence = "The cat sat on the mat."
inputs = tokenizer(sentence, return_tensors="pt")
```

→ 예시 문장을 정의하고 `tokenizer`로 토큰화  
→ 반환된 `inputs`는 input_ids와 attention_mask 등을 포함

---

```python
# 4. 모델 forward pass 및 어텐션 출력 받기
import torch

with torch.no_grad():
    outputs = model(**inputs)  # output.attentions 포함

attentions = outputs.attentions  # List of tensors (num_layers, batch, num_heads, seq_len, seq_len)
```

→ 학습 없이 forward만 실행 (`no_grad`)  
→ attention은 각 레이어와 헤드마다 존재하는 리스트 형태로 반환됨

---

```python
# 5. 마지막 레이어, 첫 번째 헤드의 attention matrix 추출
last_layer_attn = attentions[-1]       # shape: (batch, num_heads, seq_len, seq_len)
attn_matrix = last_layer_attn[0, 0]    # shape: (seq_len, seq_len)
```

→ 마지막 레이어의 첫 번째 헤드의 어텐션 가중치만 추출  
→ 이 행렬은 각 토큰이 어떤 토큰에 얼마나 주의를 기울였는지를 보여줌

---

```python
# 6. 토큰 목록 확인
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(tokens)
```

→ `[CLS]`, `[SEP]`, `'the'`, `'cat'`, … 등 토크나이즈된 실제 토큰 확인

---

```python
# 7. pandas로 시각화
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(attn_matrix.numpy(), index=tokens, columns=tokens)

plt.figure(figsize=(8, 6))
sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
plt.title("Last Layer Head 0 Attention Map")
plt.xlabel("Key (Attended To)")
plt.ylabel("Query (Attending)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

→ attention 가중치를 히트맵 형태로 시각화  
→ 행은 Query, 열은 Key → 셀 값이 클수록 해당 위치에 더 많은 주의를 줬음을 의미

---

## ✅ 결과 해석

- `"sat"` 토큰이 `"cat"`이나 `"mat"` 등에 얼마나 집중했는지 확인 가능
- `"the"` 같은 단어가 많이 등장하면 self-attention이 어떻게 분배되는지도 시각화됨
- 실제 어텐션 분포를 확인하면서 트랜스포머 모델이 문장을 어떻게 "이해"하는지 분석 가능


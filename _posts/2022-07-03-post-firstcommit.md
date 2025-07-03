---
title: "[1일차] 딥러닝이란?"
last_modified_at: 2022-07-03
categories:
  - deeplearning
tags:
  - keras
  - fashion-mnist
  - logistic-regression
  - softmax
  - activation-function
  - cross-entropy-loss
  - one-hot-encoding
  - dense-layer
  - SGDClassifier
  - reshape
  - nbytes
---

## 패션 MNIST 데이터셋 소개

딥러닝을 처음 배울 때 가장 먼저 접하게 되는 데이터셋 중 하나가 **Fashion MNIST**입니다.  
이전에는 손으로 쓴 숫자 이미지(MNIST)가 가장 많이 쓰였지만,  
실생활과 좀 더 가까운 데이터를 다뤄보고 싶다면 이 Fashion MNIST가 훌륭한 출발점입니다.

이 데이터는 28x28 픽셀로 된 흑백 이미지이며,  
티셔츠, 바지, 코트, 신발 등 **총 10가지 패션 아이템 클래스**로 구성되어 있습니다.

공식 문서: [Keras: fashion_mnist](https://keras.io/api/datasets/fashion_mnist/)

```python
from keras.datasets import fashion_mnist
(train_input, train_target), (test_input, test_target) = fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)
```

출력:
```
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
```

---

## 클래스 라벨 확인

Fashion MNIST의 클래스는 다음과 같이 정의되어 있습니다:

| 라벨 번호 | 의미         |
|-----------|--------------|
| 0         | T-shirt/top  |
| 1         | Trouser      |
| 2         | Pullover     |
| 3         | Dress        |
| 4         | Coat         |
| 5         | Sandal       |
| 6         | Shirt        |
| 7         | Sneaker      |
| 8         | Bag          |
| 9         | Ankle boot   |

모든 클래스는 6,000개씩 균등하게 구성되어 있습니다.

```python
import numpy as np

classes, counts = np.unique(train_target, return_counts=True)
print(dict(zip(classes, counts)))
```

---

## 훈련 이미지 시각화

우선 실제 데이터가 어떻게 생겼는지 시각화해 보겠습니다.

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
```

이미지는 28x28 크기의 흑백 픽셀이며, 저해상도라 다소 흐릿하지만 실루엣을 파악할 수 있습니다.

---

## 로지스틱 회귀 모델로 분류해 보기

딥러닝을 사용하기 전에, 먼저 고전적인 머신러닝 방법인 **로지스틱 회귀(Logistic Regression)** 로 이 문제를 풀어보겠습니다.

---

### 데이터 전처리

이미지 데이터는 (60000, 28, 28)의 3차원 배열입니다.  
이를 로지스틱 회귀에 넣기 위해선 **2차원 배열(샘플 수 × 특성 수)**로 펼쳐야 합니다.  
또한, 픽셀값을 0~1 사이로 정규화해야 학습이 잘 됩니다.

```python
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
```

`reshape(-1, 784)`에서 `-1`은 샘플 수를 자동으로 계산하라는 의미입니다.

```python
print(train_scaled.shape)         # (60000, 784)
print(train_input.nbytes / 1024**2)  # 약 47.6 MB
```

---

### SGDClassifier로 로지스틱 회귀 구현

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate

sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
```

출력 예:
```
0.8194
```

약 82% 정확도입니다.

---

## 로지스틱 회귀의 수식 구조

로지스틱 회귀는 다음과 같은 수식을 통해 예측값을 계산합니다:

```
z = w1*x1 + w2*x2 + ... + w784*x784 + b
```

- 각 픽셀(x₁ ~ x₇₈₄)에 대해 가중치(w)를 곱하고, 편향 b를 더합니다.
- z 값을 소프트맥스(softmax) 함수에 넣으면 **각 클래스에 속할 확률**이 됩니다.

10개의 클래스가 있으므로, 이 식은 **클래스마다 하나씩 총 10개**가 필요합니다.  
즉, 784개의 입력 × 10 + 10개의 편향 = **총 7,850개의 파라미터**가 존재합니다.

---

## 딥러닝 모델로 확장하기

로지스틱 회귀는 입력과 출력만 있는 '얕은 모델'입니다.  
딥러닝은 여기에 **은닉층(hidden layer)** 을 추가해 더 복잡한 패턴을 학습할 수 있게 합니다.

---

### 입력층 정의

```python
inputs = keras.layers.Input(shape=(784,))
```

입력층은 실제로 연산을 하진 않지만, **입력의 형태(shape)를 정의**합니다.

---

### 밀집층(Dense, Fully Connected Layer)

```python
dense = keras.layers.Dense(10, activation='softmax')
```

- 뉴런 수: 10개 (클래스 수와 동일)
- `Dense()`는 이전 층의 모든 뉴런과 현재 층의 모든 뉴런을 **전부 연결**해 주는 층입니다. 이를 Fully Connected Layer라고 합니다.
- softmax 함수는 각 뉴런의 출력값을 **확률 분포 형태로 정규화**해 줍니다.

---

### 모델 구성 및 컴파일

```python
model = keras.Sequential([inputs, dense])
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

- `sparse_categorical_crossentropy`: 타깃 레이블이 정수형일 때 사용하는 다중 분류 손실 함수입니다.

---

### 이진 분류 vs 다중 분류에서의 크로스 엔트로피

#### 이진 분류:

- 출력 뉴런은 1개
- 시그모이드 함수로 [0,1] 사이 확률 출력
- 손실 함수는: `-log(a)` (타깃이 1일 때), `-log(1-a)` (타깃이 0일 때)

#### 다중 분류:

- 출력 뉴런은 클래스 수만큼 존재
- softmax를 통해 [a₁, a₂, ..., aₙ] 확률 출력
- 타깃은 one-hot 인코딩되어 특정 클래스만 1이고 나머지는 0
- 손실은 `-log(정답 클래스 확률)`만 계산

예:

```python
# softmax 출력: [0.7, 0.1, 0.05, ..., 0.01]
# 타깃(one-hot): [0, 1, 0, ..., 0]
# 손실: -log(0.1)
```

Keras에서는 타깃값이 정수일 경우, 자동으로 해당 클래스를 one-hot으로 간주해 손실을 계산합니다.

---

### 모델 훈련 및 평가

```python
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
```

- `fit()`은 모델을 훈련합니다.
- `evaluate()`는 검증 데이터에 대한 손실과 정확도를 반환합니다.

---

## 두 모델 비교

```python
# Scikit-learn 방식
model = SGDClassifier(loss='log_loss', max_iter=5)
model.fit(train_scaled, train_target)
model.score(val_scaled, val_target)

# Keras 방식
inputs = keras.layers.Input(shape=(784,))
dense = keras.layers.Dense(10, activation='softmax')
model = keras.Sequential([inputs, dense])
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
```

---

## 핵심 정리

- **신경망**은 생물학적 뉴런에서 영감을 받은 알고리즘입니다.
- **Dense Layer**는 앞층의 모든 뉴런과 완전히 연결된 기본 구조입니다.
- **Softmax**는 다중 분류에서 사용되는 출력 정규화 함수입니다.
- **Cross Entropy Loss**는 모델이 출력한 확률과 실제 정답 간의 차이를 수치로 나타냅니다.
- **One-hot Encoding**은 다중 분류 문제에서 타깃을 표현하는 방법입니다.

---

## Keras 주요 함수 요약

| 함수 | 설명 |
|------|------|
| `Input(shape)` | 입력층 정의 |
| `Dense(units, activation)` | 밀집층(fully connected layer) 생성 |
| `Sequential([layers])` | 모델 구조 정의 |
| `compile(loss, metrics)` | 손실 함수와 지표 지정 |
| `fit(X, y, epochs)` | 모델 훈련 |
| `evaluate(X, y)` | 모델 평가 |

---
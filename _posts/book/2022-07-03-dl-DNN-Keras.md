---
title: "Dense Neural Network"
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
toc: true
toc_sticky: true
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

## DNN(심층 신경망)

다시 keras를 코랩에서 부르고, MNIST Fashion 데이터셋을 가져와 전처리(실수화, 1차원 리스트 변환)하고 학습/검증용으로 분리해봅시다.

```python
import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fasion_mnist.load_data()
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, train_target, val_scaled, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
```

오늘은 인공 신경망에 층을 하나만 추가해 봅시다. 입력층과 출력층 사이 '은닉층'이 추가됩니다. 
은닉층 내 각 뉴런의 선형 방정식 출력값에 활성화 함수가 적용됩니다. 
출력층에 적용되는 활성화 함수는 종류가 제한됩니다.
- 이진분류는 시그모이드
- 다중분류는 소프트맥스 
이에 비해 은닉층 뉴런에 적용하는 활성화함수는 자유롭게 사용할 수 있습니다. (렐루, 젤루 등)

> 회귀를 위한 신경망의 출력층에서는 활성화함수가 없습니다. 회귀 출력값 자체가 '숫자'기 때문에 출력층의 선형 방정식 계산을 그대로 출력하면 됩니다. 따라서 선형방정식 출력값의 '비선형을 위한 활성화함수'는 필요없습니다. 

### 은닉층에 왜 활성화 함수를 적용할까요? 

다음 2개의 선형 방정식을 보면, 왼쪽의 첫 번째 식(4a+2=b)에서 계산된 b가 두 번째 식(3b-5=c)에서 c를 계산하기 위해 쓰입니다.
하지만 두 번째 식에 첫 번째 식을 대입하면 오른쪽 식(12a+1=c)처럼 하나로 합칠 수 있죠. 이렇게 되면 b는 사라집니다. b가 하는 일이 없게 됩니다.

> (1)4a+2=b (2)3b-5=c ----> 12a+1=c

신경망도 은닉층에서 선형적인 산술 계산만 수행한다면 수행역할이 없는 셈이 됩니다. 선형 계산을 적당하게 비선형적으로 비틀어 주어야 합나. 그래야 다음 층의 계산과 단순하게 합쳐지지 않고, 나름의 은닉층만의 역할을 수행할 수 있습니다.

마치, 다음과 같습니다.

> 4a+2=b -> lob(b)=c -> 3k-5=c

그리고, 모든 은닉층에는 항상 활성화함수가 포함되어 있으니 생략되더라도 있다고 생각해야합니다. 활성화함수는 뉴런의 선형방정식 결과 z를 입력값으로 가진 함수 결과값 a를 다음 층에 전달합니다.

시그모이드 함수는 뉴런의 출력 z값은 0~1사이로 압축합니다. 케라스로 시그모이드 활성화 함수를 사용한 은닉층 하나를 추가해봅시다.

inputs = keras.layers.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

dense1이 은닉층이고 100개의 뉴런을 갖는 밀집층입니다. 100개의 뉴런이 출력한 값들은 모두 시그모이드를 거칩니다. 

> 은닉층 내 뉴런 수는 적어도 출력층의 뉴런보다는 많게 만들어야 합니다. 클래스 10개에 대한 확률을 예측해야 하는데 이전 은닉층의 뉴런이 10개보다 적다면 부족한 정보가 전달될 것이기 때문입니다.

그 다음 dense2는 앞서 만든 출력층으로 10개의 뉴런을 가지며, 다중분류이므로 소프트맥스 활성화함수를 지정했습니다.

### DNN 만들기

model = keras.Sequential([inputs, dense1, dense2])

Sequential 객체 생성 시 여러 층을 추가하려면 이와 같이 리스트에 순서대로 추가하면 됩니다. 주의할 것은 입력층과 출력층은 반드시 앞과 뒤에 위치해야 합니다.

> 케라스는 모델의 summary()
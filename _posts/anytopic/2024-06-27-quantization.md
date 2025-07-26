---
title: "양자화(Quantization)로 딥러닝 모델 가볍게 쓰기"
last_modified_at: 2025-07-26

tags:
  - quantization
  - model-compression
  - pytorch
  - large-language-models
  - deployment
toc: true
toc_sticky: true

categories: deep-learning
published: true
---

양자화(Quantization)는 부동소수점(FP32·FP16) 대신 "더 낮은 정밀도의 정수·저비트 표현(8‑bit, 4‑bit, 2‑bit 등)"으로  
딥러닝 모델의 파라미터·활성값을 근사해 "메모리 사용량·연산량을 크게 줄이는 기술"입니다.

---

## 1. 양자(quantum)의 사전적 정의 vs 딥러닝 ‘양자화’

| 구분 | 의미 |
| --- | --- |
| **물리학 ‘양자’** | 에너지·빛이 불연속적 최소 단위로 존재한다는 개념 |
| **컴퓨터 ‘양자화’** | 연속값(실수)을 **이산 구간(정수)**으로 근사해 디지털로 표현 |

> 딥러닝에서의 **Quantization**은 “모델 수를 **작게 쪼개서(discretize)** 표현한다”는 점에서 물리학의 ‘양자’ 개념과 유사한 **“근사·단순화”** 작업이다.

---

## 2. 양자화란?

1. **표현 비트 수 감소**   
   - FP32 → INT8(8‑bit) 또는 4‑bit 등  
   - 메모리 사용량 · 대역폭 ↓ (32 bit → 8 bit면 **4 배** 절감)
2. **양자화 스케일/제로포인트(QScale, ZP)** 로 범위 재매핑  
   - 실수 범위 ↔ 정수 범위 간 선형 변환
3. **수정된 연산(양자화 연산자 / QLinear)** 사용  
   - 정수 곱셈 + 비트시프트로 구현 가능  
   - 일부 하드웨어(NPU, TPU, ARM CPU)에서 대폭 가속
4. **종류**  
   | 방법 | 특징 | 정확도 | 준비 비용 |
   | --- | --- | --- | --- |
   | **동적 PTQ** (Post‑Training Dynamic) | 가중치만 INT8, 활성은 계층별 on‑the‑fly FP→INT 변환 | 손실 적음 | 쉽다 |
   | **정적 PTQ** (Post‑Training Static) | 가중치·활성 모두 INT8, Calibration 데이터 필요 | 약간 감소 | 중간 |
   | **QAT** (Quantization‑Aware Training) | 학습 시 양자화 노이즈를 모사 | 최고 | GPU 재학습 필요 |
   | **GPTQ / AWQ / NF4 4‑bit** | LLM 특화, 근사 행렬분해·재정렬 | 매우 좋음 | 복잡 |
   | **배터리류 2‑bit** | 특수 하드웨어 전용 | 정확도↓ | 어렵 |

---

## 3. PyTorch 기반 양자화 예제

### 3‑1. 작은 CNN – 동적/정적 PTQ

```python
import torch, torch.nn as nn
from torch.quantization import quantize_dynamic, prepare, convert

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3), nn.ReLU(),
            nn.Conv2d(8, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):                     # Bx1x28x28
        x = self.features(x).flatten(1)       # Bx16
        return self.classifier(x)             # Bx10

model_fp32 = TinyCNN()
model_fp32.load_state_dict(torch.load("cnn_fp32.pth"))

# ▶ 동적 양자화 (Linear·LSTM·Transformer 계층 지원)
model_int8_dyn = quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# ▶ 정적 양자화 (Conv 포함)
model_to_q = prepare(model_fp32, inplace=False)   # Calibration 준비
for _ in range(100):                              # 대표 데이터 100개
    dummy = torch.randn(1, 1, 28, 28)
    model_to_q(dummy)                             # 통계 수집
model_int8_stat = convert(model_to_q)             # INT8 변환

print("FP32 :", sum(p.numel() for p in model_fp32.parameters())*4/1e6, "MB")
print("INT8 :", sum(p.numel() for p in model_int8_stat.parameters())/1e6, "MB").
```

## 양자화의 다섯 가지 핵심 기법 ― 원리와 특징

모델도 양자화(Quantization)기법으로 가볍게 만들 수 있다!
아래 다섯 가지 방법은 “언제, 어디서, 어떻게 압축하느냐”에 따라 성격이 달라집니다.

---

### 1) 동적 PTQ(Post‑Training *Dynamic* Quantization)  
*― “읽으면서 바로바로 번역해 주는 통역사”*

- 두꺼운 영어 교재를 들고 다니면서, 필요한 문장만 읽으면 옆에서 통역사가 즉시 한국어로 말해 준다.  
- **원리:** 모델 파라미터(가중치)는 INT8로 미리 압축하지만, **활성값(중간 연산 결과)**은 매 호출 때마다 FP32↔INT8을 **즉시 변환**한다.  
- **장점:**  
  1. 훈련 데이터나 재학습이 필요 없다.  
  2. `nn.Linear`, `LSTM`, `Transformer` 블록만으로도 **메모리 절반 이하**로 줄어든다.  
- **제한:** 활성값은 여전히 ‘통역 대기 시간’이 든다 → CPU·RAM 대역폭이 빡빡하면 실속이 줄어든다.  
- **사용처:** “모델은 수정할 수 없고, 바로 배포해야 하는” 챗봇·서버 추론.

---

### 2) 정적 PTQ(Post‑Training *Static* Quantization)  
*― “책 전체를 번역해 둔 미리보기 해설집”*

- 교재 전체를 한국어로 **미리 번역**하여 책 옆구리에 붙여 두면, 읽을 때마다 통역사를 부를 필요가 없다.  
- **원리:** **가중치와 활성값 모두** INT8로 고정하려면, 몇 장의 ‘샘플 페이지(Calibration 데이터)’를 읽혀서  
  *“이 책엔 이런 단어·문장 구조가 반복되겠구나”* 하고 **범위(최댓값·최솟값)**를 계산한다.  
- **장점:**  
  1. 통역 단계(실시간 변환)가 제거되므로 **모바일·엣지 CPU**에서 속도 향상 폭이 가장 크다.  
  2. 정확도 손실이 의외로 작다—특히 Vision 모델.  
- **제한:**  
  1. 샘플이 너무 적으면 ‘단어 빈도’ 예측이 빗나가 정확도 하락.  
  2. Conv·BN·ReLU 등이 얽힌 그래프를 **재배치(fuse)**해야 하는 번거로움.  
- **사용처:** IoT 카메라, 오프라인 음성인식 등 **인터넷 없이도 돌아가야 하는 장치**.

---

### 3) QAT(Quantization‑Aware Training)  
*― “마우스피스를 물고 발음 교정 연습하는 스피치 트레이닝”*

- 영어 발표 연습을 **입안에 이물질(마우스피스)**을 낀 채로 하면,  
  실제 무대에서 빼고 발표할 때 발음·호흡이 훨씬 부드럽다.  
- **원리:** 학습 단계부터 계산 그래프 안에 **양자화 노이즈를 주입**해 모델이 자체적으로 오차를 보정하도록 유도.  
- **장점:**  
  1. INT8은 물론 **4‑bit**까지도 거의 원본 정확도를 유지한다.  
  2. 하드웨어 특성(대역폭·파이프라인)까지 ‘몸에 밴’ 가중치를 학습한다.  
- **제한:**  
  1. GPU 학습 리소스가 다시 필요—“모델 다 끝냈는데 또 훈련?”  
  2. 프레임워크·그래프 변환기가 지원해야 한다.  
- **사용처:** **캐시히트율이 생명**인 추천시스템, Self‑Driving 등 **정확도 손실을 거의 허용 못 하는** 영역.

---

### 4) GPTQ·AWQ·NF4 4‑bit ― LLM 전용 특수 압축  
*― “벽장을 통째로 리모델링해 옷걸이 간격까지 재설계”*

- 부피 큰 겨울 코트를 접어 넣는 대신, **옷장 봉을 앞뒤 두 줄로 만들고** 코트 어깨에 맞춰  
  옷걸이를 커스터마이징하면 **공간·무게를 절반**으로 줄이면서도 구김이 덜 간다.  
- **원리:**  
  1. 거대한 **행렬(W)**을 일괄적으로 INT4로 줄이지 않고, **블록별·채널별**로 오차를 최소화하는 **재배열·근사 분해**를 수행.  
  2. 핵심 파라미터에는 **NF4(Non‑Uniform 4‑bit)**, 나머지엔 ‘더블 양자화’ 같이 계층적 스케일을 부여.  
- **장점:**  
  - 70B 이상 LLM도 **A100 40 GB**에 로드 가능.  
  - LoRA 미세조정과 병용하여 “가벼운데 똑똑한” 대화형 모델을 만든다.  
- **제한:**  
  - 과정이 비선형·픽스라 QAT처럼 *추가 학습이 어려움*.  
  - 아직까지는 대부분 **추론 전용**.  
- **사용처:** **Colab Pro+·Lambda 1×A100**처럼 제한적 VRAM 환경에서 Llama‑2‑70B, Mistral‑MoE 8×22B 등 **초대형 LLM**을 띄울 때.

---

### 5) 극한의 저비트(2‑bit·1‑bit) & 비대칭 기법  
*― “모스 부호만으로 장문의 소설 주고받기”*

- 문자 하나하나 보내기엔 무거워서, **모스 부호**처럼 짧은 ‘·‑’ 신호에 온갖 약어·줄임말을 압축한다.  
  해독기는 복잡해지지만, 무전기로도 대하소설을 전송할 수 있다.  
- **원리:**  
  1. **2‑bit**에서는 선택 가능한 값이 네 개뿐이므로, 가중치 분포를 **Log 또는 Power‑of‑Two** 형태로 매핑.  
  2. 정확도 손실을 버티려면 **인터폴레이션·Look‑Up Table** 같은 보조 구조가 필요.  
- **장점:** 휴대폰 NPU 같이 캐시 < 4 MB 환경에서도 대형모델 호흡 가능.  
- **제한:**  
  - 정확도 손실이 급격히 상승—“문장을 줄이다 못해 자음만 남긴 수준”.  
  - 범용 GPU 연산자는 아직 지원이 적다.  
- **사용처:** 초저전력 **웨어러블·마이크로컨트롤러** 또는 **ASIC 설계** 단계에서.

---

### 마무리 한 줄 요약

> 양자화 기법을 선택할 경우, 고려할 세 가지!

> ■ 재훈련 여유(시간·돈) 
> ■ 목표 플랫폼(CPU/GPU/NPU)  
> ■ 허용 가능한 정확도 손실
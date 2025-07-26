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

양자화(Quantization)는 부동소수점(FP32·FP16) 대신 **더 낮은 정밀도의 정수·저비트 표현(8‑bit, 4‑bit, 2‑bit 등)**으로  
딥러닝 모델의 **파라미터·활성값을 근사**해 **메모리 사용량·연산량을 크게 줄이는** 기술입니다.

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


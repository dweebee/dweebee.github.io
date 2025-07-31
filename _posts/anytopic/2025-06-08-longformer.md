---
title: "Longformer"
last_modified_at: 2025-06-08

tags:
  - attention
  - longformer
  - context
toc: true
toc_sticky: true

categories: anytopic
published: true
---

------------------------------------------------------------
1. TRANSFORMER 기본 복습
------------------------------------------------------------
- Self-Attention: 모든 토큰이 서로의 Query•Key•Value 를 곱해 가중합.
- 복잡도: 토큰 수 n → 연산/메모리 O(n²). 512 토큰까지만 실용적.
- BERT 류 Dense 모델: 긴 문서는 반드시 “토막-내기(Sliding Window)”
  혹은 Hierarchical 모델을 따로 설계해야 한다.

------------------------------------------------------------
1. DENSE vs. SPARSE ATTENTION
------------------------------------------------------------
[Dense]  모든 ○ 가 서로 연결 –> n²     (느림, 메모리 폭발)
[Sparse] 필요한 ● 만 연결 –> n·k      (k ≪ n)

핵심 아이디어
  “멀리 떨어진 대부분의 단어는 직접 대화할 필요가 없다.”
   → 연결(어텐션)을 과감히 0 으로 마스킹 → 계산 생략!

------------------------------------------------------------
2. LONGFORMER 아키텍처 한눈에
------------------------------------------------------------
(1) Sliding-Window  :  토큰 i 가 좌우 w/2 토큰만 본다.
(2) Global Tokens   :  선택된 소수 토큰은 모든 토큰과 상호작용.
(3) Dilated Window  :  (옵션) 창 간격을 d 로 벌려 먼 곳 힐끗.
→ 결과 복잡도 O(n)  (w, |G| 는 n 보다 훨씬 작다)

기본 설정
  • 사전학습 모델: longformer-base-4096 (L=12, hidden=768)
  • window 크기 w = 512  (왼쪽 256 + 오른쪽 256)

------------------------------------------------------------
3. SLIDING-WINDOW ATTENTION
------------------------------------------------------------
ASCII 그림 (n=12, w=4):

토큰1:  ○───●───●
토큰2:  ●───○───●───●
토큰3:  ●───●───○───●───●
... (이하 동일 패턴)

→ Conv1D 와 동일한 “1-D 합성곱” 형태, GPU 커널 최적화 쉬움.

------------------------------------------------------------
4. GLOBAL ATTENTION (전망대 토큰)
------------------------------------------------------------
• 모든 토큰과 양방향 연결.  
• 태스크별 전략 예
    – 분류       : 첫 토큰 [CLS] 를 Global.
    – QA         : 질문 토큰 전체를 Global.
    – Claim-검증 : [CLS] + Claim 문장 토큰을 Global.
• 개수 |G| 가 10 이하이면 추가 비용 O(n·|G|) 무시 가능.

------------------------------------------------------------
5. DILATED WINDOW
------------------------------------------------------------
• rate d = 2  →  한 칸 건너뛰며 w/2 개씩 어텐션.
• 장점: w 유지하면서 정보 파급 거리 ↑.
• 사용 예: 상위 레이어 일부 헤드만 dilated 로 설정.

------------------------------------------------------------
6. 계산 복잡도 & 메모리
------------------------------------------------------------
Dense  Attention : O(n²)  (512 토큰 = 262k 연결)
Longformer       : O(n·w + n·|G|)  ≈  O(n)  (n=4096, w=512 → 2M)
→ n 증가 시 기울기(gradient) 메모리도 선형 증가.  
  *4096 토큰, batch 1 → 약 24 GB (FP32 기준).*

------------------------------------------------------------
7. 코드 사용 – GLOBAL MASK
------------------------------------------------------------
from transformers import LongformerTokenizer, LongformerForSequenceClassification
tok   = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained(...)

doc   = "<CLS> CLAIM : ...  본문 시작 ..."
inputs = tok(doc, return_tensors="pt", padding="max_length",
             truncation=True, max_length=4096)

# (1) 모든 위치 0, 글로벌 토큰만 1
gmask = torch.zeros_like(inputs["input_ids"])
gmask[0, 0] = 1                               # <CLS>
claim_pos = (inputs["input_ids"] == tok.vocab['CLAIM']).nonzero()
gmask[0, claim_pos[:,1]] = 1

outputs = model(**inputs, global_attention_mask=gmask)

------------------------------------------------------------
8. 메모리 절약 TIP
-----------------------------------------------------------
• Mixed Precision (FP16): 메모리 ↓ ~40%, 속도 ↑(Tensor Core).
• Gradient Checkpointing: 중간 활성값 미저장, 역전파 시 재계산 –
  메모리 ↓↓, 연산 ↑ 약 20%.  → transformers.utils.apply_chunking_to_forward
• 둘 다 켜면 T4(16GB) 에서도 2K~3K 토큰 batch 1 학습 가능


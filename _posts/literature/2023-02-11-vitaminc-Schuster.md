---
title: "Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence"
authors: [Tal Schuster, Adam Fisch, Regina Barzilay]
conference: NAACL 2021, 2021년 6월 6일-11일
arXivPubdate: 2021-03-15
github: https://github.com/TalSchuster/VitaminC
arxiv: 2103.08541
tags: [ContrastiveEvidence, FactVerification, WikipediaRevision, RobustModel, AdversarialTraining]
description: "Wikipedia 개정본 기반 대조적 증거를 활용한 강건한 사실 확인 시스템 개발"
categories: literature
published: true
---
## 1. 문제 정의
- 기존 사실 검증 데이터셋은 정적 증거에 의존, 증거 변경에 둔감
- 동적 자료(예: Wikipedia)에서 발생하는 미세한 사실 변경 인지·대응 필요
- 목표: 유사 언어·구조의 증거 쌍을 통해 모델의 **context-sensitive inference** 능력 강화

## 2. 제안 방법
- **VITAMINC**: 100K+ Wikipedia 실제 수정과 합성 수정 결합, 총 400K+ 대조적(contrastive) claim–evidence 쌍
- 파이프라인:
  1. [[FactualRevisionFlagging]]: 수정 전·후 문장 쌍 중 사실 변경 여부 판별
  2. [[FactVerification]]: claim↔evidence 관계(SUP/REF/NEI) 분류기 강화
  3. [[WordLevelRationales]]: 판단에 기여한 증거 내 단어 마스킹
  4. FactuallyConsistentGeneration:  
     - **Revision**: 주어진 claim에 맞춰 구문 수정  
     - **ClaimExtraction**: 수정 전·후 비교로 claim 자동 생성
- 차별점: 실전 수정 이력 기반 대조적 학습, 세부 과제(task) 추가 정의

## 3. Figures | Tables
- Wikipedia 문장 수정 전후 대비 예시: "76,129→86,205" → "91,757"로 업데이트, 초기 반박 → 지지 전환 설명 (Figure 1)
- VITAMINC 데이터 4개 과제 예시 시퀀스: 수정 감지 → 진위 분류 → 단어 근거 태깅 → 자동 수정/추출 흐름도 (Figure 2)
- VITAMINC 비율에 따른 adversarial 평가 정확도 곡선: 실 데이터 비율↑ → 강건성↑ (Figure 3)
- 문장 수정 유형별 예시: 사실·비사실 수정 vs. 대응하는 참/거짓 claim 작성 예시 (Table 1)
- VITAMINC 데이터 분할 통계: real vs synthetic, SUP/REF/NEI 비율 (Table 2)
- 수작업 수정 감지 과제 성능 비교: EditDist, BOW, ALBERT(diff/full), AUC·F1 (Table 3)
- Fact Verification/NLI 모델별 정확도: FEVER/MNLI/VitC 단독·조합 학습, adversarial·symmetric 평가 (Table 4)
- 단어 근거 태깅: unsupervised vs. distantly supervised(F1) (Table 5)
- 생성 과제 평가: Revision(SARI·BLEURT·fverdict), ClaimExtraction(ROUGE2·BLEU·fverdict), human 평가 (Table 6)


## 4. 실험 환경
- 프레임워크: PyTorch, HuggingFace Transformers  
- 모델: ALBERT-base/large, BERT, fastText, BART-base, T5-base  
- 데이터: Wikipedia 수정 이력 + FEVER 합성  
- 하이퍼파라미터: lr=2e-5, batch=32, max_seq_len=256  
- 학습 인프라: NVIDIA V100 GPU  

## 5. 실험 결과
- **수정 감지**: ALBERT-full AUC 91.97, F1 83.18%  
- **사실 검증**: VitC+FEVER ALBERT-xlarge 정확도 94.01%, adversarial 개선  
- **근거 태깅**: distantly supervised F1 47.36% (unsup 37.33%)  
- **자동 생성**: BART revision fverdict 76.26%, claim extraction fverdict 85.83%  

## 6. 배경지식 및 핵심 용어
[[ContrastiveEvidence]]  
유사한 언어·구조의 증거 쌍 중 하나만 claim을 지지하거나 반박하도록 구성된 데이터 구조.  

[[FactualRevisionFlagging]]  
수정 전·후 문장 비교를 통해 **사실적** 변경 여부(수치·정보 변경) 식별 과제.  

[[FactVerification]]  
주어진 증거 문장과 claim 간 관계를 **SUPPORTED/REFUTED/NEI**로 분류하는 문장 쌍 추론 과제.  

[[WordLevelRationales]]  
판단 근거가 된 증거 문장 내 단어 토큰을 마스킹·예측하여 **판단 이유**를 설명하는 과제.  

[[FactuallyConsistentGeneration]]  
주어진 claim에 맞춰 구문을 수정(Revision)하거나, 수정 전·후 차이로 claim을 추출(ClaimExtraction)하는 **사실 일관성** 생성 과제.  

## 7. 관련자료 링크
https://github.com/TalSchuster/VitaminC  
https://aclanthology.org/2021.naacl-main.52/  
https://arxiv.org/abs/2103.08541  

## 8. 논문의 기여 및 한계
- 기여: 대규모 대조적 벤치마크 VITAMINC 제안, 네 가지 세부 과제 정의  
- 한계: 증거 검색·다중 문장 추론 미포함, 복합 수정 유형 학습 필요  

## 9. 추가
- 후속 연구: multi-hop 증거 연결, cross-document 대조 학습  
- 실전 적용: 실제 위키피디아 수정 모니터링·자동화 도구 개발 가능  

---
title: "MAPLE: Micro Analysis of Pairwise Language Evolution for Few-Shot Claim Verification"
authors:
  - Xia Zeng
  - Arkaitz Zubiaga
conference: EACL 2024
arXivPubdate: 2024-01-29
github: https://github.com/XiaZeng0223/MAPLE
arxiv: 2401.16282
tags:
  - FewShotLearning
  - ClaimVerification
  - LanguageEvolution
  - PairwiseAnalysis
  - SemanticSimilarity
description: unlabeled data를 활용한 micro language evolution path를 통해 few-shot 환경에서 주장 검증 성능을 향상시키는 혁신적 접근법
categories: literature
published: true
---

## 1. 문제 정의
- **Few-shot claim verification**: 제한된 1–5샷 레이블만으로 claim 진위 판정  
- 기존 NLI 기반 방법(SEED, PET) 대규모 컴퓨팅 자원·대량 데이터 의존  
- 목표: **저비용·고효율** few-shot 검증 기법 제안

## 2. 제안 방법
- **MAPLE**: T5-small + LoRA를 이용해 claim↔evidence 생성 과정에서 발생하는 미세 언어 진화 경로 활용  
- 파이프라인 세 단계  
  1. **In-Domain Seq2Seq Training**  
     - claim→evidence, evidence→claim 양방향 학습  
     - LoRA로 unlabeled 쌍 학습 중 매 epoch마다 mutation(변형) 생성  
  2. **SemSim Transformation**  
     - (c,e,m) 트리플을 (c–e, c–m, e–m) 세 쌍으로 분리  
     - mpnet-base-v2 임베딩 후 코사인 유사도 ‘SemSim’ 점수 계산  
  3. **Logistic Classifier**  
     - n-shot 레이블 데이터(3n 인스턴스)로 SemSim 특성 학습  
     - SUPPORTS/REFUTES/NOT_ENOUGH_INFO 분류  

## 3. Figures | Tables
| 번호     | 유형    | 설명                                                                                             |
|---------|-------|------------------------------------------------------------------------------------------------|
| Figure 1 | 그림   | MAPLE 전체 흐름도: seq2seq 학습→mutation 생성→SemSim 계산→로지스틱 분류 단계별 구조                    |
| Figure 2 | 그림   | 5샷 환경 F1 곡선: MAPLE이 FEVER·cFEVER·SciFact에서 SOTA 성능 달성                                  |
| Figure 3 | 그림   | LoRA vs SFT vs NLPO 비교: LoRA 학습 효율·성능 유지 강조                                           |
| Figure 4 | 그림   | NLG 메트릭(NIST, BLEU, METEOR 등) 및 SemSim 비교: SemSim이 미세 언어 변화 포착에 최적               |
| Figure 5 | 그림   | 클래스별 SemSim 분포 예시: NOT_ENOUGH_INFO 낮은 유사도, SUPPORTS/REFUTES 높은 구분도                |
| Figure 6 | 그림   | 50샷 환경 F1 곡선: MAPLE 빠른 수렴·안정적 학습 입증                                                |
| Table 1  | 표    | FEVER, cFEVER, SciFact_oracle, SciFact_retrieved 샘플 수·클래스 분포                                 |
| Table 2  | 표    | unlabeled 데이터 풀의 클래스 비율 통계                                                             |
| Table 3–6| 표    | 1–5샷 환경 F1·Accuracy(100회 평균·표준편차) 세부 성능                                               |
| Table 7  | 표    | 클래스별 F1 세부 결과 (SUPPORTS/REFUTES/NOT_ENOUGH_INFO)                                           |
| Table 8  | 표    | LoRA vs SFT 런타임 비교: 파라미터 수·학습 시간                                                    |
| Table 9  | 표    | MAPLE 전체 런타임 및 자원 소모 (다양한 설정)                                                        |

## 4. 실험 환경
- 프레임워크: PyTorch, HuggingFace Transformers  
- 모델: T5-small+LoRA, BERT-base(MNLI), mpnet-base-v2, LLaMA 2 7B  
- 데이터셋: FEVER, Climate FEVER(cFEVER), SciFact_oracle, SciFact_retrieved  
- 하이퍼파라미터: lr=1e-4, batch=16, max_len=512, epoch=20, LoRA_dropout=0.1, α=32  
- 인프라: NVIDIA V100 GPU, Queen Mary Apocrita HPC

## 5. 실험 결과
- **1-shot F1**: MAPLE(FEVER) >0.60 vs SEED≈0.25, PET≈0.37, LLaMA≈0.38  
- **5-shot F1**: FEVER >0.70, cFEVER >0.40, SciFact_oracle≈0.45, SciFact_retrieved≈0.50  
- **안정성·노이즈 내성**: MAPLE 빠른 수렴·일관된 성능 향상, 베이스라인 대비 표준편차 감소  

## 6. 배경지식 및 핵심 용어
[[MicroAnalysis]]  
claim↔evidence seq2seq 학습 중 발생하는 **미세 언어 진화 경로** 분석  

[[SemSim]]  
mpnet-base-v2 임베딩 기반 **코사인 유사도**로 미세 의미 유사성 측정 지표  

[[Low-Rank Adaptation (LoRA)]]  
대형 PLM에 **저차원 행렬 추가**해 파라미터 효율성·학습 속도 개선 기법  

[[Few-Shot Learning]]  
소량(n샷)의 라벨만으로 모델 성능 학습·평가하는 **데이터 효율성** 강조 학습 방식  

## 7. 관련자료 링크
https://aclanthology.org/2024.findings-eacl.79/  
https://arxiv.org/abs/2401.16282  
https://github.com/XiaZeng0223/MAPLE  

## 8. 논문의 기여 및 한계
- 기여: unlabeled 쌍 기반 few-shot claim verification 프레임워크 제안, SemSim 메트릭 도입  
- 한계: higher-shot 시나리오 일반화 검증 필요, **증거 검색·multi-hop 추론** 미포함  

## 9. 추가
- 응용: 저자원 도메인·언어 적응, 실시간 fact-checking 도구 통합  
- 후속 연구: SemSim 범용 NLG 평가 지표화, human-in-loop 워크플로우 구축  

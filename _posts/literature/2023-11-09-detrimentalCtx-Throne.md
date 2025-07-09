---
title: "Detrimental Contexts in Open-Domain Question Answering"
authors: [Philhoon Oh, James Thorne]
conference: EMNLP, 2023
arXivPubdate: 2023-10-27
github: https://github.com/xfactlab/emnlp2023-damaging-retrieval
arxiv: 2310.18077
tags: [DetrimentalContexts, RetrieveThenRead, DamagingRetrieval, ContextFiltering, OpenDomainQA]
description: "retrieve-then-read 구조에서 ‘과도한 문맥’이 독이 되어 성능을 저하시킴을 규명하고, 해로운 passages 제거로 10% 이상 EM 개선"
categories: literature
published: true
---

## 1. 문제 정의
- Open-Domain QA의 retrieve-then-read 파이프라인에서, **더 많은** 문맥 제공이 성능 향상으로 직결되지 않음[1].  
- 일부 passages는 **모델 성능을 저해(detrimental)**, 정답 예측을 오히려 악화시킴.  
- 핵심 과제: 해로운 passages 식별·제거로 context-efficient QA 달성.

## 2. 제안 방법
- **Detrimental Context Identification**: reader(FiD)를 블랙박스로 사용, top-K passages 순차 투입하며 EM@k 변동 관찰[1].  
- **Passage Filtering**: EM 패턴 전이(0→1: definite positive, 1→0: definite negative) 기반 해로운 passages 판별 후 제거.  
- **효과**: 최대 10% EM 개선, 기존 FiD 구조 유지·추가 훈련 없이 달성.

## 3. Figures | Tables
| 번호      | 유형     | 설명                                                             |
|---------|--------|----------------------------------------------------------------|
| Figure 1  | 그림     | EM@k 패턴 예시: gold만 사용 시 정답 유지, 추가 문맥(P4) 투입 시 오답 전환[1]         |
| Figure 2  | 그림     | 시뮬레이션: 무작위/negative sampling 문맥 투입에 따른 EM 변화 및 Stability Error Ratio |
| Figure 3  | 그림     | DPR/SEAL/Contriever top-100 incremental EM vs AcEM 곡선              |
| Table 1   | 표      | Stability Error Ratio: random noise vs BM25 negative sampling 비교        |
| Table 2   | 표      | Probe 3(positive-leaning passages) 활용 시 NQ/TQA EM@100 vs AcEM@100 성능 |
| Table 3   | 표      | attention threshold별 EM@20 성능                                       |
| Table 4   | 표      | DP/DN 비율: highest attention vs transitioned prediction           |
| Table 5   | 표      | DP vs DN binary classification 성능                                 |

## 4. 실험 환경
- 프레임워크: PyTorch, HuggingFace Transformers  
- Reader: FiD-large (T5-base 기반)  
- Retrievers: DPR, SEAL, Contriever (top-100)  
- 데이터: NaturalQuestions, TriviaQA 개발·테스트 세트  
- 하이퍼파라미터: default  

## 5. 실험 결과
- **Incremental Inference**: top-100 AcEM@100 최대—NQ 62.3% vs EM@100 52.5%, TQA 77.7% vs 72.3%[1].  
- **Probe 3**: DP+SP 제외 DN 제거 입력 시 EM@100—NQ 61.8%, TQA 77.6% (AcEM@100 근접)[1].  
- **Few-Sentence Prediction**: EM@5로 5 passages만 사용해 NQ EM+12%, TQA +8% 상승.  
- **Attention Filtering**: 높은 cross-attention 문맥만 사용 시 EM·AcEM 모두 하락(Table 3).  
- **Binary Classification**: RoBERTa/T5/FiD-encoder 모두 DN 식별 불가, DP 과대예측(Table 5).

## 6. 배경지식 및 핵심 용어
[[RetrieveThenRead]]  
retriever가 passages 검색 후 reader가 그 문맥을 이용해 답 생성하는 **2단계 QA 아키텍처**.

[[Detrimental Retrieval]]  
관련하지만 **해로운 정보**가 모델 예측을 잘못 유도해 성능 저하를 초래하는 현상.

[[Exact Match (EM) Pattern]]  
top-k passages 사용 시 정답 일치 여부(1/0) 시퀀스. 전이 분석으로 문맥 타입 판별.

[[Accumulated EM (AcEM)]]  
top-1…k 모든 EM@i의 최대값. subset에서 정답 예측 가능성 측정 지표.

[[Probe-Based Selection]]  
EM 패턴 전이에 기반해 **definite positive/negative** passages 식별·필터링 기법.

## 7. 관련자료 링크
https://github.com/xfactlab/emnlp2023-damaging-retrieval  
https://aclanthology.org/2023.findings-emnlp.776/  
https://arxiv.org/abs/2310.18077  

## 8. 논문의 기여 및 한계
- **기여**: QA context filtering으로 FiD 성능 10% 이상 EM 개선, retrieve-then-read 모델의 해로운 문맥 문제 규명.  
- **한계**: EM 평가 한계(동의어·정답 변형 미포착), 대규모 LLM(RAG) order-variant 모델에 적용 어려움.  

## 9. 추가
- **후속 연구**: semantic answer equivalence 평가 통합, dynamic reranking·context selection 모델 개발.  
- **응용**: context-efficient QA 시스템 경량화, 실시간 retrieval pipeline 최적화.

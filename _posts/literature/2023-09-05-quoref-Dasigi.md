---
title: "Quoref: A Reading Comprehension Dataset with Questions Requiring Coreferential Reasoning"
authors: [Pradeep Dasigi, Nelson F. Liu, Ana Marasović, Noah A. Smith, Matt Gardner]
conference: EMNLP-IJCNLP, 2019
arXivPubdate: 2019-08-16
github: null
arxiv: 1908.05803
tags: [CoreferentialReasoning, SpanSelection, AdversarialCrowdsourcing, ReadingComprehension, Dataset]
description: "영문 위키피디아 문단에서 핵심 개체의 대등언급(coreference) 해소를 요구하는 24K+ 질의-응답 쌍으로 구성된 읽기 이해 벤치마크 QUOREF 제안"
categories: literature
published: true
---

## 1. 문제 정의
- 기존 MRC 벤치마크는 문장 내 패턴 매칭 중심, **coreference** 현상 평가 미흡  
- 긴 문단 내 **anaphoric expressions** 추적·해소 능력 필요  
- 목표: span-selection 질의가 **coreferential reasoning** 요구하는 데이터셋 구축  

## 2. 제안 방법
- **Crowdsourced collection**: 4.7K 문단, 24K+ 질문  
- 질문 작성 과제:  
  1. 문단 내 동일 개체를 가리키는 여러 표현 식별  
  2. 두 표현 연결해 질문 작성 (answer span 선택)  
  3. **Adversarial crowdsourcing**: uncased BERT QA(최초 SQuAD-1.1 모델) 답변 가능한 질문 배제  
- 의의: 모델이 surface cue 아닌 coreference 해소 뒤에 위치한 증거 사용하도록 유도  

## 3. Figures | Tables
| 번호    | 유형     | 설명                                                                                  |
|-------|---------|-------------------------------------------------------------------------------------|
| Figure 1 | 그림     | 예시 문단과 질문: 대명사·명사 해소 필요 지점 표시(이탤릭·볼드·밑줄) 및 핵심 멘션 연결             |
| Table 1  | 표      | 학습·검증·테스트 분할 통계: 문단 수, 질문 수, 길이·어휘 크기, multi-span 비율                    |
| Table 2  | 표      | 100샘플 분석: pronominal(69%)·nominal(54%)·multiple(32%)·commonsense(10%) 비율            |
| Table 3  | 표      | Baseline 성능: Heuristic vs QANet/BERT/XLNet, Dev/Test EM·F1 비교                           |

## 4. 실험 환경
- 프레임워크: AllenNLP, pytorch-transformers  
- 모델: QANet, QANet+BERT, BERT-base QA, XLNet-base QA  
- 전처리: 문단 truncate 400/1000 token, 질문 50/100 token  
- 최적화: AdamW lr=3e-5, batch=10, seq_len=512, stride=128, epoch=10  
- 평가: SQuAD EM, DROP-style macro-F1  

## 5. 실험 결과
- XLNet QA 최고: Dev F1 71.49%, Test F1 70.51%  
- BERT QA: Dev 64.95%, Test 66.39%  
- QANet+BERT: Dev 47.38%, Test 47.20%  
- Heuristic(passage-only) F1≈25–28%  
- 인간 성능 추정: EM 86.75, F1 93.41  

## 6. 배경지식 및 핵심 용어
[[Coreference]]  
두 개 이상의 **표현이 동일 개체**를 지칭하는 현상. 문맥 내 장거리 해소 필요.

[[Anaphoric expression]]  
문단 내 등장한 **선행 멘션**에 의존해 의미 해석되는 참조 어휘(예: 대명사).

[[Adversarial Crowdsourcing]]  
질문 작성 시 **모델 예측**(uncased BERT QA) 결과를 실시간 필터링해, surface cue 질문 배제.

[[Span Selection]]  
문단에서 모델이 **정답 구간(start/end index)** 을 직접 예측하는 MRC 태스크 형태.

[[Reading Comprehension Benchmark]]  
텍스트 이해·추론 능력 평가 위한 **질의-응답** 형태의 학습·평가 데이터셋.

## 7. 관련자료 링크
https://aclanthology.org/D19-1606/  
https://allennlp.org/quoref  
https://huggingface.co/datasets/allenai/quoref

## 8. 논문의 기여 및 한계
- **기여**: coreference-집중형 MRC 벤치마크·Adversarial crowdsourcing 절차 제안  
- **한계**: 완전한 코어퍼런스 클러스터링 미포함, multi-sentence retrieval 과제 미고려  

## 9. 추가
- 후속 연구: **multi-hop retrieval**, cross-paragraph reasoning 통합  
- 응용: 엔티티 추적 기반 QA·대화 시스템 핵심 역할  

---
title: "Automated Fact-Checking in Dialogue: Are Specialized Models Needed?"
authors: [Eric Chamoun, Marzieh Saeidi, Andreas Vlachos]
conference: EMNLP, 2023
arXivPubdate: 2023-11-14
github: null
arxiv: 2311.08195
tags: [RetrievalAdaptation,ClaimDetection,SentenceRetrievalEnhancement,CatastrophicForgetting,DialogueFactChecking]
description: "대화 맥락의 발화를 검증하기 위해 특화 모델 대신 기존 사실검증 모델의 입력·검색 단을 적응시키는 기법 제안"
categories: literature
published: true
---

## 1. 문제 정의
- 전통적 사실검증 모델은 독립적 문장(claim)에 최적화됨[1].  
- 대화 발화는 불완전·지시적 표현, 대명사·ellipsis·colloquialism 포함으로 검색·검증 어려움[1].  
- Fine-tuning 대화 전용 모델은 FEVER 등 독립문장 검증 성능 대폭 저하(**catastrophic forgetting**) 초래[1].  
- 목표: 별도 모델 없이 단일 모델로 **stand-alone** & **dialogue** 검증 모두 유지.

## 2. 제안 방법
- Retrieval Adaptation:  
  • 문서 검색 시 대화 맥락과 claim 결합하되, claim 유사도에 가중치 집중해 노이즈 감소[1].  
- Claim Detection:  
  • 발화 분절 후 각 부분의 evidence 유사도(semantic textual similarity) 최고인 서브문장 선택해 “검증 대상 claim”으로 변환[1].  
- Sentence Retrieval Enhancement:  
  • 문장(search hit) 유사도뿐 아니라 해당 문서 전체의 relevance 신호(rD)를 추가 입력해 evidence 랭킹 개선[1].  
- 이 기법들만 적용한 기존 모델(FEVER+VitC+retrieval+claimdet)이 DialFact 성능 상위권 유지하며 FEVER 성능 손실 無[1].

## 3. Figures | Tables
| 번호      | 유형    | 설명                                                                                          |
|----------|--------|---------------------------------------------------------------------------------------------|
| Figure 1 | 그림    | (1) 문서 검색: context+claim 가중 결합(파랑), (2) 문장 검색: r_s / r_D / R_D 결합(초록), (3) 발화 분절·STS 기반 claim 추출(빨강) 파이프라인[1] |
| Table 1  | 표     | FEVER & DialFact 검증 정확도 비교: Fine-tuned dialogue 모델 vs. Retrieval+ClaimDet 적용 모델[1]  |
| Table 2  | 표     | FEVER DEV vs. DialFact TEST 성능: 기존 VitC 적용 모델, dialogue fine-tuned, 제안기법 조합 모델 그룹별 비교[1] |

## 4. 실험 환경
- 프레임워크: PyTorch, HuggingFace Transformers  
- 기본 모델: RoBERTa-base (VitC fine-tuned)  
- 데이터: FEVER (stand-alone), DialFact (dialogue)  
- 하이퍼파라미터: default for retrieval·STS encoders, claim detector  
- 인프라: NVIDIA V100 GPU

## 5. 실험 결과
- Dialogue fine-tuning 모델: DialFact ↑12% vs. FEVER ↓12% (catastrophic forgetting)  
- 제안기법 적용 모델: DialFact +4% 향상 유지, FEVER 성능 회복 (기존 모델 성능 유지)  
- Retrieval Adaptation + Claim Detection 효과적: 별도 fine-tuning 없이 양쪽 검증 모두 달성[1].

## 6. 배경지식 및 핵심 용어
[[RetrievalAdaptation]]  
대화 컨텍스트와 claim을 결합해 문서 검색 시 claim 유사도에 가중치를 두는 기법.  

[[ClaimDetection]]  
긴 dialogue 발화를 sub-sentence로 분할 후 evidence와의 STS를 통해 검증 가능한 부분만 추출.  

[[SentenceRetrievalEnhancement]]  
문장 유사도(r_s)뿐 아니라 문서 전체 relevance(rD, R_D) 신호를 결합해 evidence 선택 정밀도 향상.  

[[CatastrophicForgetting]]  
새 데이터 학습 시 기존 태스크 능력 급격히 소실되는 현상(FEVER→DialFact fine-tuning 시 FEVER 성능 급락).  

[[DialogueFactChecking]]  
대화 내 발화(utterance)를 맥락 포함 증거와 대조해 SUPPORTS/REFUTES/NEI 분류하는 과제.

## 7. 관련자료 링크
https://aclanthology.org/2023.emnlp-main.993/  
https://arxiv.org/abs/2311.08195  

## 8. 논문의 기여 및 한계
- **기여**: 별도 fine-tuning 없이 단일 모델로 대화·독립문장 검증 모두 달성하는 retrieval·input 적응 기법 제안[1].  
- **한계**: 간접 질문·의도적 우회 발화 처리 미흡, 모델 카드를 통한 사회적 영향·제한 사항 상세 필요.

## 9. 추가
- 후속: indirect claim detection, multi-turn 사실 연쇄 추론, dynamic context weighting 연구.

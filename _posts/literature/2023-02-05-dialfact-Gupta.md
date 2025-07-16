---
title: "DialFact: A Benchmark for Fact-Checking in Dialogue"
authors: ["Prakhar Gupta", "Chien-Sheng Wu", "Wenhao Liu", "Caiming Xiong"]
conference: "ACL 2022"
arXivPubdate: "2021-10-15"
github: "https://github.com/salesforce/DialFact"
arxiv: "2110.08222"
tags: ["DialFact", "FactChecking", "DialogueSystem", "EvidenceRetrieval", "VerifiableClaim"]
description: "대화 도메인에서 사실 확인을 위한 첫 번째 벤치마크 데이터셋"
categories: literature
published: true
---

## 1. 문제 정의
- 기존 사실 검증 데이터셋 대부분은 **stand-alone** 정형화된 문장 대상.  
- **대화**에서의 발화는 **ellipsis**·**coreference**·**colloquialism** 많아 모호.  
- 해결 과제  
  - 대화 맥락 고려한 *Verifiable Claim Detection*  
  - Wikipedia 스니펫 기반 *Evidence Retrieval*  
  - SUPPORTS/REFUTES/NEI 분류 (*Claim Verification*)  

## 2. 제안 방법
- **핵심 아이디어**: 대화 특성 반영한 파이프라인  
  1. [[Verifiable Claim Detection]]: 발화가 검증 가능한지 구분  
  2. [[Evidence Retrieval]]: Wikipedia 문서·문장 추출  
  3. [[Claim Verification]]: 증거와 대화 맥락 기반 진위 판정  
- **구성 요소**  
  - *Document Retriever*: WikiAPI (entity linking 기반) vs DPR (dense retrieval)  
  - *Sentence Retriever*: BERT 기반 랭킹 모델, 대화 맥락 포함 입력  
  - *Verification Model*: Colloquial 데이터·약지도 학습 활용한 Aug-WoW  
- **차별점**  
  - 대화 맥락 활용한 retrieval/adaptation 기법  
  - 합성 데이터(negation, substitution, mask-and-fill, generation)로 약지도 학습  

## 3. Figures | Tables
- 대화 사실 검증 파이프라인: Verifiable → Evidence → Verification 단계 설명 (Figure 1)
- Aug-WoW 모델 혼동 행렬 (Figure 2)
- MTurk 라벨링 인터페이스 예시 (Figure 3)
- DIALFACT 통계 (학습·검증·테스트, 생성/작성, 레이블 분포) (Table 1)
- REFUTE 레이블 상위 bigram + LMI; 명백한 부정어 바이어스 없음 (Table 2)
- Verifiable Claim Detection 베이스라인 정확도·F1 (Table 3)
- Document recall: WikiAPI vs DPR; 맥락 활용 시 향상 (Table 4)
- Evidence sentence recall@5; 맥락 포함 시 더 높음 (Table 5)
- Claim Verification 성능 비교 (Oracle/Wiki/DPR 증거) (Table 6)
- Aug-WoW ablation: 맥락 제외·BERT-Large 영향 (Table 7)
- 생성 vs 작성 발화별 검증 성능 (Table 8)
- 예시 대화·발화·증거·모델 예측 (Table 9)
- 검증 세트 검증 성능 (Table 10)
- NEI-Personal 제거 후 3-way 성능 (Table 11)
- 2-way 분류(Supported vs Not-Supported) 성능 (Table 12)
- DIALFACT 샘플: 자동 생성 vs 인간 작성 발화 (Table 13)

## 4. 실험 환경
- **프레임워크**: PyTorch, HuggingFace Transformers  
- **데이터**: Wizard of Wikipedia 기반, 합성·인간 작성 발화 라벨링  
- **모델**: BERT-base/large, DPR, SpaCy, T5-base for mask-and-fill  
- **하이퍼파라미터**: learning rate 2e-5, 배치 크기 32, max seq len 256  
- **학습 환경**: NVIDIA V100 GPU

## 5. 실험 결과
- **Verifiable Claim Detection**: Lexical+DNLI 최고 정확도 82.8% (NEI F1 낮음) [Table 3]  
- **Document Retrieval**: Wiki-ctx recall 75.0% vs DPR-ctx 58.8% [Table 4]  
- **Evidence Selection**: recall@5 Wiki-ctx+Ret-ctx 75.4% [Table 5]  
- **Claim Verification**: Aug-WoW 최고—Oracle 69.2%/69.0%, Wiki 51.6%/51.3% [Table 6]  
  - 생성 발화(63.9%)보다 작성 발화(74.2%)에서 더 우수 [Table 8]  
  - BERT-Large ablation: Oracle 70.9%/70.9% vs Wiki 45.8% [Table 7]  

## 6. 배경지식 및 핵심 용어
[[Verifiable Claim Detection]]  
대화 발화가 **검증 가능한 factual 정보** 포함 여부 식별.  

[[Evidence Retrieval]]  
Wikipedia 문서 및 문장에서 **가장 관련 있는 증거** 추출.  

[[Claim Verification]]  
주어진 증거·대화 맥락으로 발화가 **SUPPORTED/REFUTED/NEI**인지 분류.  

[[Coreference]]  
여러 언급이 **동일 개체** 가리키는 현상. 대화 맥락 통해 해소 필요.  

[[Coreferential reasoning]]  
여러 mention 간 **참조 관계** 이해·추론 능력.  

## 7. 관련자료 링크
https://github.com/salesforce/DialFact  
https://aclanthology.org/2022.acl-long.263/  
https://arxiv.org/abs/2110.08222  

## 8. 논문의 기여 및 한계
- **기여**: 대화 특화 fact-checking 벤치마크·파이프라인 제안  
- **한계**: Retrieval 오류, 복합 추론 부족, 인간-머신 생성 간 성능 격차  

## 9. 추가
- **합성 데이터 품질** 향상 방안: 더 정교한 스타일 변환  
- **후속 연구**: multi-hop reasoning, cross-dialogue 장기 문맥 활용  

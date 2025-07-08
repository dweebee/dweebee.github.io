---
title: "DialFact: A Benchmark for Fact-Checking in Dialogue"
authors:
  - Prakhar Gupta
  - Chien-Sheng Wu
  - Wenhao Liu
  - Caiming Xiong
conference: ACL, 2022
arXiv 게시일: Oct 2021
github: https://github.com/salesforce/DialFact
tags:
  - fact-checking
  - dialogue
  - NLP
  - benchmark
  - dataset
description: 대화 기반 사실 검증을 위한 최초의 대규모 벤치마크 DialFact를 제안하며, 세부 서브태스크와 약점 보완을 위한 학습 기법을 제시

categories:
  - literature
published: true
---

## 1. 문제 정의
- 기존 fact-checking 모델(FEVER 등)은 기사 기반 데이터에 최적화되어 대화체에 적용 시 성능 저하
- 대화 응답은 구어체, 생략, 코어퍼런스 등으로 인해 검증 난이도가 높음
- 대화 기반 사실 검증을 위한 공개 벤치마크 부재

## 2. 제안 방법
- 대화 기반 사실 검증 벤치마크 DialFact 구축 (총 22,245개 응답, Wikipedia evidence와 매핑)
- 3가지 서브태스크 정의: Verifiable Claim Detection, Evidence Retrieval, Claim Verification
- 자동 생성 및 인간 작성 응답 혼합 구성
- 약한 감독 기반 클레임 생성 기법 사용: Negation, Entity Substitution, Mask-and-Fill, LM 기반 생성
- 각 응답에 대해 다단계 크라우드소싱을 통한 라벨링 및 evidence 수집

## 3. Figure / Table / 수학공식 설명
- **Figure 1**: 전체 파이프라인 구조 시각화 (대화 context → verifiability 판단 → evidence retrieval → claim verification)
- **Table 1**: 데이터 분포 (지원/반박/불충분 + 자동/인간 생성 비율)
- **Table 3~5**: 각 서브태스크별 baseline 모델 성능 비교
- 수학 공식은 포함되지 않음

## 4. 구현 및 실험 환경
- **프레임워크**: PyTorch, HuggingFace Transformers
- **데이터셋**: Wizard of Wikipedia 기반 클레임 변형 및 생성
- **전처리**: SpaCy NER, WordNet, Sense2Vec, BM25, GPT-2 perplexity filtering 등
- **모델 구조**: BERT, ALBERT, DPR, CorefBERT, KGAT 등 다양한 baseline 구성
- **하이퍼파라미터**: baseline별로 다르며 논문 내 상세 설정 일부 제공
- **학습 환경**: GPU 기반 학습 (세부 사양 미기재)

## 5. 실험 결과
- **Verifiability Detection**: Lexical+DNLI 조합 최고 성능 (Accuracy 82.8%)
- **Evidence Retrieval**: WikiAPI 기반 retrieval이 DPR보다 높은 recall@5 기록 (Wiki-ctx 75.4%)
- **Claim Verification**: Aug-WoW 모델이 기존 모델(DNLI, VitaminC, Colloquial 등) 대비 우수 성능 달성
- Ablation 실험 통해 context 사용 여부, evidence 품질, 생성 방식이 성능에 미치는 영향 분석
- 주요 오류 원인: 코어퍼런스 해석 실패, 불명확한 retrieval 결과

## 6. 배경지식 및 핵심 용어 설명
- **Verifiable Claim**: 외부 문서에 근거하여 검증 가능한 응답
- **NEI**: Not Enough Information — 증거가 불충분하여 판단 불가
- **DPR**: Dense Passage Retrieval — BERT 기반의 dual encoder retrieval
- **Coreference Resolution**: 지시어와 참조 대상 연결
- **Wizard-of-Wikipedia**: 위키 기반 대화형 지식 응답 데이터셋

## 7. 코드 및 자료 링크
- GitHub: [https://github.com/salesforce/DialFact](https://github.com/salesforce/DialFact)

## 8. 논문의 기여 및 한계
- **기여**:
  - 대화 기반 fact-checking을 위한 최초의 대규모 벤치마크 제공
  - 대화체에 특화된 증거 검색 및 검증 태스크 제안
  - 약한 감독 기반 생성 및 평가 프레임워크 제시

- **한계**:
  - evidence retrieval 품질이 전체 성능의 병목
  - 복잡한 대화 맥락 및 코어퍼런스 처리의 어려움
  - 멀티턴 대화 검증이나 멀티홉 reasoning 확장 미흡

## 9. 추가
- DialFact는 LLM 평가에 활용 가능한 테스트셋으로 확장 가능성 존재
- 후속 연구로 multilingual 대화 검증, **이미지 포함 multimodal 검증**

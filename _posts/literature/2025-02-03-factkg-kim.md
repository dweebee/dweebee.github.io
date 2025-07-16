---
title: "FactKG: Fact Verification via Reasoning on Knowledge Graphs"
authors: [Jiho Kim, Sungjin Park, Yeonsu Kwon, Yohan Jo, James Thorne, Edward Choi]
conference: ACL, 2023
arXivPubdate: 2023-05-11
github: https://github.com/jiho283/FactKG
arxiv: 2305.06590
tags: [FactChecking, KnowledgeGraphReasoning, MultiHopInference, ColloquialClaim, GraphEvidence]
description: "지식 그래프 기반 추론을 활용해 다양한 reasoning 유형(One-hop, Multi-hop, Conjunction, Existence, Negation)과 구어체·문어체 claim을 포함하는 대규모 fact verification 벤치마크 및 baseline 분석"
categories: literature
published: true
---

## 1. 문제 정의
- 기존 fact verification 연구는 주로 텍스트·테이블 등 비구조적 데이터에 집중, KG 활용 미흡
- KG는 신뢰성·추론 가능성 높으나, 자연어 claim과 KG 간 매핑·추론 경로 설계가 난제
- 목표: 다양한 reasoning 유형(One-hop, Multi-hop, Conjunction, Existence, Negation)과 구어체 claim을 포함하는 대규모 KG 기반 fact verification 벤치마크 제안

## 2. 제안 방법
- **FactKG**: DBpedia 기반 108,000개 자연어 claim, 각 claim은 SUPPORTED/REFUTED 라벨 및 관련 subgraph evidence 포함
- reasoning 유형별 claim 생성: One-hop(단일 triple), Conjunction(여러 triple 결합), Existence(존재성 주장), Multi-hop(연속 경로 필요), Negation(부정 포함)
- claim 다양성 확보: WebNLG 텍스트-그래프 쌍 변형, 구어체 변환(style transfer), presupposition template 활용
- baseline: claim only(BERT, BlueBERT, Flan-T5) vs. graph evidence 활용(GEAR 기반 subgraph retrieval+claim verification)

## 3. Figures | Tables
- Figure 1: DBpedia에서 triple 추출 → claim 진위 판단 구조 예시

- Table 1: 5가지 reasoning 유형별 claim 예시 (One-hop, Conjunction 등)

- Figure 2: 엔터티/관계 치환 통한 REFUTED claim 생성 및 NLI 기반 품질 관리
- Figure 3: 복합 reasoning claim 위한 그래프 패턴 시각화

- Table 2: reasoning 유형·스타일별 데이터 분포 통계

- Figure 4: 관계/경로 분류 기반 subgraph 추출 → claim 검증 파이프라인

- Table 3: reasoning 유형별 baseline 정확도 (claim only vs. evidence 기반)

- Table 4: 스타일 교차 학습 성능 비교 (문어체 ↔ 구어체 등)

## 4. 실험 환경
- 프레임워크: PyTorch, HuggingFace Transformers
- 모델: BERT, BlueBERT, Flan-T5, GEAR(Transformer encoder 기반)
- 데이터: DBpedia(2015-10), WebNLG(2020), claim–subgraph pair
- 하이퍼파라미터: BERT/BlueBERT fine-tune, Flan-T5 zero-shot, GEAR 구조 수정
- 환경: GPU(V100), claim–evidence split 8:1:1(train/dev/test)

## 5. 실험 결과
- **claim only**: BERT(65.2%), BlueBERT(59.9%), Flan-T5(62.7%)
- **with evidence(GEAR)**: 전체 평균 77.65%, Existence/Negation 유형에서 claim only 대비 15~20%p↑
- Multi-hop: BERT가 GEAR보다 우수(70.06% vs 68.84%)—복잡 추론 시 evidence retrieval 한계
- 스타일 교차(Written↔Colloquial): GEAR 성능 소폭 하락, BERT/BlueBERT는 일부 구어체 학습 시 오히려 일반화 성능↑

## 6. 배경지식 및 핵심 용어
[[Knowledge Graph (KG)]]  
엔티티(노드)와 관계(엣지)로 구성된 구조화된 지식 네트워크, 논리적 추론 경로 제공

[[One-hop Reasoning]]  
단일 triple(주어-관계-객체)만으로 claim 검증, 직접적 연결성 활용

[[Multi-hop Reasoning]]  
여러 triple을 연속적으로 따라가며 claim 검증, 경로 기반 복합 추론

[[Conjunction]]  
둘 이상의 triple이 모두 성립해야 claim이 SUPPORTED되는 논리적 결합

[[Existence Claim]]  
특정 엔티티가 어떤 관계를 갖는지(상대 엔티티 미지정) 주장

[[Negation]]  
not/never 등 부정 표현 포함 claim, 라벨 반전·복합 추론 요구

[[Colloquial Style Transfer]]  
문어체 claim을 구어체로 변환, 대화형 시스템 적용성·다양성 확보

[[Subgraph Retrieval]]  
claim 내 엔티티·관계 기반 KG에서 관련 서브그래프 추출, evidence로 활용

## 7. 관련자료 링크
https://github.com/jiho283/FactKG  
https://aclanthology.org/2023.acl-long.895/  
https://arxiv.org/abs/2305.06590  

## 8. 논문의 기여 및 한계
- 기여: reasoning 유형별·스타일별 대규모 KG 기반 fact verification 벤치마크, subgraph evidence 활용 효과 검증
- 한계: 최신 DBpedia 미반영, 단일 문장 claim 중심, multi-hop evidence retrieval 한계

## 9. 추가
- 후속 연구: 최신 KG 반영, 문단 단위 claim, multi-hop evidence retrieval 개선, medical QA 등 explainable KG 기반 응용 확장

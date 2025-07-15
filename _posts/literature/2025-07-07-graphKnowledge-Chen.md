---
title: "Improving Factuality for Dialogue Response Generation via Graph-Based Knowledge Augmentation"
authors: [Xiangyan Chen, Yujian Gan, Matthew Purver]
conference: NAACL, 2025
arXivPubdate: 2025-06-14
github: https://github.com/XiangyanChen/Triton
arxiv: 2506.12496
tags: [KnowledgeTripleRetriever, DialogueRewrite, KnowledgeAugmentedGeneration, FactScore, NEIP]
description: "대화 응답 생성의 factuality 향상을 위해 지식 그래프 기반 triple 검색·대화 재작성·지식 증강 생성 기법과 평가 지표 제안"
categories: literature
published: true
---

## 1. 문제 정의
- LLMs 대화 응답 생성 시 **hallucination**: 그럴싸하지만 사실과 불일치하는 텍스트 생성  
- 대화 특유의 **coreference** 구조, 문맥 의존성으로 기존 QA용 지식 증강 방식 적용 한계  
- 목표: 대화 맥락 고려한 지식 증강으로 응답의 factual consistency 보장

## 2. 제안 방법
- **Knowledge Triple Retriever**: query 핵심 엔티티 연관 triple 후보 수집 후, Llama3-8B+LoRA 기반 matcher로 **relevance** 정밀 분류  
- **Dialogue Rewrite**: GPT-4o CoT로 coreference 해소 예시 수집·Llama3-8B+LoRA fine-tuning으로 대화 재작성  
- **Knowledge-Augmented Generation**  
  - *Prompt-Based*: 재작성 대화와 상위 N개 triple을 템플릿 prompt에 삽입 후 LLM 생성  
  - *Graph-Based*: 선택된 triple로 구성한 subgraph를 GNN(LoRA fine-tuned)로 인코딩, 대화 임베딩과 결합해 디코더로 응답 생성

## 3. Figures | Tables
제안한 triple + rewrite 방식은 환각성 응답을 사실적 응답으로 전환하며, 이는 ‘rewrite → triple 검색 → 그래프·프롬프트 기반 생성’의 세 단계 파이프라인으로 구현된다. 자동 평가(OpendialKG·HybriDialogue의 BLEU·ROUGE‑L·PPL·fact score·NEIP·F1)와 인간 평가(일관성·유창성·정보성) 모두에서 기존 대비 성능이 유의미하게 향상됐고, 각 모듈을 제거하면 사실성 지표가 크게 떨어져 접근법의 기여도가 입증됐다.

## 4. 실험 환경
- **프레임워크**: PyTorch, HuggingFace Transformers  
- **모델**: Llama3-8B (retriever & rewrite & graph) + Flan-T5-XXL, ChatGLM-6B (generation)  
- **데이터**: OpendialKG, HybriDialogue  
- **하이퍼파라미터**: lr=1e-4, batch=32, epochs=3, LoRA rank=8  
- **인프라**: NVIDIA V100 GPU

## 5. 실험 결과
- **FactScore**: Triton-X 87.7%→73.0% (+ 2.9% over G-Retriever) on OpendialKG, 73.0%→66.6% on HybriDialogue  
- **NEIP**: Triton-X 59.2%→54.9% on OpendialKG, 54.9%→46.5% on HybriDialogue (낮을수록 검증 가능 비율↑)  
- **텍스트 품질**: BLEU-4·ROUGE-L·PPL 유지 혹은 소폭 향상  
- **Ablation**: retriever·rewrite·graph 각각 제거 시 fact score 1–3% 하락  
- **Human Eval**: Triton-X coherence/fluency/informativeness 전반적 개선

## 6. 배경지식 및 핵심 용어
[[Knowledge Triple Retriever]]  
대화 query 엔티티와 연관된 KG triples 중 **relevance** 높은 것을 LLM 기반 matcher로 선별하는 모듈.

[[Dialogue Rewrite]]  
LLM의 CoT 추론으로 대화 내 **coreference** 해소, entity linking 정확성 및 generation factuality 개선.

[[Knowledge-Augmented Generation]]  
Prompt-/Graph-based 두 방식으로 외부 지식 triple/subgraph를 대화 응답 생성에 통합하는 기법.

[[FactScore]]  
LLM과 외부 지식으로 분할된 atomic facts의 **precision** 기반 평가 지표(대화 context 추가, NEI label 확장).

[[NEIP (Not Enough Information Proportion)]]  
응답 내 **검증 불가 atomic facts** 비율, 낮을수록 factual consistency 우수.

## 7. 관련자료 링크
https://github.com/XiangyanChen/Triton  
https://arxiv.org/abs/2506.12496  

## 8. 논문의 기여 및 한계
- **기여**: triple retrieval·rewrite·generation 통합 프레임워크로 대화 factuality 대폭 개선  
- **한계**: entity overlap 기반 retrieval의 **semantic 한계**, KG 최신성 문제  

## 9. 추가
- **후속 연구**: 시맨틱 entity retrieval, dynamic KG 업데이트, multi-hop reasoning integration  
- **응용**: 실시간 QA 챗봇, domain-specific fact-checking 도구 개발  

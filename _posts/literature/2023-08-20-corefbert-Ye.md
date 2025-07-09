---
title: "Coreferential Reasoning Learning for Language Representation"
authors: ["Deming Ye", "Yankai Lin", "Jiaju Du", "Zhenghao Liu", "Peng Li", "Maosong Sun", "Zhiyuan Liu"]
conference: "EMNLP 2020"
arXivPubdate: "2020-04-15"
github: "https://github.com/thunlp/CorefBERT"
arxiv: "2004.06870"
tags: ["CorefBERT", "CoreferenceResolution", "MentionReferencePrediction", "LanguageRepresentation", "CopyMechanism"]
description: "coreference 정보를 명시적으로 처리하는 언어 표현 모델"
categories: literature
published: true
---

## 1. 문제 정의

**기존 접근 방식의 한계**  
- BERT 등 기존 모델, coreference 비처리  
- MLM, 지역 정보만 요구, 장거리 연결 부실  
- Coreference 필요 태스크 성능 미흡  

**핵심 문제**  
- 문맥 일관성 위한 coreference 처리 능력 학습  
- 대규모 비지도 말뭉치에서 coreferential reasoning 학습 방법 개발  
- 기존 언어 이해 능력 유지하며 coreference 정보 향상  

## 2. 제안 방법

**핵심 아이디어**  
- CorefBERT: coreferential relations 포착 모델  
- Mention Reference Prediction (MRP): 새 사전훈련 태스크  

**구성 요소**  
1. Deep bidirectional Transformer  
2. 병행 태스크: MRP + MLM  

**Mention Reference Masking**  
- 명사 mention으로 추출, 그룹화, 샘플링  
- 마스킹 비율 MLM:MRP = 4:1  

**Copy-based Objective**  
- 문맥에서 복사 예측, 전체 어휘 대상 아님  
- 시작/끝 토큰까지 적용  

**차별점**  
- 일반 텍스트 대상, task-agnostic  
- copy mechanism 활용 coreference 모델링  

## 3. Figures | Tables

| 구분       | 설명                              | 핵심 내용                                                                                 |
|----------|---------------------------------|------------------------------------------------------------------------------------------|
| Figure 1 | CorefBERT 훈련 과정 예시               | "Claire" 마스킹, MRP는 context 복사 후보, MLM은 vocabulary 후보 선택 과정 시각화                     |
| Table 1  | QUOREF 성능                        | CorefBERTBASE F1 72.96% (+4.4%), CorefBERTLARGE 76.89% (+2.9%)                              |
| Table 2  | MRQA 6개 데이터셋 성능             | BASE +1.7%, LARGE +1.0% 평균 F1 향상, NewsQA·HotpotQA 효과적                                  |
| Table 3  | DocRED 관계 추출                   | BASE 57.51% (+0.74%), LARGE 59.01% (+0.31%)                                                |
| Table 4  | FEVER 사실 검증                    | KGAT+CorefBERTBASE FEVER 69.82% (+0.42%), CorefRoBERTaLARGE 72.30% (+1.92%)               |
| Table 5  | Coreference resolution 태스크 결과 | GAP, DPR, WSC, Winogender, PDP에서 일관된 성능 향상                                         |
| Table 6  | GLUE 벤치마크                      | BERT과 유사 성능, 일반 언어 이해 능력 유지                                                   |
| Table 7  | Ablation study                    | NSP 제거, WWM/MRM 비교, copy-based 효과 분석. QUOREF +2.3% F1 향상                            |

## 4. 실험 환경

**도구 및 프레임워크**  
Hugging Face Transformers, spaCy, RTX 2080 Ti ×8, Mixed precision  

**데이터셋 및 전처리**  
사전훈련: Wikipedia 3,000M 토큰  
평가: QUOREF, MRQA, DocRED, FEVER, GAP, DPR, WSC, Winogender, PDP, GLUE  

**모델 및 하이퍼파라미터**  
- CorefBERTBASE: 110M 파라미터, 12 레이어, hidden size 768  
- CorefBERTLARGE: 340M 파라미터, 24 레이어, hidden size 1024  
- Adam optimizer, batch size 256  
- Learning rate: Base 5e-5, Large 1e-5  
- MRP:MLM 손실 비율 1:1  

**학습 환경**  
33k 스텝, 20% warmup 후 선형 감소  
Base 모델 1.5일, Large 모델 11일 소요  
RoBERTa 초기화 모델도 훈련  

## 5. 실험 결과

**주요 지표**  
- QUOREF: CorefBERTLARGE F1 76.89% (+2.9%)  
- MRQA: 평균 F1 +1.0–1.7%  
- DocRED: F1 +0.3–0.7%  
- FEVER: FEVER Score 최대 +1.9% (SOTA)  

**비교**  
- BERT 대비 coreference resolution 성능 일관된 향상  
- task-agnostic 모델의 일반성 입증  

**Ablation**  
- NSP 제거 시 성능 향상  
- MRM이 WWM과 동등 또는 우수  
- copy-based objective로 QUOREF +2.3% F1 추가 향상  

## 6. 배경지식 및 핵심 용어

[[Coreference]]  
텍스트 내에서 **두 개 이상의 표현이 동일한 개체를 지칭**하는 현상. 문장 경계를 넘어 장거리 관계를 포착해야 하는 핵심적 언어 이해 문제.

[[Coreferential reasoning]]  
여러 언급(mentions) 간의 **참조 관계를 이해하고 추론**하는 능력. 동일 개체에 대한 다양한 표현을 연결해 담화의 일관성을 유지.

[[Mention Reference Prediction (MRP)]]  
여러 번 등장하는 **명사 mention 중 하나를 마스킹(masking)**하고, 문맥 내 다른 mention에서 **복사(copy) 기반**으로 원래 토큰을 예측하는 새로운 사전훈련 태스크.

[[Copy-based training objective]]  
전체 어휘(vocabulary)에서 선택하는 대신, **문맥(context) 내의 토큰을 복사(copy)**하여 마스킹된 토큰을 예측하는 목표. context-sensitive한 표현 학습을 촉진.

[[Mention reference masking]]  
문장 내 **명사 mention을 그룹화(grouping)**한 뒤, 각 그룹에서 무작위로 하나를 선택해 마스킹(masking)하는 전략. MLM 대비 **coreference 정보 학습 비율을 높이기 위해** MRP:MLM = 1:4 비율로 적용.

[[Distant supervision assumption]]  
반복되는 mention들이 **서로 동일 개체를 참조한다는 가정**을 통해, 별도 라벨없이 coreference 관계를 자동 생성하는 지도(signal)로 활용.

[[Whole Word Masking (WWM)]]  
하나의 단어가 서브워드(subword) 여러 개로 분리될 때, **서브워드 전체를 함께 마스킹**하는 방식. mention masking과 대비되는 일반 MLM 전략.

[[Copy mechanism]]  
Sequence-to-sequence 모델의 **입력(input)에서 출력(output)으로 토큰을 직접 복사**하는 메커니즘. MRP의 학습 목표 설정에 영감 제공.

[[Extractive QA]]  
주어진 문단(paragraph)에서 정답(answer)이 되는 **스팬(span)만을 선택**해 응답을 추출하는 질의응답 방식.

[[Document-level relation extraction]]  
문서 전체(document)에서 **개체(entity) 간의 관계를 추출**하는 태스크로, 단일 문장을 넘어선 종합적 정보 처리 요구.

## 7. 관련자료 링크

- GitHub: https://github.com/thunlp/CorefBERT  
- arXiv: https://arxiv.org/abs/2004.06870  
- ACL Anthology: https://aclanthology.org/2020.emnlp-main.582/  
- Hugging Face Model Hub: coref-bert-base / coref-bert-large  

## 8. 기여 및 한계

**기여**  
- MRP 사전훈련 태스크 제안  
- copy mechanism 도입  
- 다양한 downstream 태스크 성능 향상  

**한계**  
- distant supervision 노이즈  
- 대명사 처리 한계  
- 복잡 추론 능력 부족  
- 높은 훈련 비용  

## 9. 추가

- 대명사 포함 모델링 강화  
- self-supervised 노이즈 완화 연구  
- 멀티모달 coreference 확장  
- 다국어 coreference 모델 개발  
- 실시간 대화 시스템 적용  

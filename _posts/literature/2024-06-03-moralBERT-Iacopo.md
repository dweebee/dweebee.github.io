---
title: "MoralBERT: Detecting Moral Values in Social Discourse"
authors:
  - Vjosa Preniqi
  - Iacopo Ghinassi
  - Kyriaki Kalimeri
  - Charalampos Saitis
conference: GoodIT ’24, 2024
arXivPubdate: 2024-03-12
github: https://github.com/vjosapreniqi/MoralBERT
arxiv: "2403.07678"
tags:
  - MoralFoundationsTheory
  - DomainAdversarialTraining
  - SingleLabelClassification
  - LibertyOppression
description: Moral Foundations Theory 기반 다양한 소셜미디어 데이터로 파인튜닝된 BERT 모델을 통해 사회적 담론 내 도덕적 가치를 탐지
categories: literature
published: true
---

## 1. 문제 정의
- 인간 언어에 내재된 감정·성격·도덕 가치 탐지 필요 (Graham et al., 2013).  
- 기존 도구(lexicon, Word2Vec, LLM 제로샷)는 문맥 민감성·플랫폼 간 일반화 한계.  
- 목표: 소셜미디어(트위터, 레딧, 페이스북) 발화에서 12개 MFT 도덕 축(virtue/vice) 단일-라벨 예측 및 도메인 간 일반화.

## 2. 제안 방법
- 핵심: **MoralBERT**(BERT-base-uncased) 파인튜닝 + **MoralBERT_adv** 도메인-적대적 학습.  
- 학습  
  1. aggregated fine-tuning: 세 플랫폼 합산 데이터  
  2. domain-adversarial: gradient reversal layer로 도메인 불변 표현 학습  
  3. class-weighting: King & Zeng(2001) 방식으로 불균형 처리  
- 분류: 각 도덕 축별 single-label 바이너리 분류 (presence/absence).

## 3. Figures | Tables
| 번호       | 유형     | 요약 설명                                                                                                       |
|----------|--------|--------------------------------------------------------------------------------------------------------------|
| Figure 1  | 그림     | 세 플랫폼 데이터 분포 UMAP 시각화: neutral 포함·제외 시 feature 클러스터링 차이                                    |
| Figure 2  | 그림     | in-domain F1 Binary/Macro 비교: MoralStrength, Word2Vec+RF, GPT-4, MoralBERT, MoralBERT_adv 성능 곡선              |
| Figure 3  | 그림     | out-of-domain F1 비교: 세 플랫폼 교차 검증 시 MoralBERT_adv 일관된 개선                                          |
| Figure 5  | 그림     | Liberty/Oppression in-domain vs out-of-domain 성능(F1) 비교                                                     |
| Table 1   | 표      | MFTC, MFRC, FB 도덕 레이블 분포(virtue/vice/neutral) 통계                                                   |
| Table 2   | 표      | in-domain F1(Binary/Macro): MoralBERT_adv 최고—F1 Binary +0.17 vs GPT-4, +0.32 vs Word2Vec[1][19]                |
| Table 3   | 표      | out-of-domain FB 테스트: GPT-4 vs MoralBERT vs MoralBERT_adv 유사 성능                                          |
| Table 4   | 표      | 예시 발화별 human/GPT-4/MoralBERT_adv 라벨 비교: informal·논리적 뉘앙스 반영 차이                                  |
| Table 5   | 표      | Liberty/Oppression in-domain 및 out-of-domain F1 세부 결과                                                     |

## 4. 실험 환경
- 프레임워크: PyTorch, Transformers (Devlin et al., 2018)  
- 데이터:  
  • MFTC(20,628 tweets)  
  • MFRC(13,995 Reddit)  
  • FB(1,509 posts; Cohen’s κ=0.32)  
- 전처리: URL·mention·hashtag·emoji 제거 및 대체, non-ASCII 제거  
- 하이퍼파라미터: lr=5e-5, batch=16, max_len=150, epoch=5  
- 최적화: Adam, class weighting  

## 5. 실험 결과
- In-domain  
  • MoralBERT_adv F1 Binary 0.45, F1 Macro 0.73; GPT-4 대비 +0.17/+0.11[19]  
- Out-of-domain  
  • FB 테스트: GPT-4·MoralBERT_‍adv 유사, 단 일부 foundation(Degradation, Loyalty, Authority) 개선  
- Liberty/Oppression  
  • In-domain: MoralBERT_adv F1 Binary 0.66, F1 Macro 0.71 vs GPT-4 0.24/0.48[19]  
  • FB→MFTC: MoralBERT_adv F1↑0.25/0.08 vs GPT-4  

## 6. 배경지식 및 핵심 용어
[[Moral Foundations Theory]]  
여섯 심리적 기반(Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Purity/Degradation, Liberty/Oppression)으로 도덕 판단 구조화(​Haidt & Graham, 2007; Haidt, 2012).

[[Domain-Adversarial Training]]  
gradient reversal layer로 도메인 분류 손실을 최대화·목표 손실 최소화해 **도메인 불변 표현** 학습(​Ganin & Lempitsky, 2015).

[[Single-Label Classification]]  
각 도덕 foundation별 별도 이진 분류; **multi-label 간 상호의존성** 회피해 성능↑.

[[Class Weighting]]  
불균형 데이터에서 각 클래스 가중치 조정해 **편향 완화**(King & Zeng, 2001).

[[Zero-Shot LLM]]  
사전학습된 대형언어모델(GPT-4) 제로샷 분류; prompting 기반 도덕 라벨 예측.

## 7. 관련자료 링크
https://github.com/vjosapreniqi/MoralBERT  
https://aclanthology.org/2024.goodit.263/  
https://arxiv.org/abs/2403.07678  

## 8. 논문의 기여 및 한계
- 기여: MFT 전 foundation 포함·single-label BERT 파인튜닝, domain-adv adversarial 학습, 대규모 소셜미디어 벤치마크 제시  
- 한계: English 전용, data imbalance, Liberty/Oppression 부분 자원 제한, out-of-domain 여전한 성능 제약  

## 9. 추가
- 후속: 다국어 모델·cross-cultural 도덕 분석, 음악·영화 등 타 도메인 적용, LLM→BERT 지식 증류를 통한 합성 데이터 활용  

---

title: "MoralBERT: A Fine‑Tuned Language Model for Capturing Moral Values in Social Discussions"
authors: [Vjosa Preniqi, Iacopo Ghinassi, Julia Ive, Charalampos Saitis, Kyriaki Kalimeri]
conference: International Conference on Information Technology for Social Good (GoodIT), 2024
arXivPubdate: 2024-03-12
github: https://github.com/vjosapreniqi/MoralBERT
arxiv: 2403.07678
tags: [MoralBERT, MoralFoundationsTheory, DomainAdversarial, SocialMedia, MoralClassification]
description: "MFT 기반 fine‑tuned BERT로 트위터·레딧·페이스북 속 도덕성 표현 분류, lexicon 대비 +11~32% F1 개선"

---

## 1. 문제 정의
- 소셜미디어 텍스트 속에서 Moral Foundations Theory(MFT) 기반 도덕적 가치 탐지
- 기존 lexicon‑기반 방식 및 Word2Vec/RF, zero‑shot GPT‑4보다 성능 부족

## 2. 제안 방법
- BERT‑base‑uncased 사전학습모델을 MFT 데이터로 fine‑tuning하여 MoralBERT 구성
- aggregated training vs. domain‑adversarial training 비교
- single‑label, multi‑label 분류 설정 실험

## 3. Figures
| 번호 | 설명 | 의의/논문 주장 |
|--|--|--|
| Fig2‑Table2 | 10개 도덕 기초 각 F1 (binary, macro) | in‑domain에서 기존모델 초과 성능 입증 :contentReference[oaicite:0]{index=0} |
| Table5 | Liberty/Oppression F1 비교 (GPT‑4 vs MoralBERT) | GPT‑4 보다 우수한 성능, domain‑advantage 확인 :contentReference[oaicite:1]{index=1} |

## 4. 실험 환경
- 프레임워크: BERT‑base‑uncased, Adam lr=5e‑5 :contentReference[oaicite:2]{index=2}
- 데이터셋: 트위터(MFTC, 35K tweets), 레딧(MFRC, 14K posts), 페이스북
- 도메인 적응: domain‑adversarial module 포함 
- 하드웨어: (언급 없음)

## 5. 실험 결과
- in‑domain aggregated training: lexicon‑기반 대비 F1 평균 +11~32% 향상 :contentReference[oaicite:3]{index=3}
- domain‑adversarial training: out‑of‑domain 예측에서 우수, zero‑shot GPT‑4 수준 도달 :contentReference[oaicite:4]{index=4}

## 6. 배경지식 및 핵심 용어
- **Moral Foundations Theory (MFT)**: 인간 도덕 인식은 Care/Harm, Fairness/Cheating, Loyalty/Betrayal, Authority/Subversion, Purity/Degradation, Liberty/Oppression 여섯 축으로 구성됨
- **domain‑adversarial training**: 도메인 특화 특성 제거하면서 일반화 목적 fine‑tuning 기법
- **aggregate training**: 트위터, 레딧, 페이스북의 모든 데이터 통합 후 학습
- **single‑/multi‑label 분류**: 하나의 도덕 가치만 예측 vs 여러 도덕 가치를 동시 예측
- **zero‑shot GPT‑4**: 추가 학습 없이 GPT‑4를 직접 분류기로 사용

## 7. 관련자료 링크
- GitHub 코드베이스: https://github.com/vjosapreniqi/MoralBERT :contentReference[oaicite:5]{index=5}

## 8. 논문의 기여 및 한계
- **기여**: MFT 기반 fine‑tuned BERT 모델로 도덕성 분류 정확도 크게 향상; 도메인 적응으로 플랫폼 간 일반화 성능 확보
- **한계**: out‑of‑domain 예측력 여전히 완전치 않음, 멀티라벨 설정에서 추가 연구 필요

## 9. 추가
- policy‑making, 소셜미디어 분석, 갈등 해소 분야에 적용 가능성 높음
- future work: 더 다양한 플랫폼·언어 적용, 비지도적 방법과 결합한 moral detection 탐구

## 10. 기타
- ethical considerations, limitations 섹션 포함 :contentReference[oaicite:6]{index=6}

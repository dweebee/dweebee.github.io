---
title: "DialFact: A Benchmark for Fact‑Checking in Dialogue"
authors: [Prakhar Gupta, Chien‑Sheng Wu, Wenhao Liu, Caiming Xiong]
conference: ACL, 2022
arXivPubdate: 2021‑10‑15
github: https://github.com/salesforce/DialFact
tags: [DialogueFactChecking, EvidenceRetrieval, ClaimVerification]
description: "대화 속 발화에 대한 사실 검증을 위한 벤치마크와 접근법 제시"
categories: literature
published: true
---

## 1. 문제 정의
- 기존 fact‑checking은 정형화된 단일 문장(claim)에 집중, 대화체 문장에서는 성능 저하
- 목표: 대화 속 발화에서 ‘verifiable claim detection’, ‘evidence retrieval’, ‘claim verification’ 수행

## 2. 제안 방법
- 핵심: **DIALFACT** 벤치마크 + 약식 학습 전략
- 대화 발화를 유형별로 분류하고, 위키피디아 기반 evidence retrieval 후, 해당 evidence로 claim을 증명
- FEVER 기반 모델 미사용 → 대화 특성 반영한 학습 방식 제안
- 약한 지도 신호(weak supervision) 활용: synthetic claim 생성 후 fine‑tune

## 3. Figures
| 번호 | 내용 | 의의 | 논문 주장과 연결 |
|-----|------|------|-----------------|
| Fig 1 | 데이터 파이프라인: synthetic claim 생성→ crowd annotation 프로세스 | benchmark의 신뢰도 확립 | 대화 맞춤 태스크 구성 |
| Table 2 | Det,Rtrv,Verify 세 과제 성능 비교 | FEVER-trained 모델의 한계 수치화 | 대화 특화 접근 필요성 강조 |
| Fig 3 | error analysis 케이스별 분포 | fill‑in, coreference, colloquial 표현 오류 빈도 분석 | future 연구 방향 제시 |

## 4. 실험 환경
- 도구: Python, PyTorch 기반 BERT/Transformer 활용
- 데이터: Wizard‑of‑Wikipedia 기반 human/machine 발화 22,245건, crowd‑annotated
- 데이터 전처리: synthetic claim via contradiction, substitution, infilling
- 모델: evidence retrieval용 dense retriever + classification head
- 학습: GPU (NVIDIA V100 등)

## 5. 실험 결과
- Verifiable Claim Detection F1 ≈ 0.85, Evidence Retrieval recall@5 ≈ 0.70, Claim Verification accuracy ≈ 0.65
- FEVER 기반 모델보다 모든 서브태스크에서 크게 향상
- 약한 지도(supervision) 추가 시 verify accuracy +5~7% 상승
- Ablation: synthetic data 없을 경우 성능 3~5% 감소

## 6. 배경지식 및 핵심 용어
- **Verifiable Claim Detection**: 발화가 사실 검증 대상인지 식별
- **Evidence Retrieval**: 관련 Wikipedia snippet을 검색
- **Claim Verification**: SUPPORTS / REFUTES / NOT_ENOUGH_INFORMATION 예측
- **weak supervision**: 합성 데이터로 미세 조정
- **coreference**, **colloquialisms**: 대화체 특징으로 성능 저하 이유

## 7. 관련자료 링크
- GitHub: https://github.com/salesforce/DialFact :contentReference[oaicite:0]{index=0}
- arXiv: 2021‑10‑15 :contentReference[oaicite:1]{index=1}

## 8. 논문의 기여 및 한계
- **기여**: 대화 발화 fact‑checking 벤치마크 구축 및 multi‑task framework 제시
- **실용성**: 챗봇, 정보 모니터링 시스템에 통합 가능
- **한계**: 위키피디아 기반 도메인 제한, crowd annotation 오차 존재
- 향후: 다양한 도메인 확장, 코어퍼런스 및 colloquial 특화 처리 모델 개발 필요

## 9. 추가
- 후속 연구: Chamoun et al.(2023) 등이 retrieval 조정 및 입력 변환 방식으로 성능 개선됨 :contentReference[oaicite:2]{index=2}
- 실제 상용 챗봇 시스템에 통합 시, real‑time weak supervision 활용 연구 가능

---
title: "DialFact: A Benchmark for Fact‑Checking in Dialogue"
authors: [Prakhar Gupta, Chien‑Sheng Wu, Wenhao Liu, Caiming Xiong]
conference: ACL, 2022
arXivPubdate: 15 Oct 2021
github: https://github.com/salesforce/DialFact
tags: [DialogueFactChecking, EvidenceRetrieval, NLP, Benchmark]
description: "대화 문맥에서 사실 검증을 위한 다단계 데이터셋 및 평가 벤치마크 제안"
categories: literature
published: true
---

## 1. 문제 정의
- 기존 사실검증은 주로 단일 문장(뉴스, 위키) 기반이며, 비공식·비정형 대화에서는 한계 존재
- 핵심 문제: 대화 속 비공식 응답에서 **Verifiable claim detection**, **Evidence retrieval**, **Claim verification** 수행

## 2. 제안 방법
- **DialFact** 데이터셋 구축: 22,245개의 대화 응답+위키 증거 쌍, 사람+모델 생성 발화 포함
- 파이프라인 구조:
  1. **Verifiable** vs **Non‑verifiable** 감지
  2. 위키에서 관련 문장 검색
  3. 응답이 **SUPPORTED**, **REFUTED**, **NOT ENOUGH INFO**인지 분류
- 기존 FEVER 기반 모델은 대화 도메인에 부적합. 논문은 weak supervision 활용한 학습 보강 제안 (부정 변환, entity swap 등)

## 3. Figures

| 번호 | 설명 | 의의 및 논문 주장 연결 |
|------|------|------------------------|
| Fig 1 | 대화 팩트체크 파이프라인 예시 | 전체 과제 흐름 시각화 |
| 데이터 통계 | 검증/증거/라벨 분포 | 데이터 규모 및 다양성 제시 |
| 성능 곡선 | baseline vs weak supervised | 제안 기법의 효과 입증 |

## 4. 실험 환경
- 환경: PyTorch 기반 코드 (GitHub 참고)
- 데이터: Wizard of Wikipedia 기반 human/model responses, crowd‑annotation
- 모델: FEVER 기반 retrieval+verification baseline, weak supervision 추가
- 하이퍼파라미터: 논문에 상세 기록
- 학습 리소스: GPU 사용

## 5. 실험 결과
- FEVER로 사전 학습한 baseline, 대화에서 성능 저조
- Weak supervised 학습 적용 시 3개 sub-task 모두 성능 유의미 향상
- 에러 분석: 대화 특유의 colloquialisms, coreference, ellipsis 처리 어려움 강조

## 6. 배경지식 및 핵심 용어
- **Verifiable claim detection**: 발화에 검증 가능한 사실 정보 포함 여부 판단
- **Evidence retrieval**: 위키에서 해당 사실을 뒷받침할 수 있는 문장 스니펫 선택
- **Claim verification**: 응답이 증거에 의해 SUPPORT, REFUTE, NEI로 분류
- **Weak supervision**: 데이터 부정, entity 교체 등 자동 변형 기법으로 모델 학습 신호 생성
- **Colloquialisms**, **Coreference**, **Ellipsis**: 대화문 특성으로 인한 retrieval/verification 장애 요인

## 7. 관련자료 링크
- GitHub: https://github.com/salesforce/DialFact

## 8. 논문의 기여 및 한계
- **기여**: 대화 도메인 전용 사실검증 벤치마크 제공, weak supervision 기법 제안
- **한계**: colloquial 표현, 생략/지시 표현 처리 어려움 존재. DIALFACT는 위키 기반. 특히 개인정보, 실시간 정보 검증에는 부적합

## 9. 추가
- 실제 챗봇 응답 품질 향상, 팩트체크 서비스 도입 등에 직접 활용 가능
- 후속연구: retrieval 정확도 향상, multi-hop 증거 연결, transformer 기반 end‑to‑end 설계 가능성 있음

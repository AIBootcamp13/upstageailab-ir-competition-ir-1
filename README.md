# Title (Please modify the title)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김문수](https://github.com/UpstageAILab)             |            [이상현](https://github.com/UpstageAILab)             |            [조선미](https://github.com/UpstageAILab)             |            [채병기](https://github.com/UpstageAILab)             |            [염창환](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- Python 3.10 이상
- Elasticsearch 8.8.0
- Upstage Solar LLM API
- 주요 라이브러리: pandas, matplotlib, seaborn, elasticsearch, sentence-transformers

### Requirements
- Elasticsearch 서버 동작 및 연결 환경 구축
- Upstage Solar API 키 발급 및 설정
- 필요한 Python 라이브러리 설치: `pip install -r requirements.txt`

## 1. Competiton Info

### Overview

- 본 프로젝트는 Upstage AI 기반의 RAG(Retrieval-Augmented Generation) 시스템을 개발하여 LLM 대회 평가에 참여하는 것을 목표로 합니다.
- Elasticsearch를 활용한 검색 인덱스 구축과 Upstage Solar LLM 모델 연동을 통해 질문에 적합한 답변 생성과 평가 점수 산출이 핵심입니다.

### Timeline

- 2025년 9월 08일 - 프로젝트 착수 및 데이터 준비
- 2025년 9월 17일 - 최초 모델 개발 및 내부 평가
- 2025년 9월 19일 - 최종 제출 및 대회 평가

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption


### Dataset overview

- `eval.jsonl`: 모델 평가를 위한 질문 데이터셋 (jsonl 형식)
- `documents.jsonl`: RAG 검색용 문서 데이터베이스
- 질문 및 문서 데이터는 한국어 과학 상식 위주로 구성됨

### EDA

- 데이터 내 질문 분포 및 문서 종류 분석
- 정답 문서 매칭 정보를 바탕으로 AP 점수 산출 준비

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling

### Model descrition

- Upstage Solar LLM 기반의 대규모 사전학습 언어 모델 활용
- Elasticsearch 8.8.0과 SentenceTransformer로 구성된 검색 엔진과 연동
- RAG 아키텍처를 참고하여 질문별 관련 문서 검색+응답 생성

### Modeling Process

- 질문 데이터를 입력으로, Elasticsearch에서 관련 문서 3개 검색
- 검색 결과를 포함해 LLM에게 최종 답변 생성 요청
- 생성된 답변과 검색 문서 ID를 제출용 파일로 저장
- 평가용으로 AP 점수를 산출하여 리포트 생성
- matplotlib과 seaborn를 활용해 AP 점수 분포 시각화

## 5. Result

### Leader Board
- 내부 평가 MAP 점수: 0.55 달성
- 제출 파일 `sample_submission_pro.csv` 정상 생성 완료


### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_

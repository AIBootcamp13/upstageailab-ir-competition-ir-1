# Korean Science RAG System (v3 Turbo Prompt)
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김문수](https://github.com/ashrate)             |            [이상현](https://github.com/UpstageAILab)             |            [조선미](https://github.com/LearnSphere-2025)             |            [채병기](https://github.com/avatar196kc)             |            [염창환](https://github.com/UpstageAILab)             |
|                    팀장 · 파이프라인 총괄                    |                 데이터 수집 및 관리                 |                  검색 모듈 개발                  |                 모델 서빙 및 리포트                 |              인프라 및 실험 관리              |

## 0. Overview
### Environment
- Python 3.10+
- Ubuntu 22.04 LTS, CUDA 12.x (선택 사항)
- Elasticsearch 8.8.0 (로컬 실행 또는 원격 클러스터)
- OpenAI API (gpt-3.5-turbo-1106)

### Requirements
```bash
pip install -r requirements.txt
```

**주요 의존성:**
- `elasticsearch`: ES 클라이언트 및 벌크 인덱싱
- `sentence-transformers`: KLUE RoBERTa 임베딩 모델
- `openai`: GPT-3.5 Turbo, 4o mini API 클라이언트
- `UPSTAGE`: solar api 클라이언트
- `python-dotenv`: 환경 변수 관리
  

**설정 파일:**
- `config/.env_v3_turbo_prompt`에 다음 정보 설정 필요:
  - `OPENAI_API_KEY`: OpenAI API 키
  - `ES_PASSWORD`: Elasticsearch 비밀번호
  - `ES_HOST`, `ES_PORT`: Elasticsearch 연결 정보

## 1. System Overview

### 시스템 구조

- **RAG (Retrieval Augmented Generation)** 기반 한국어 과학 상식 QA 시스템
- **검색 단계**: Elasticsearch Hybrid Retrieval (BM25 + Dense Vector)
- **생성 단계**: OpenAI GPT-3.5 Turbo를 활용한 함수 호출 기반 답변 생성
- **출력 형식**: JSONL/CSV 형태의 구조화된 답변 및 참조 문서

### 핵심 특징

- **RRF (Reciprocal Rank Fusion)** 기반 하이브리드 검색으로 BM25와 Dense 검색 결과 융합
- **Function Calling** 패턴으로 독립적인 검색 쿼리 생성 및 컨텍스트 기반 답변 생성
- **한국어 최적화**: Nori 형태소 분석기 및 KLUE RoBERTa 임베딩 활용
- **배치 처리**: 대량 평가 데이터셋에 대한 효율적인 일괄 처리

### Development Timeline

- 2024-04-01: 내부 베이스라인 설계 착수
- 2024-05-15: Elasticsearch 인덱스 및 하이브리드 검색 실험
- 2024-06-10: v3 Turbo Prompt 파이프라인 제출 버전 확정

## 2. System Architecture

### Core Components

**1. Config 클래스 (`v3_turbo_prompt.py:31-101`)**
- 환경 변수 기반 설정 관리
- Elasticsearch, OpenAI API, 경로 설정 통합 관리
- 실험명 자동 생성 (`model_search-type` 형식)

**2. IRBaselineSystem 클래스 (`v3_turbo_prompt.py:104-503`)**
- Elasticsearch 서버 관리 및 클라이언트 설정
- KLUE RoBERTa 임베딩 모델 초기화
- 하이브리드 검색 및 RAG 파이프라인 실행

**3. 검색 모듈**
- `sparse_retrieve()`: BM25 기반 키워드 검색
- `dense_retrieve()`: 벡터 유사도 기반 의미 검색
- `hybrid_retrieve()`: RRF 기반 하이브리드 검색 (alpha=0.9 기본값)

**4. 생성 모듈**
- `answer_question()`: Function Calling 기반 2단계 생성
  - 1단계: 독립적인 검색 쿼리 생성
  - 2단계: 검색 결과 기반 최종 답변 생성

### Directory Structure

```
├── v3_turbo_prompt.py                    # 메인 RAG 시스템
├── config/
│   └── .env_v3_turbo_prompt              # 환경 설정 (API 키, ES 정보)
├── data/
│   ├── documents.jsonl                   # 검색 문서 (docid, content)
│   └── eval.jsonl                        # 평가 질의 (eval_id, msg)
├── results/
│   └── submission_klue-roberta_hybrid.*  # 결과 파일 (JSONL/CSV)
└── requirements.txt                      # 의존성 패키지
```

## 3. Data Pipeline

### Dataset Structure

**입력 데이터:**
- `documents.jsonl`: 검색 대상 문서
  ```json
  {"docid": "doc_001", "content": "과학 지식 내용..."}
  ```
- `eval.jsonl`: 평가 질의
  ```json
  {"eval_id": "eval_001", "msg": [{"role": "user", "content": "질문 내용..."}]}
  ```

**출력 데이터:**
- `submission_klue-roberta_hybrid.jsonl`: 구조화된 답변
  ```json
  {
    "eval_id": "eval_001",
    "standalone_query": "검색 쿼리",
    "topk": ["doc_001", "doc_002", "doc_003"],
    "answer": "생성된 답변",
    "references": [{"score": 0.95, "content": "참조 문서..."}]
  }
  ```

### Processing Pipeline

**1. 문서 인덱싱 (`index_documents`)**
- 배치 단위 임베딩 생성 (기본 100개씩)
- Elasticsearch 벌크 인덱싱으로 문서 + 임베딩 저장
- Nori 형태소 분석기로 한국어 토크나이징

**2. 검색 처리 (`hybrid_retrieve`)**
- BM25와 Dense 검색을 병렬 수행 (각각 size*2 검색)
- RRF 점수 계산: `1/(k + rank)` (k=60)
- 가중 평균: `alpha * sparse_rrf + (1-alpha) * dense_rrf`

**3. 답변 생성 (`answer_question`)**
- Function Calling으로 독립적 검색 쿼리 생성
- Top-3 문서 검색 후 컨텍스트로 최종 답변 생성
- 검색 결과 및 참조 정보와 함께 구조화된 응답 반환

### Data Quality

- **임베딩 차원**: 1024차원 (KLUE RoBERTa Large)
- **검색 설정**: L2 정규화 벡터 유사도, top-3 문서 활용
- **하이브리드 비율**: BM25 90% + Dense 10% (alpha=0.9)
- **자동 변환**: JSONL 결과를 CSV로 자동 복사하여 리더보드 제출 지원

## 4. Model Architecture

### 검색 모델

**1. BM25 (Sparse Retrieval)**
- Elasticsearch 기본 BM25 알고리즘
- Nori 형태소 분석기로 한국어 키워드 매칭 최적화
- 불용어 필터링: E, J, SC, SE, SF, VCN, VCP, VX 품사 제거

**2. Dense Vector Search**
- **임베딩 모델**: `klue/roberta-large` (1024차원)
- **유사도 계산**: L2 정규화 기반 코사인 유사도
- **인덱스 설정**: `dense_vector` 필드, KNN 검색 지원

**3. Hybrid Retrieval (RRF)**
```python
# RRF 점수 계산
rrf_score = 1.0 / (k + rank)  # k=60
final_score = alpha * sparse_rrf + (1-alpha) * dense_rrf  # alpha=0.9
```

### 생성 모델

**GPT-3.5 Turbo Function Calling**
- **모델**: `gpt-3.5-turbo-1106`
- **온도**: 0 (결정적 출력)
- **시드**: 1 (재현 가능한 결과)

**2단계 생성 과정:**
1. **검색 쿼리 생성**: Function calling으로 독립적인 검색 쿼리 추출
2. **답변 생성**: 검색된 컨텍스트를 바탕으로 최종 답변 생성

### Training & Inference

**시스템 초기화:**
```bash
# 전체 파이프라인 실행
python v3_turbo_prompt.py

# 개별 단계 제어
python v3_turbo_prompt.py --no-es-server     # ES 서버 시작 건너뛰기
python v3_turbo_prompt.py --no-indexing      # 인덱싱 건너뛰기
python v3_turbo_prompt.py --eval-limit 50    # 평가 샘플 수 제한
```

**처리 단계:**
1. `setup_elasticsearch()`: ES 서버 시작 및 클라이언트 연결
2. `setup_models()`: KLUE RoBERTa 및 OpenAI 클라이언트 초기화
3. `setup_index()` + `index_documents()`: 인덱스 생성 및 문서 인덱싱
4. `eval_rag()`: 평가 데이터에 대한 배치 추론 및 결과 저장

### 성능 최적화

- **배치 처리**: 임베딩 생성 시 100개 단위 배치 처리
- **캐싱**: SentenceTransformer 모델 로딩 최적화
- **타임아웃**: Function calling 10초, QA 생성 30초 제한
- **병렬 검색**: BM25와 Dense 검색 동시 수행 후 융합

## 5. Usage & Results

### 실행 방법

**1. 환경 설정**
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (config/.env_v3_turbo_prompt)
OPENAI_API_KEY=your_openai_api_key
ES_PASSWORD=your_elasticsearch_password
ES_HOST=localhost
ES_PORT=9200
```

**2. 시스템 실행**
```bash
# 전체 파이프라인 실행 (ES 서버 시작 + 인덱싱 + 평가)
python v3_turbo_prompt.py

# 기존 인덱스 사용하여 평가만 실행
python v3_turbo_prompt.py --no-es-server --no-indexing

# 제한된 샘플로 테스트
python v3_turbo_prompt.py --eval-limit 10
```

### 결과 분석

**출력 파일:**
- `results/submission_klue-roberta_hybrid.jsonl`: 상세 결과 (참조 문서 포함)
- `results/submission_klue-roberta_hybrid.csv`: 리더보드 제출용

**성능 지표:**
- 검색 정확도: Top-3 문서 검색 기준
- 하이브리드 효과: BM25(90%) + Dense(10%) 조합
- 답변 품질: GPT-3.5 Turbo 기반 한국어 자연어 생성

**주요 특징:**
- RRF 기반 하이브리드 검색으로 키워드와 의미 검색의 상호 보완
- Function calling을 통한 구조화된 검색 쿼리 생성
- 한국어 특화 Nori 분석기 및 KLUE 임베딩 활용

### 시스템 모니터링

```python
# 검색 성능 테스트
system.test_retrieval("금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?")

# Elasticsearch 연결 상태 확인
system.es.info()

# 인덱스 문서 수 확인
system.es.count(index=config.es_index_name)
```

## 6. Technical Specifications

### 시스템 요구사항

- **메모리**: 최소 8GB RAM (KLUE RoBERTa 모델 로딩)
- **저장공간**: 최소 5GB (Elasticsearch 인덱스 + 모델)
- **네트워크**: OpenAI API 호출을 위한 인터넷 연결
- **Java**: Elasticsearch 8.8.0 실행을 위한 Java 11+

### 주요 제한사항

⚠️ **현재 구현의 한계:**
- 코드 13번째 줄 주석에서 언급한 대로, `answer_question()` 메서드에서는 실제로 `sparse_retrieve()`만 사용
- `hybrid_retrieve()` 메서드는 정의되어 있지만 실제 RAG 파이프라인에서는 활용되지 않음
- 진정한 하이브리드 검색을 위해서는 `answer_question()` 메서드 수정 필요

### 확장 가능성

- **다른 임베딩 모델**: 환경변수 `EMBEDDING_MODEL` 변경으로 손쉬운 모델 교체
- **검색 비율 조정**: `hybrid_retrieve()` 메서드의 alpha 파라미터 튜닝
- **LLM 모델 변경**: OpenAI 다른 모델 또는 로컬 LLM 연동 가능
- **평가 메트릭**: 자동 점수 계산 모듈 추가 개발 여지

## 7. References

### 핵심 기술 문서

- **Sentence-Transformers**: https://www.sbert.net/
- **Elasticsearch Dense Vector**: https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **KLUE RoBERTa**: https://huggingface.co/klue/roberta-large

### 관련 논문

- **RRF (Reciprocal Rank Fusion)**: "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
- **Dense Passage Retrieval**: "Dense Passage Retrieval for Open-Domain Question Answering"
- **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

### 프로젝트 관리

- **Meeting Log**: 노션 프로젝트 대시보드 예정 (링크 미정)
- **Issue Tracking**: GitHub Issues 활용 예정
- **Version Control**: Git 기반 협업 및 실험 관리

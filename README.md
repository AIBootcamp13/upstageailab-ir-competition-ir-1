# IR v2: Information Retrieval System
## Team

| [김문수](https://github.com/UpstageAILab) | [이상현](https://github.com/UpstageAILab) | [조선미](https://github.com/LearnSphere-2025) | [채병기](https://github.com/UpstageAILab) | [염창환](https://github.com/UpstageAILab) |
|:------:|:------:|:------:|:------:|:------:|
| 팀장, 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 | 담당 역할 |

## 0. Overview

# Information Retrieval Baseline System v2

한국어 과학 상식 질문답변을 위한 RAG (Retrieval Augmented Generation) 시스템

## 📋 개요

이 시스템은 Elasticsearch와 OpenAI GPT-3.5 turbo를 활용한 한국어 과학 상식 QA 시스템입니다. BM25와 Dense Vector 검색을 결합한 하이브리드 검색 방식을 통해 정확한 정보 검색과 답변 생성을 지원합니다.

### 주요 특징

- **하이브리드 검색**: BM25 희소 검색과 벡터 유사도 밀집 검색 결합
- **한국어 최적화**: KLUE RoBERTa 임베딩 모델과 Nori 한국어 분석기 사용
- **RRF 정규화**: Reciprocal Rank Fusion 기반 검색 결과 통합
- **Function Calling**: OpenAI Function Calling을 활용한 검색 쿼리 최적화

## 🏗️ 시스템 아키텍처

```
사용자 질문 → Function Calling → 검색 쿼리 생성 → 하이브리드 검색 → 답변 생성
                                    ↓
                            [BM25 검색 + Dense 검색]
                                    ↓
                                RRF 점수 결합
```

## 🚀 설치 및 설정

### 1. 필요 라이브러리 설치

```bash
pip install elasticsearch sentence-transformers openai python-dotenv
```

### 2. Elasticsearch 설치

```bash
# Elasticsearch 8.8.0 다운로드 및 설치
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.8.0-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.8.0-linux-x86_64.tar.gz
```

### 3. 환경 설정

`config/.env_v3_turbo_prompt` 파일 생성:

```env
# 모델 설정
EMBEDDING_MODEL=klue/roberta-large
LLM_MODEL=gpt-3.5-turbo-1106

# Elasticsearch 설정
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=ir_documents_klue
ES_CERT_PATH=./elasticsearch-8.8.0/config/certs/http_ca.crt
ES_BINARY_PATH=./elasticsearch-8.8.0/bin/elasticsearch

# 인증 정보
ES_USERNAME=elastic
ES_PASSWORD=your_elasticsearch_password
OPENAI_API_KEY=your_openai_api_key

# 데이터 경로
DOCUMENTS_PATH=./data/documents.jsonl
EVAL_PATH=./data/eval.jsonl

# 결과 경로
RESULTS_PATH=./results/submission_klue-roberta_hybrid.jsonl
```

## 📊 데이터 형식

### 문서 데이터 (documents.jsonl)
```json
{"docid": "doc_001", "content": "금성은 태양계에서 두 번째 행성으로..."}
```

### 평가 데이터 (eval.jsonl)
```json
{"eval_id": "eval_001", "msg": [{"role": "user", "content": "금성이 밝게 보이는 이유는?"}]}
```

## 🔧 사용법

### 기본 실행
```bash
python v3_turbo_prompt.py
```

### 고급 옵션
```bash
# Elasticsearch 서버가 이미 실행 중인 경우
python v3_turbo_prompt.py --no-es-server

# 인덱싱 건너뛰기 (이미 인덱싱된 경우)
python v3_turbo_prompt.py --no-indexing

# 평가만 실행
python v3_turbo_prompt.py --no-indexing --eval-limit 10

# 모든 단계 건너뛰고 평가만
python v3_turbo_prompt.py --no-es-server --no-indexing
```

## 🔍 검색 방법론

### 1. 하이브리드 검색
- **BM25 (희소 검색)**: 키워드 매칭 기반
- **Dense Vector (밀집 검색)**: 의미적 유사도 기반
- **가중치**: 기본값 α=0.9 (90% BM25, 10% Dense)

### 2. RRF 점수 계산
```python
def calculate_rrf_score(rank, k=60):
    return 1.0 / (k + rank)

final_score = α × sparse_rrf + (1-α) × dense_rrf
```

### 3. 한국어 분석
- **토크나이저**: Nori Korean Tokenizer
- **필터링**: 불용어 제거 (조사, 어미 등)
- **임베딩**: KLUE RoBERTa Large (1024차원)

## 📈 성능 최적화

### 검색 성능
- **배치 처리**: 100개 단위 임베딩 생성
- **인덱스 최적화**: L2 norm 유사도, dense_vector 인덱싱
- **결과 수**: 기본 top-3 검색 결과 활용

### 답변 생성
- **Function Calling**: 검색 쿼리 자동 최적화
- **컨텍스트 제한**: 상위 3개 문서만 활용
- **Temperature**: 0 (일관된 답변)

## 📝 출력 형식

### 평가 결과 (submission.jsonl)
```json
{
  "eval_id": "eval_001",
  "standalone_query": "금성 밝기 이유",
  "topk": ["doc_123", "doc_456", "doc_789"],
  "answer": "금성이 밝게 보이는 이유는...",
  "references": [
    {
      "score": 0.8521,
      "content": "금성은 태양계에서..."
    }
  ]
}
```

### CSV 변환
평가 완료 후 자동으로 `.csv` 파일 생성 (리더보드 제출용)

## ⚠️ 주의사항

1. **실제 구현**: 코드에서는 하이브리드 검색을 구현했지만, `answer_question` 메서드에서는 희소 검색만 사용됨
2. **메모리 사용**: KLUE RoBERTa Large 모델은 상당한 GPU 메모리 필요
3. **OpenAI API**: 사용량에 따른 비용 발생

## 🐛 문제 해결

### Elasticsearch 연결 오류
```bash
# 서버 상태 확인
curl -k -u elastic:password https://localhost:9200

# 인증서 경로 확인
ls -la ./elasticsearch-8.8.0/config/certs/
```

### 임베딩 모델 로딩 오류
```python
# GPU 메모리 부족 시 CPU 사용
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## 📚 참고자료

- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/)
- [Sentence Transformers](https://www.sbert.net/)
- [KLUE RoBERTa](https://huggingface.co/klue/roberta-large)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

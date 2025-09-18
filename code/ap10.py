import os
import json
import traceback
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- 1. 환경 변수 및 인증 정보 설정 ---
# 이 값들을 실제 환경에 맞게 정확히 설정해야 합니다.
ES_HOST = "https://localhost:9200"
ES_USERNAME = "elastic"             # 실제 Elasticsearch 사용자명
ES_PASSWORD = "PASSWORD"    # ★★★ 1. 실제 Elasticsearch 암호를 여기에 할당하세요! ★★★
ES_CA_CERT_FILE = "./elasticsearch-8.8.0/config/certs/http_ca.crt" # ★★★ 2. 실제 CA 인증서 파일 경로로 수정하세요! ★★★

UPSTAGE_API_KEY = "Upstage Solar API"    # 실제 Upstage Solar API 키
UPSTAGE_API_KEY = "your_upstage_api_key" # ★★★ 3. 실제 Upstage Solar API 키 값을 여기에 할당하세요! ★★★
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# --- 2. Elasticsearch 클라이언트 초기화 ---
try:
    # ES_CA_CERT_FILE 경로가 정확하지 않으면 연결 실패. (예: /etc/elasticsearch/certs/http_ca.crt)
    es = Elasticsearch(
        hosts=[ES_HOST],
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        ca_certs=ES_CA_CERT_FILE
    )
    es.info() # 연결 확인
    print("Elasticsearch 클라이언트 초기화 및 연결 성공.")
except Exception as e:
    print(f"Elasticsearch 연결 실패: {e}")
    if "TlsError" in str(e) and "No such file or directory" in str(e):
        print(f"오류 원인: '{ES_CA_CERT_FILE}' 경로에 CA 인증서 파일을 찾을 수 없습니다.")
        print("  - 해결 방법 (권장): 해당 파일을 찾아서 'ES_CA_CERT_FILE' 변수를 올바르게 수정해주세요.")
        print("  - 해결 방법 (개발/테스트용, 보안 위험): 'ca_certs=ES_CA_CERT_FILE' 라인을 주석 처리하고 'verify_certs=False' 옵션을 사용해보세요.")
    exit(1)

# --- 3. SentenceTransformer 모델 초기화 ---
# 한국어 임베딩 모델 (필요시 도메인 특화 모델로 교체 가능)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
print("SentenceTransformer 모델 초기화 완료.")

def get_embedding(sentences):
    """문장 또는 문서 리스트에 대한 임베딩 생성"""
    return model.encode(sentences)

# --- 4. Elasticsearch 인덱스 관리 및 검색 함수 ---
# 하이브리드 검색을 위해서는 Elasticsearch 'test' 인덱스에 'embeddings' 벡터 필드가 존재해야 합니다.
def sparse_retrieve(query_str, size=3):
    """BM25 기반 희소(sparse) 검색"""
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index="test", query=query, size=size, sort="_score")

def dense_retrieve(query_str, size=3):
    """벡터 임베딩 기반 밀집(dense) 검색 (KNN 검색)"""
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings", # 이 필드가 Elasticsearch 인덱스에 존재해야 합니다.
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100 # 검색 효율성 고려
    }
    return es.search(index="test", knn=knn)

# --- ★★★ 핵심 개선 1: 하이브리드 검색 함수 도입 (RRF 기반) ★★★
def hybrid_retrieve(query_str: str, size: int = 3, alpha: float = 0.5) -> dict:
    """
    RRF (Reciprocal Rank Fusion) 기반 하이브리드 검색.
    sparse_retrieve (BM25)와 dense_retrieve (벡터)의 결과를 융합합니다.
    Args:
        query_str: 검색 쿼리 문자열
        size: 최종적으로 반환할 문서의 개수
        alpha: sparse 검색 결과의 가중치 (0.0: dense만, 1.0: sparse만)
               일반적으로 0.5~0.8 사이의 값을 사용합니다.
    """
    initial_search_size = size * 5 # 각 검색에서 더 많은 문서를 가져와 RRF가 효과적
    sparse_results = sparse_retrieve(query_str, initial_search_size)
    dense_results = dense_retrieve(query_str, initial_search_size)

    # RRF 점수 계산 함수 (k 값은 일반적으로 60으로 설정)
    def calculate_rrf_score(rank, k=60):
        return 1.0 / (k + rank)

    combined_scores = {} # {'_id': {'sparse_rrf': score, 'dense_rrf': score, 'doc': hit_data}}

    # Sparse 결과의 RRF 점수 계산 및 저장
    for rank, hit in enumerate(sparse_results['hits']['hits'], 1):
        doc_id = hit['_id']
        combined_scores[doc_id] = {
            'sparse_rrf': calculate_rrf_score(rank),
            'dense_rrf': 0.0, # 초기값
            'doc': hit # 원본 문서 정보 저장
        }

    # Dense 결과의 RRF 점수 계산 및 기존 결과와 병합
    for rank, hit in enumerate(dense_results['hits']['hits'], 1):
        doc_id = hit['_id']
        if doc_id in combined_scores:
            combined_scores[doc_id]['dense_rrf'] = calculate_rrf_score(rank)
        else:
            combined_scores[doc_id] = {
                'sparse_rrf': 0.0, # 초기값
                'dense_rrf': calculate_rrf_score(rank),
                'doc': hit
            }
    
    # 가중치를 적용한 최종 RRF 점수 계산
    for doc_id, scores in combined_scores.items():
        # alpha를 sparse 검색에 대한 가중치로 사용
        scores['final_score'] = (alpha * scores['sparse_rrf']) + ((1 - alpha) * scores['dense_rrf'])

    # 최종 점수 기준으로 정렬 및 상위 size개 문서 선택
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)

    # Elasticsearch 검색 결과와 유사한 형식으로 구성
    hybrid_hits = []
    for doc_id, scores in sorted_results[:size]:
        hit = scores['doc'].copy()
        hit['_score'] = scores['final_score']
        hybrid_hits.append(hit)
    
    return {
        'hits': {
            'hits': hybrid_hits,
            'total': {'value': len(hybrid_hits)}
        }
    }


# --- 5. Upstage Solar OpenAI 클라이언트 초기화 ---
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1")
llm_model = "solar-pro2"
print("Upstage Solar LLM 클라이언트 초기화 완료.")

# --- ★★★ 핵심 개선 2: 프롬프트 엔지니어링 강화 ★★★
persona_function_calling = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 질문 내용을 명확히 파악하여 검색에 가장 적합한 단독 검색 쿼리를 생성합니다.
- 검색 쿼리는 한국어로 작성하며, 핵심 키워드를 포함하고 모호함을 없애는 방향으로 구체적으로 작성해야 합니다.
- 만약 질문이 과학 상식과 무관한 일반적인 대화인 경우, 검색 API 호출 없이 직접 답변을 생성합니다. (예: "안녕", "고마워", "너는 누구니?")
"""

persona_qa = """
## Role: 과학 상식 전문가
## Instructions
- 사용자의 질문과 제공된 참고 자료(검색 결과)를 신중하게 분석하여 답변을 생성합니다.
- 답변은 오직 제공된 참고 자료의 정보만을 바탕으로 해야 하며, 참고 자료에 없는 내용은 절대 추측하거나 추가하지 마십시오.
- 답변은 200자 이내의 간결하고 정확한 문장으로 구성합니다.
- 답변은 항상 존대하는 한국어로 작성하고, 필요시 전문 용어에 대한 간략한 설명을 덧붙일 수 있습니다.
- 만약 제공된 참고 자료만으로는 질문에 명확하게 답변하기 어렵거나 관련이 없는 경우, "제공된 정보만으로는 답변하기 어렵습니다."라고 명시하고 다른 질문을 요청하십시오.
"""

tools = [
     {
         "type": "function",
         "function": {
             "name": "search",
             "description": "search relevant documents",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "standalone_query": {
                         "type": "string",
                         "description": "Final query suitable for use in search"
                     }
                 },
                 "required": ["standalone_query"]
             }
         }
     }
]

# --- 6. RAG 질의 처리 함수 ---
def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    msg = [{"role": "system", "content": persona_function_calling}] + messages

    # Function Calling LLM 호출 (검색 필요 여부 및 검색 쿼리 생성)
    start_time_func_call = time.time()
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            temperature=0,
            seed=1,
            timeout=30
        )
        print(f"  [LLM FuncCall] 소요 시간: {time.time()-start_time_func_call:.2f}초")
    except Exception as e:
        print(f"  [LLM FuncCall] 호출 실패: {e}")
        traceback.print_exc()
        response["answer"] = "API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요." # 오류 메시지 개선
        return response

    # 검색이 필요한 경우 (tool_calls가 존재)
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")
        response["standalone_query"] = standalone_query

        print(f"  [RAG Step] 생성된 독립 검색 쿼리: '{standalone_query}'")
        
        # --- ★★★ 핵심 개선 1 적용: 하이브리드 검색 호출 ★★★
        # alpha=0.7은 BM25에 70%, Dense에 30% 가중치를 부여합니다. 필요에 따라 조정하세요.
        search_result = hybrid_retrieve(standalone_query, 3, alpha=0.7) 
        print(f"  [RAG Step] 하이브리드 검색 완료. 상위 문서 개수: {len(search_result['hits']['hits'])}")

        retrieved_context = []
        for rst in search_result['hits']['hits']:
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({
                "score": rst["_score"],
                "content": rst["_source"]["content"]
            })
        
        # LLM에게 전달할 검색된 컨텍스트 (JSON 형식, ensure_ascii=False로 한글 인코딩 방지)
        content_for_llm = json.dumps(retrieved_context, ensure_ascii=False, indent=2) # 가독성 향상
        messages.append({"role": "assistant", "content": content_for_llm})

        # QA Generation LLM 호출 (검색 결과 기반 답변 생성)
        start_time_qa_call = time.time()
        try:
            # messages 리스트가 계속 쌓이므로, 매 호출마다 persona_qa 프롬프트와 현재까지의 메시지를 조합
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": persona_qa}] + messages,
                temperature=0,
                seed=1,
                timeout=30
            )
            print(f"  [LLM QA Gen] 소요 시간: {time.time()-start_time_qa_call:.2f}초")
            response["answer"] = qaresult.choices[0].message.content
        except Exception as e:
            print(f"  [LLM QA Gen] 호출 실패: {e}")
            traceback.print_exc()
            response["answer"] = "검색 결과를 바탕으로 답변 생성 중 오류가 발생했습니다."
            return response
    else: # 검색이 필요 없는 경우 (function_calling LLM이 직접 답변)
        print(f"  [LLM FuncCall] 검색 API 호출 없이 직접 답변 생성.")
        response["answer"] = result.choices[0].message.content
    
    # 추가 개선점: 답변 후처리 (예: 스펠링 체크, 불필요한 문장 제거 등)
    # def post_process_answer(answer_text):
    #     # 여기에서 후처리 로직을 구현
    #     return answer_text
    # response["answer"] = post_process_answer(response["answer"]) 
    return response

# --- 7. AP 점수 계산 함수 ---
# (이전 코드와 동일, 비과학 상식 질문 처리 로직 포함)
def calculate_average_precision(eval_id, retrieved_topk, ground_truth_data=None):
    key = str(eval_id)
    if not ground_truth_data or key not in ground_truth_data:
        # 과학 상식 질문이 아니면 topk 있으면 0점, 없으면 1점 처리
        return 0.0 if retrieved_topk else 1.0
    relevant = ground_truth_data[key]
    hit = 0
    precision_sum = 0.0

    # 상위 3개 문서만 고려하여 AP 계산 (대회 규칙에 따름)
    for i, docid in enumerate(retrieved_topk[:3]):
        if docid in relevant:
            hit += 1
            precision_sum += hit / (i+1)
    return precision_sum / hit if hit > 0 else 0.0

# --- 8. 평가, 제출, 보고서, 그래프 생성 함수 ---
# (이전 코드와 동일, 진행률 및 로깅 강화)
def process_evaluation_and_generate_reports(
    eval_jsonl_path,
    submission_output_path,
    eval_report_excel_path=None,
    eval_report_csv_path=None,
    ground_truth_data=None
):
    if not os.path.exists(eval_jsonl_path):
        print(f"오류: 평가 데이터 파일이 없습니다: {eval_jsonl_path}")
        return

    # 총 라인 수 계산 (진행률 표시용)
    total_lines = sum(1 for _ in open(eval_jsonl_path, "r", encoding="utf-8"))
    all_results = []

    print(f"'{eval_jsonl_path}' 파일을 읽고 평가를 시작합니다...")
    
    with open(eval_jsonl_path, "r", encoding="utf-8") as fin, \
         open(submission_output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, 1):
            entry = json.loads(line)
            eval_id = entry["eval_id"]
            messages = entry["msg"]

            print(f"\n--- [ {idx}/{total_lines} ] eval_id={eval_id} 질문 처리 시작 ---")
            print(f"  원본 질문: '{messages[-1]['content']}'")
            
            # RAG 모델 호출
            response = answer_question(messages)
            
            # 제출 파일에 기록할 항목
            submission_entry = {
                "eval_id": eval_id,
                "standalone_query": response.get("standalone_query", ""),
                "topk": response.get("topk", []),
                "answer": response.get("answer", "")
            }
            fout.write(json.dumps(submission_entry, ensure_ascii=False) + "\n")
            
            print(f"  최종 응답: '{response['answer']}'")

            # AP 점수 계산
            ap = calculate_average_precision(eval_id, response.get("topk", []), ground_truth_data)
            print(f"  AP 점수: {ap:.4f}")

            all_results.append({
                "eval_id": eval_id,
                "question": messages[-1]["content"],
                "average_precision": ap,
                "standalone_query": response.get("standalone_query", ""),
                "topk": response.get("topk", []),
                "answer": response.get("answer", "")
            })

    print(f"\n제출 파일 '{submission_output_path}' 생성이 완료되었습니다.")

    df = pd.DataFrame(all_results)
    map_score = df["average_precision"].mean()
    print(f"\n총 {len(df)}개 질문에 대한 평가 완료, 전체 MAP 점수: {map_score:.4f}")

    if eval_report_excel_path:
        df.to_excel(eval_report_excel_path, index=False)
        print(f"평가 보고서 엑셀 파일 '{eval_report_excel_path}' 저장 완료.")
    if eval_report_csv_path:
        df.to_csv(eval_report_csv_path, index=False, encoding="utf-8-sig")
        print(f"평가 보고서 CSV 파일 '{eval_report_csv_path}' 저장 완료.")

    # 한글 폰트 설정
    font_name = None
    for fpath in fm.findSystemFonts(fontext="ttf"):
        p = fm.FontProperties(fname=fpath)
        if "NanumGothic" in p.get_name():
            font_name = p.get_name()
            break
    if font_name:
        plt.rcParams["font.family"] = font_name
    else:
        print("경고: 한글 폰트를 찾지 못해 그래프 한글이 깨질 수 있습니다.")
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["average_precision"], bins=10, kde=True, ax=ax[0])
    ax[0].set_title("Average Precision (AP) 점수 분포")
    ax[0].set_xlabel("AP 점수")
    ax[0].set_ylabel("질문 개수")

    sns.boxplot(y=df["average_precision"], ax=ax[1])
    ax[1].set_title("Average Precision (AP) 점수 요약")
    ax[1].set_ylabel("AP 점수")

    plt.tight_layout()
    graph_path = "./ap_scores_distribution.png"
    plt.savefig(graph_path)
    print(f"AP 점수 분포 그래프가 '{graph_path}'에 저장되었습니다.")


# --- ★★★ 9. 메인 실행 블록 및 인덱스 생성/문서 색인 통합 ★★★
if __name__ == "__main__":
    # --- 파일 경로 설정 ---
    EVAL_JSONL_PATH = "/home/0911_ir/data/eval.jsonl" # ★★★ 4. 평가할 질문 데이터 파일 경로 (필수) ★★★
    DOCUMENTS_JSONL_PATH = "/home/0911_ir/data/documents.jsonl" # ★★★ 5. 원본 문서 데이터 파일 경로 (필수) ★★★

    SUBMISSION_OUTPUT_PATH = "sample_submission_pro.csv" 
    EVAL_REPORT_EXCEL_PATH = "./evaluation_report.xlsx"
    EVAL_REPORT_CSV_PATH = "./evaluation_report.csv"

    # --- Elasticsearch 인덱스 준비 (문서 색인 과정 포함) ---
    INDEX_NAME = "test" # 사용할 인덱스 이름
    INDEX_SETTINGS = {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
    # 'embeddings' 필드를 dense_vector 타입으로 정의하는 매핑.
    # SentenceTransformer 모델의 임베딩 차원(dims)을 명시해야 합니다.
    INDEX_MAPPINGS = {
        "properties": {
            "docid": {"type": "keyword"},
            "content": {"type": "text"},
            "embeddings": {
                "type": "dense_vector",
                "dims": model.get_sentence_embedding_dimension() 
            }
        }
    }

    print("\n--- Elasticsearch 인덱스 준비 시작 (필요시 문서 재색인) ---")
    try:
        # 1. 인덱스 삭제 (재생성을 위해)
        if es.indices.exists(index=INDEX_NAME):
            es.indices.delete(index=INDEX_NAME)
            print(f"기존 인덱스 '{INDEX_NAME}' 삭제 완료.")
        
        # 2. 인덱스 생성 (embeddings 필드 매핑 포함)
        es.indices.create(index=INDEX_NAME, settings=INDEX_SETTINGS, mappings=INDEX_MAPPINGS)
        print(f"인덱스 '{INDEX_NAME}' 생성 완료 (embeddings 필드 포함).")

        # 3. 원본 문서 로드 및 임베딩 생성
        def load_documents(path):
            docs = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    docs.append(json.loads(line))
            return docs
        
        raw_documents = load_documents(DOCUMENTS_JSONL_PATH)
        print(f"총 {len(raw_documents)}개의 원본 문서 로드 완료.")

        document_contents = [doc["content"] for doc in raw_documents]
        # 대량 임베딩 시 show_progress_bar=True로 진행 상황 확인
        document_embeddings = model.encode(document_contents, show_progress_bar=True)
        print("모든 문서 임베딩 생성 완료.")

        # 4. 문서 재색인 (docid, content, embeddings 포함)
        documents_for_indexing = []
        for i, doc in enumerate(raw_documents):
            documents_for_indexing.append({
                "docid": doc["docid"],
                "content": doc["content"],
                "embeddings": document_embeddings[i].tolist() # NumPy 배열을 리스트로 변환
            })
        
        # bulk_add 함수를 사용하여 한 번에 색인
        success, failed = helpers.bulk(es, documents_for_indexing, index=INDEX_NAME)
        print(f"총 {success}개 문서 색인 성공, {len(failed)}개 문서 색인 실패.")
        es.indices.refresh(index=INDEX_NAME) # 색인 완료 후 refresh
        print(f"인덱스 '{INDEX_NAME}'에 문서 색인 완료. 총 문서 수: {es.count(index=INDEX_NAME)['count']}")

    except Exception as e:
        print(f"\nElasticsearch 인덱스 준비 또는 문서 색인 중 치명적인 오류 발생: {e}")
        traceback.print_exc()
        exit(1) # 인덱싱 실패 시 평가 진행 불가능하므로 종료
    
    print("\n--- Elasticsearch 인덱스 준비 완료 ---")


    # --- 평가용 과학 상식 정답 데이터 (Ground Truth) ---
    # ★★★ 6. 이 딕셔너리를 모든 과학 상식 질문에 대해 정확한 정답 문서 ID 리스트로 채워 넣어야 합니다! ★★★
    # eval.jsonl을 검토하여 각 과학 상식 질문의 eval_id와 해당 질문의 정답 문서 ID 리스트를 수동으로 추가하세요.
    # 비과학 상식 질문은 이 딕셔너리에 포함시키지 않습니다. (calculate_average_precision 함수가 자동으로 처리)
    ground_truth_for_evaluation = {
        "223": ["doc_physics_force_motion", "doc_newton_laws", "doc_force_and_acceleration"],
        "303": ["doc_animal_camouflage", "doc_evolutionary_adaptations", "doc_polar_bear_fur_science"],
        "286": ["doc_chemistry_methane_combustion", "doc_chemical_reaction_basics", "doc_oxidation_reduction"],
        "266": ["doc_fruit_cultivation_peach", "doc_gardening_tips_for_peach", "doc_peach_disease_prevention"],
        # 여기에 'eval.jsonl'에 있는 모든 과학 상식 질문에 대한 eval_id와
        # 해당 질문의 정답 문서 ID 리스트를 정확하게 추가하세요.
        # 이 부분이 실제 MAP 점수와 밀접하게 관련됩니다.
    }

    # --- 평가 프로세스 실행 ---
    process_evaluation_and_generate_reports(
        eval_jsonl_path=EVAL_JSONL_PATH,
        submission_output_path=SUBMISSION_OUTPUT_PATH,
        eval_report_excel_path=EVAL_REPORT_EXCEL_PATH,
        eval_report_csv_path=EVAL_REPORT_CSV_PATH,
        ground_truth_data=ground_truth_for_evaluation
    )

import os
import sys

# --- 重要：在导入 ragas 之前设置环境变量，避免其后台 analytics 线程/网络调用干扰本地检索 ---
# 关闭 Ragas usage tracking（否则会启动后台线程；在部分 Windows + Rust 扩展组合下可能触发崩溃）
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
# 限制并发线程，降低与 chromadb-rust 的潜在冲突概率
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 强制离线/无 chroma：评估阶段不要触发 chromadb rust 查询
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pandas as pd
from datasets import Dataset
import numpy as np
import json
import hashlib
from typing import Optional
import requests

# 注意：不要在模块加载时 import ragas（它会启动后台线程）。
# 我们会在完成本地检索（收集 contexts）之后，再在函数内部导入 ragas 并执行评估。

# 将项目根目录添加到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import Config
from src.data_processing.document_processor import LegalDocumentProcessor
from src.rag_system.rag_engine import LegalRAGSystem

def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _build_chunk_corpus(config: Config):
    """
    从 Database 文档构建 chunks 语料库（用于评估检索）。
    返回: chunks(list[str]), metas(list[dict])
    """
    processor = LegalDocumentProcessor(config.DATABASE_PATH)
    docs = processor.load_documents()
    chunks = []
    metas = []
    for d in docs:
        for idx, c in enumerate(processor.split_document(d["content"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            c = (c or "").strip()
            if not c:
                continue
            chunks.append(c)
            metas.append({"law_name": d.get("law_name"), "source": d.get("source"), "chunk_id": idx})
    return chunks, metas

def _embed_chunks(embedding_model, chunks: list[str], batch_size: int = 32):
    """对 chunks 做 embedding，并返回 numpy array (N, D)"""
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vecs = embedding_model.embed_documents(batch)
        vectors.extend(vecs)
    return np.asarray(vectors, dtype=np.float32)

def _retrieve_topk(embedding_model, chunk_vectors: np.ndarray, chunks: list[str], question: str, top_k: int = 4):
    """用 embedding 做 Top-K 召回（BGE 已归一化则 dot≈cosine）"""
    qv = np.asarray(embedding_model.embed_query(question), dtype=np.float32)
    scores = chunk_vectors @ qv
    top_idx = np.argsort(-scores)[:top_k]
    return [chunks[i] for i in top_idx]

def _generate_answer_llm(config: Config, question: str, contexts: list[str]) -> str:
    """不用 Chroma 的生成: 将 contexts + question 交给 LLM 生成答案 (通过 OpenAI 兼容接口)"""
    # 默认在评估脚本中禁用代理（仅影响当前进程），避免"Unable to connect to proxy"导致评估中断。
    if os.environ.get("EVAL_DISABLE_PROXY", "1").lower() in ("1", "true", "yes"):
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        no_proxy = os.environ.get("NO_PROXY", "")
        add = "127.0.0.1,localhost"
        if add not in no_proxy:
            os.environ["NO_PROXY"] = (no_proxy + ("," if no_proxy else "") + add)

    context_text = "\n\n".join([f"[{i+1}] {c[:800]}" for i, c in enumerate(contexts)])
    prompt = f"""你是专业法律助手。请只依据给定的【参考资料】回答【用户问题】；如果资料不足，请明确说明不足并给出下一步建议。

【参考资料】:
{context_text}

【用户问题】:
{question}

回答要求：
1) 结构化：问题分析/法律依据/建议/注意事项
2) 不要编造法条编号；如引用，请用"根据《xx法》相关规定"或直接引用原文片段
"""
    # 轻量重试：网络抖动/偶发断连时不至于整次评估失败
    import time
    url = f"{config.LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    last_err = None
    for attempt in range(int(os.environ.get("EVAL_LLM_RETRIES", "3"))):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
        time.sleep(1.5 * (attempt + 1))
    return f"[LLM调用失败] {last_err}"

def _clean_contexts(contexts: list[str], *, max_items: int = 4, max_chars_per_item: int = 800, max_total_chars: int = 2400) -> list[str]:
    """
    让 RAGAS 的 faithfulness/context_* 更稳定：
    - 去重（保序）
    - 单段截断
    - 总长度控制（避免超长上下文让 judge 不稳定/NaN）
    """
    seen = set()
    cleaned: list[str] = []
    total = 0
    for c in contexts:
        c = (c or "").strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        if len(c) > max_chars_per_item:
            c = c[:max_chars_per_item] + "…"
        if total + len(c) > max_total_chars:
            break
        cleaned.append(c)
        total += len(c)
        if len(cleaned) >= max_items:
            break
    return cleaned

def _load_eval_set(path: str) -> list[dict]:
    """
    读取 eval_set.jsonl
    每行: {"question": "...", "ground_truth": "...", ...}
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" in obj and "ground_truth" in obj:
                items.append(obj)
    return items

class ChromaV2HttpRetriever:
    """
    直接通过 Chroma Server v2 HTTP API 检索（绕开 chromadb.HttpClient 初始化时的 identity 异常包装）。
    已验证端点：
    - GET  /api/v2/heartbeat
    - GET  /api/v2/auth/identity
    - GET  /api/v2/tenants/{tenant}/databases/{db}/collections
    - GET  /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/count
    - POST /api/v2/tenants/{tenant}/databases/{db}/collections/{id}/query
    """

    def __init__(self, rag: LegalRAGSystem, host: str = "127.0.0.1", port: int = 8000, collection_name: Optional[str] = None):
        self.rag = rag
        self.base = f"http://{host}:{port}"
        self.tenant = "default_tenant"
        self.database = "default_database"
        self.collection_id = None
        self.collection_name = collection_name

        # health check
        hb = requests.get(self.base + "/api/v2/heartbeat", timeout=10)
        hb.raise_for_status()
        ident = requests.get(self.base + "/api/v2/auth/identity", timeout=10)
        ident.raise_for_status()
        ident_json = ident.json()
        self.tenant = ident_json.get("tenant", self.tenant)
        dbs = ident_json.get("databases") or [self.database]
        self.database = dbs[0]
        print(f"[Eval] Chroma v2 ok: tenant={self.tenant}, db={self.database}")

        self.collection_id = self._pick_collection_id()
        print(f"[Eval] 使用 collection: {self.collection_name} ({self.collection_id})")

    def _collections_url(self) -> str:
        return f"{self.base}/api/v2/tenants/{self.tenant}/databases/{self.database}/collections"

    def _pick_collection_id(self) -> str:
        cols = requests.get(self._collections_url(), timeout=20)
        cols.raise_for_status()
        cols_json = cols.json()
        # server 返回 list[collection]
        if not isinstance(cols_json, list) or not cols_json:
            raise RuntimeError("Chroma Server 未发现任何 collection。请确认 server 的 --path 指向正确的 vector_db。")

        if self.collection_name:
            for c in cols_json:
                if c.get("name") == self.collection_name:
                    return c["id"]
            raise RuntimeError(f"未找到指定 collection_name={self.collection_name}。可用: {[c.get('name') for c in cols_json]}")

        # 默认优先选名为 langchain 的 collection（你当前库就是这个）
        for c in cols_json:
            if c.get("name") == "langchain":
                self.collection_name = "langchain"
                return c["id"]

        # 否则选 count 最大的
        best_id = cols_json[0]["id"]
        best_name = cols_json[0].get("name")
        best_count = -1
        for c in cols_json:
            cid = c["id"]
            name = c.get("name")
            try:
                cnt = requests.get(self._collections_url() + f"/{cid}/count", timeout=20)
                cnt.raise_for_status()
                ccount = int(cnt.text.strip() or "0")
            except Exception:
                ccount = -1
            if ccount > best_count:
                best_count = ccount
                best_id = cid
                best_name = name
        self.collection_name = best_name
        return best_id

    def retrieve(self, question: str, top_k: int = 4) -> list[str]:
        qv = self.rag.embedding_model.embed_query(question)
        url = self._collections_url() + f"/{self.collection_id}/query"
        payload = {
            "query_embeddings": [qv],
            "n_results": int(top_k),
            "include": ["documents"],
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        docs = (data.get("documents") or [[]])[0]
        return [d for d in docs if d]

def run_ragas_evaluation():
    config = Config()
    rag = LegalRAGSystem(config)
    rag.initialize_embeddings()

    # 1) 评估检索器选择：
    # - chroma_server（推荐）：使用已存在的 vector_db，通过 HTTP 访问，避免本进程调用 chromadb rust
    # - local_corpus（fallback）：从 Database 构建 chunks + 缓存向量（首次较慢）
    retriever = (os.environ.get("EVAL_RETRIEVER", "") or "chroma_server").strip().lower()
    chroma_retriever = None
    chunks = None
    chunk_vectors = None

    if retriever == "chroma_server":
        host = os.environ.get("CHROMA_HOST", "127.0.0.1")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        collection_name = os.environ.get("CHROMA_COLLECTION", "") or None
        print(f"[Eval] 使用 Chroma Server 检索：{host}:{port}")
        chroma_retriever = ChromaV2HttpRetriever(rag, host=host, port=port, collection_name=collection_name)
    else:
        print("[Eval] 使用本地语料库检索（不依赖 Chroma Server）")
        chunks, _metas = _build_chunk_corpus(config)
        print(f"[Eval] 已构建切片语料库: {len(chunks)} chunks")

        # --- 关键优化：向量缓存（避免每次评估都把全库重新向量化导致"电脑像死机"） ---
        cache_dir = os.path.join(project_root, "eval_cache")
        os.makedirs(cache_dir, exist_ok=True)

        corpus_fingerprint = {
            "database_path": str(config.DATABASE_PATH),
            "chunk_size": int(config.CHUNK_SIZE),
            "chunk_overlap": int(config.CHUNK_OVERLAP),
            "num_chunks": len(chunks),
        }
        cache_key = _sha1_text(json.dumps(corpus_fingerprint, ensure_ascii=False, sort_keys=True))
        chunks_path = os.path.join(cache_dir, f"chunks_{cache_key}.json")
        vecs_path = os.path.join(cache_dir, f"vectors_{cache_key}.npy")

        if os.path.exists(chunks_path) and os.path.exists(vecs_path):
            print("[Eval] 命中缓存：加载已生成的 chunks/vectors ...")
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            chunk_vectors = np.load(vecs_path)
        else:
            # 可选：限制最大 chunks 数（默认不限制）。用于低配机器先跑通流程。
            max_chunks = int(os.environ.get("EVAL_MAX_CHUNKS", "0") or "0")
            if max_chunks > 0 and len(chunks) > max_chunks:
                print(f"[Eval] EVAL_MAX_CHUNKS={max_chunks}，将随机采样 chunks 以加速评估")
                rng = np.random.default_rng(42)
                idx = rng.choice(len(chunks), size=max_chunks, replace=False)
                idx = np.sort(idx)
                chunks = [chunks[i] for i in idx.tolist()]

            print("[Eval] 正在为 chunks 生成向量（首次运行会较慢）...")
            batch_size = int(os.environ.get("EVAL_EMBED_BATCH_SIZE", "16"))
            chunk_vectors = _embed_chunks(rag.embedding_model, chunks, batch_size=batch_size)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False)
            np.save(vecs_path, chunk_vectors)
            print(f"[Eval] 已写入缓存: {chunks_path} / {vecs_path}")
    
    # 2. 评估数据集：优先从 jsonl 加载（推荐），否则 fallback 到内置小样本
    eval_path = os.environ.get("EVAL_SET", "").strip()
    if eval_path:
        test_data = _load_eval_set(eval_path)
        print(f"[Eval] 已加载评估集: {len(test_data)} 条 <- {eval_path}")
    else:
        test_data = [
            {
                "question": "公司股东有哪些出资方式？",
                "ground_truth": "根据《中华人民共和国公司法》，股东可以用货币出资，也可以用实物、知识产权、土地使用权、股权、债权等可以用货币估价并可以依法转让的非货币财产作价出资。"
            },
            {
                "question": "劳动者在什么情况下可以随时解除劳动合同？",
                "ground_truth": "根据《劳动合同法》，用人单位有下列情形之一的，劳动者可以解除劳动合同：未按照劳动合同约定提供劳动保护或者劳动条件的；未及时足额支付劳动报酬的；未依法为劳动者缴纳社会保险费的等。"
            },
            {
                "question": "民法典中关于违约金的规定是什么？",
                "ground_truth": "根据《中华人民共和国民法典》，当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。"
            },
            {
                "question": "刑法中正当防卫的定义是什么？",
                "ground_truth": "根据《中华人民共和国刑法》，为了使国家、公共利益、本人或者他人的人身、财产和其他权利免受正在进行的不法侵害，而采取的制止不法侵害的行为，对不法侵害人造成损害的，属于正当防卫，不负刑事责任。"
            }
        ]

    # 可选：只跑前 N 条（快速回归）
    limit = int(os.environ.get("EVAL_LIMIT", "0") or "0")
    if limit > 0 and len(test_data) > limit:
        test_data = test_data[:limit]
        print(f"[Eval] EVAL_LIMIT={limit}，仅评估前 {limit} 条")
    
    # 3. 运行 RAG 系统获取预测结果
    print("正在运行评估用 RAG（embedding 检索 + Qwen 生成）收集回答和上下文...")
    results = []
    failed_llm = 0
    for data in test_data:
        print(f"处理问题: {data['question']}")
        top_k = int(os.environ.get("EVAL_TOP_K", "4"))
        if chroma_retriever is not None:
            contexts = chroma_retriever.retrieve(data["question"], top_k=top_k)
        else:
            contexts = _retrieve_topk(rag.embedding_model, chunk_vectors, chunks, data["question"], top_k=top_k)
        contexts = _clean_contexts(
            contexts,
            max_items=int(os.environ.get("EVAL_CTX_ITEMS", str(top_k))),
            max_chars_per_item=int(os.environ.get("EVAL_CTX_CHARS", "800")),
            max_total_chars=int(os.environ.get("EVAL_CTX_TOTAL", "2400")),
        )
        answer = _generate_answer_llm(config, data["question"], contexts)
        if answer.startswith("[LLM调用失败]"):
            failed_llm += 1
            # 不中断评估：仍然记录检索结果，便于分析检索质量；生成相关指标会受影响
            if os.environ.get("EVAL_SKIP_ON_LLM_FAIL", "0").lower() in ("1", "true", "yes"):
                continue
        # RAGAS 0.4.x 期望的字段名（非常关键：否则会把 contexts 当成空）
        results.append(
            {
                "user_input": data["question"],
                "retrieved_contexts": contexts,  # List[str]
                "response": answer,
                "reference": data["ground_truth"],
            }
        )
    if failed_llm:
        print(f"[Eval] 注意：LLM 失败 {failed_llm} 次（已不中断流程）。如需走代理，请设置 EVAL_DISABLE_PROXY=0 并正确配置 NO_PROXY。")
    
    # 现在再导入 ragas，避免其后台线程与本地检索阶段的 chromadb-rust 冲突
    from ragas import evaluate  # noqa: WPS433
    from ragas.metrics import (  # noqa: WPS433
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # 4. 转换为 Ragas 所需的 Dataset 格式
    # 注意：用 from_list 保留 retrieved_contexts 的 List[str] 类型，避免被转成字符串
    dataset = Dataset.from_list(results)
    
    # 5. 配置 Ragas 使用的 Judge LLM (使用当前配置的 LLM 厂商)
    # Ragas 默认使用 OpenAI，我们用 langchain 的 ChatOpenAI 指向兼容接口
    from langchain_community.chat_models import ChatOpenAI

    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "") or config.LLM_MODEL
    llm = ChatOpenAI(
        model_name=judge_model,
        openai_api_key=config.LLM_API_KEY,
        openai_api_base=config.LLM_BASE_URL,
        temperature=0.1,
    )
    # 用本地 BGE embeddings 做指标 embedding，避免额外的网络 embedding 调用，提高速度与稳定性
    embeddings = rag.embedding_model
    
    # 6. 执行评估
    print("\n开始 Ragas 评估 (这可能需要一些时间，取决于 API 响应)...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    # 7. 输出结果
    print("\n" + "="*50)
    print("RAG 评估报告 (Ragas)")
    print("="*50)
    print(result)
    
    # 保存结果到 CSV
    result_df = result.to_pandas()
    result_df.to_csv("rag_eval_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n详细评估结果已保存至: rag_eval_results.csv")

if __name__ == "__main__":
    try:
        run_ragas_evaluation()
    except ImportError as e:
        print(f"缺失库: {e}")
        print("请运行: pip install ragas datasets pandas langchain-community")
    except Exception as e:
        import traceback
        print(f"评估过程中出错: {repr(e)}")
        traceback.print_exc()


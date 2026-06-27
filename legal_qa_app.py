import os
import json
import re
import requests
import urllib3
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

from config.config import Config
from prompts import load_prompt
from src.rag_system.rag_engine import LegalRAGSystem
from src.online_resources.web_search import LegalWebSearch
from src.rag_system.chroma_v2_http import ChromaV2HttpClient

# 禁用 urllib3 的 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class LegalAgent:
    def __init__(self, logger=None):
        self.config = Config()
        self.logger = logger if logger else print
        self.logger(f"[OK] LLM 配置: {self.config.LLM_PROVIDER_NAME} / {self.config.LLM_MODEL}")

        self.rag_system = LegalRAGSystem(self.config, logger=self.logger)
        self.rag_system.initialize_embeddings()

        # 加载向量存储
        from langchain_community.vectorstores import Chroma
        from chromadb.config import Settings
        try:
            self.vector_store = Chroma(
                persist_directory=self.config.VECTOR_DB_PATH,
                embedding_function=self.rag_system.embedding_model,
                client_settings=Settings(anonymized_telemetry=False)
            )
            self.logger(f"[OK] 向量存储加载成功")
        except Exception as e:
            self.logger(f"[X] 向量存储加载失败: {e}")
            self.vector_store = None

        # 可选：使用 Chroma Server（更稳定，且可避免本地 chromadb rust 崩溃/版本不一致）
        self.chroma_http = None
        self.chroma_http_collection = None
        chroma_url = os.environ.get("CHROMA_SERVER_URL", "").strip()
        if chroma_url:
            try:
                http = ChromaV2HttpClient(chroma_url, timeout=30)
                http.heartbeat()
                col = http.pick_collection(prefer_name="langchain")
                self.chroma_http = http
                self.chroma_http_collection = col
                self.logger(f"[OK] Chroma Server 已连接: {chroma_url} collection={col.name} count={col.count}")
            except Exception as e:
                self.logger(f"[X] Chroma Server 连接失败: {e}")

        self.web_searcher = LegalWebSearch(logger=self.logger)
        self.chat_history = []

        # 工具定义
        self.tools = {
            "local_search": self._local_knowledge_tool,
            "web_search": self._web_search_tool,
            "calculator": self._calculator_tool
        }

    def _call_llm(self, prompt, stream=False, **kwargs):
        """统一的 LLM 调用方法（通过 requests 调用 OpenAI 兼容接口，支持流式和重试）"""
        import time
        max_retries = 3
        retry_delay = 2

        url = f"{self.config.LLM_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "stream": stream,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, stream=stream, timeout=120)
                resp.raise_for_status()

                if not stream:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return self._iter_stream(resp)

            except Exception as e:
                self.logger(f"[重试] LLM 调用异常: {str(e)} (尝试 {attempt + 1}/{max_retries})")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        return "LLM调用失败，请检查网络和 API Key。" if not stream else iter([])

    @staticmethod
    def _iter_stream(resp):
        """解析 SSE 流式响应，逐块 yield 内容"""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            line = line.strip()
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def answer_question_stream(self, question: str):
        """流式回答方法"""
        # 构建历史上下文
        history_str = ""
        if self.chat_history:
            recent_history = self.chat_history[-3:]
            history_parts = []
            for q, a in recent_history:
                history_parts.append(f"用户: {q}")
                history_parts.append(f"助手: {a[:200]}...")
            history_str = "\n".join(history_parts)

        # 1. 预处理 (同步)
        pre_enhanced = self._pre_enhancement(question, history_str)
        enhanced_query = pre_enhanced.get("enhanced_query", question)
        is_follow_up = pre_enhanced.get("is_follow_up", False)
        
        # 2. 意图分析 (同步)
        intent = self._router_analyze(enhanced_query, history_str)
        final_enhanced = self._query_enhancement(enhanced_query, intent)
        final_query = final_enhanced.get("enhanced_query", enhanced_query)

        # 3. 工具执行 (同步)
        local_results = []
        web_results = []
        calc_result = None

        if intent.get('need_local', False):
            local_results = self._local_knowledge_tool(final_query)
        if intent.get('need_web', False):
            web_results = self._web_search_tool(intent.get('search_query', final_query))
        
        # 3c. 计算器 (need_calc)
        if intent.get('need_calc', False):
            self.logger(f"[计算器] 正在计算...")
            calc_result = self._run_calculation(question, history_str, local_results, web_results, final_query)

        # 4. 聚合验证 (同步)
        if intent.get('need_local', False) or intent.get('need_web', False):
            aggregated = self._aggregate_and_verify(local_results, web_results, final_query, calc_result)
        else:
            aggregated = {"aggregated_answer": "正在为您生成建议..."}

        # 5. 流式生成最终回答 (异步/生成器)
        sources = []
        if web_results: sources.extend([r['url'] for r in web_results])
        if local_results: sources.append("本地知识库")

        evidence_snippets = self._build_evidence_snippets(local_results, web_results)

        # 构建模板变量
        if is_follow_up and history_str:
            history_context = f"\n历史对话上下文:\n{history_str}\n"
            follow_up_instruction = (
                "这是对之前问题的后续提问，必须结合历史对话中的信息来回答。"
            )
        else:
            history_context = ""
            follow_up_instruction = "这是一个全新的、独立的法律问题。"

        # 动态构建板块说明
        has_calc = calc_result and calc_result != "无法提取计算所需的信息，请提供具体的金额和时间信息"
        calc_instruction = f"计算器结果（必须包含在回答中）:\n{calc_result}" if has_calc else ""

        sections = ["问题分析", "法律依据", "具体建议", "注意事项"]
        if has_calc:
            sections.insert(2, "计算结果")
        if is_follow_up:
            section_instruction = f"包括以下板块：{'、'.join(sections)}。如果是后续问题，要结合历史上下文进行详细说明。"
        else:
            section_instruction = f"包括以下板块：{'、'.join(sections)}。"

        final_prompt = load_prompt("final_answer",
            original_query=question,
            history_context=history_context,
            aggregated_answer=aggregated.get("aggregated_answer", ""),
            consistent_info="\n".join(aggregated.get("consistent_info", [])) or "无",
            credible_sources="\n".join(aggregated.get("credible_sources", [])) or "无",
            evidence_snippets=evidence_snippets or "（无）",
            calc_instruction=calc_instruction,
            follow_up_instruction=follow_up_instruction,
            section_instruction=section_instruction,
        )
        responses = self._call_llm(final_prompt, stream=True)
        
        full_content = ""
        for chunk in responses:
            full_content += chunk
            yield chunk
        
        # 添加来源 (非流式部分，最后一次性添加)
        if sources:
            source_text = "\n\n【参考来源】:\n" + "\n".join(sources)
            full_content += source_text
            yield source_text

        # 存入历史
        self.chat_history.append((question, full_content))


    def _pre_enhancement(self, original_query, history_str=""):
        """智能提问增强（在意图识别之前）"""
        self.logger(f"[调试] 原始查询: {original_query}")
        self.logger(f"[调试] 历史上下文: {history_str[:100]}..." if history_str else "[调试] 无历史上下文")

        # 判断是否为追问：需要同时满足 有历史 + 有明确的指代/承接词
        follow_up_phrases = ["那", "那我", "这样的话", "如果是这样", "接着", "然后",
                             "上面", "之前", "刚才", "你说的", "这个方案"]
        pronoun_refs = ["这个", "那个", "他", "她", "它", "这笔", "这个情况"]
        has_explicit_ref = any(p in original_query for p in follow_up_phrases)
        has_pronoun_ref = any(p in original_query for p in pronoun_refs) and history_str
        is_likely_follow_up = history_str and (has_explicit_ref or has_pronoun_ref)

        self.logger(f"[调试] 追问判断: {is_likely_follow_up} (显式引用={has_explicit_ref}, 代词引用={has_pronoun_ref})")

        # 如果是后续问题，先从历史中提取关键信息
        key_info_from_history = {}
        if is_likely_follow_up and history_str:
            key_info_from_history = self._extract_key_info_from_history(history_str)

        # 构建历史上下文（只在是后续问题时使用）
        history_context = ""
        if is_likely_follow_up and history_str:
            key_info_str = ""
            if key_info_from_history.get("amount"):
                key_info_str += f"金额：{key_info_from_history['amount']:,}元；"
            if key_info_from_history.get("time_period"):
                key_info_str += f"时间：{key_info_from_history['time_period']:.1f}个月；"
            if key_info_str:
                history_context = f"\n历史对话上下文:\n{history_str}\n\n从历史中提取的关键信息：{key_info_str}\n"
            else:
                history_context = f"\n历史对话上下文:\n{history_str}\n"

        follow_up_instruction = (
            "**这是后续问题，必须从历史对话中提取关键信息（金额、时间、主体等）并自然地整合到句子中。**"
            if is_likely_follow_up
            else "这是独立问题，不需要结合历史"
        )

        enhancement_prompt = load_prompt("pre_enhancement",
            original_query=original_query,
            history_context=history_context,
            follow_up_instruction=follow_up_instruction,
            is_follow_up=str(is_likely_follow_up).lower(),
        )
        response = self._call_llm(enhancement_prompt, temperature=0.3, max_tokens=500)

        default_result = {
            "enhanced_query": original_query,
            "is_follow_up": False
        }

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_result = json.loads(json_match.group())
                if isinstance(parsed_result, dict) and "enhanced_query" in parsed_result:
                    enhanced_query = parsed_result.get("enhanced_query", original_query).strip()

                    # 1. 如果增强后的查询和原问题完全相同，说明没有优化
                    if enhanced_query == original_query:
                        self.logger(f"[警告] 提问增强未优化，使用原问题: {original_query}")
                        return default_result
                    
                    # 调试：显示LLM返回的增强查询
                    self.logger(f"[调试] LLM返回的增强查询: {enhanced_query}")
                    
                    # 2. 检查是否包含原问题的核心词汇
                    # 中文问题不应该用空格分割，使用更合理的关键词提取方法
                    original_keywords = []
                    # 提取长度大于1的中文词汇
                    for i in range(len(original_query)):
                        if i + 2 <= len(original_query):
                            word = original_query[i:i+2]
                            if len(word.strip()) > 1:
                                original_keywords.append(word)
                    
                    # 同时包含一些重要的单字词
                    important_chars = ["欠", "拖", "赔", "法", "律", "工", "资"]
                    for char in important_chars:
                        if char in original_query:
                            original_keywords.append(char)
                    
                    if original_keywords:
                        matched_keywords = [kw for kw in original_keywords if kw in enhanced_query]
                        # 进一步放宽匹配阈值到15%，主要检查是否完全偏离
                        if len(matched_keywords) < len(original_keywords) * 0.15:
                            self.logger(f"[警告] 增强后查询严重偏离原意，使用原问题: {original_query}")
                            self.logger(f"[调试] 原问题关键词: {original_keywords}")
                            self.logger(f"[调试] 匹配的关键词: {matched_keywords}")
                            return default_result
                        else:
                            self.logger(f"[调试] 验证通过 - 匹配 {len(matched_keywords)}/{len(original_keywords)} 个关键词")

                    # 3. 如果是后续问题，检查是否合理结合了历史信息
                    if is_likely_follow_up and key_info_from_history:
                        amount = key_info_from_history.get("amount")
                        time_period = key_info_from_history.get("time_period")
                        
                        # 检查是否包含金额信息（如果有的话）
                        if amount and str(amount) not in enhanced_query:
                            # 尝试自动补充金额信息
                            amount_text = f"{int(amount / 10000)}万元" if amount >= 10000 else f"{amount}元"
                            if "工资" in enhanced_query:
                                enhanced_query = enhanced_query.replace("工资", f"工资{amount_text}", 1)
                        
                        # 检查是否包含时间信息（如果有的话）
                        if time_period and str(time_period) not in enhanced_query:
                            time_text = "半年" if time_period == 6 else (
                                f"{int(time_period / 12)}年" if time_period >= 12 else f"{int(time_period)}个月")
                            if "已经" in enhanced_query:
                                enhanced_query = enhanced_query.replace("已经", f"已经{time_text}")
                            elif "拖了" in enhanced_query:
                                enhanced_query = enhanced_query.replace("拖了", f"拖了{time_text}")

                    return {
                        "enhanced_query": enhanced_query,
                        "is_follow_up": parsed_result.get("is_follow_up", is_likely_follow_up)
                    }

            # 如果解析失败，使用原问题
            self.logger(f"[警告] 提问增强解析失败，使用原问题: {original_query}")
            return default_result

        except Exception as e:
            self.logger(f"[警告] 提问增强处理异常: {e}，使用原问题")
            return default_result

    def _rule_based_intent(self, query):
        """基于规则的关键词匹配（快速判断）"""
        query_lower = query.lower()

        # 网络搜索触发关键词（需要最新信息）
        web_keywords = [
            "最新", "最近", "今年", "本月", "2024", "2025", "2026",
            "案例", "判例", "判决", "司法", "法院", "庭审",
            "专家", "律师", "意见", "解读", "分析",
            "即将", "实施", "生效", "发布", "出台"
        ]

        # 本地搜索关键词（法律条款、概念）
        local_keywords = [
            "法律", "法条", "条款", "规定", "条例", "办法",
            "概念", "定义", "解释", "是什么", "什么意思",
            "怎么", "如何", "流程", "程序", "步骤"
        ]

        # 计算关键词
        calc_keywords = [
            "计算", "多少", "金额", "利息", "赔偿", "违约金",
            "罚款", "费用", "成本", "比例", "百分比"
        ]

        need_web = any(kw in query_lower for kw in web_keywords)
        need_local = any(kw in query_lower for kw in local_keywords)
        need_calc = any(kw in query_lower for kw in calc_keywords)

        return {
            "need_web": need_web,
            "need_local": need_local,
            "need_calc": need_calc
        }

    def _router_analyze(self, enhanced_query, history_str=""):
        """意图识别和路由分析（两阶段：规则快筛 + LLM 语义分析）

        规则提供初判线索，LLM 做最终决策。
        即使规则全未命中，也必须调 LLM 做语义判断，避免漏掉纯语义问题。
        """
        # 第一阶段：规则快筛（零成本，给 LLM 参考）
        rule_intent = self._rule_based_intent(enhanced_query)

        # 第二阶段：LLM 语义分析（始终调用，规则结果仅作参考）
        history_part = f"历史对话:\n{history_str}\n" if history_str else ""

        router_prompt = load_prompt("router_analyze",
            enhanced_query=enhanced_query,
            history_part=history_part,
            rule_need_local=rule_intent["need_local"],
            rule_need_web=rule_intent["need_web"],
            rule_need_calc=rule_intent["need_calc"],
        )
        response = self._call_llm(router_prompt, temperature=0.1, max_tokens=600)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "need_local": result.get("need_local", rule_intent["need_local"]),
                    "need_web": result.get("need_web", rule_intent["need_web"]),
                    "need_calc": result.get("need_calc", rule_intent["need_calc"]),
                    "search_query": result.get("search_query", enhanced_query),
                    "question_type": result.get("question_type", "legal"),
                    "reasoning": result.get("reasoning", "")
                }
        except Exception as e:
            self.logger(f"[警告] LLM意图识别解析失败: {e}，使用规则判断结果")

        # LLM 解析失败时 fallback 到规则结果
        return {
            "need_local": rule_intent["need_local"],
            "need_web": rule_intent["need_web"],
            "need_calc": rule_intent["need_calc"],
            "search_query": enhanced_query,
            "question_type": "legal",
            "reasoning": "LLM解析失败，使用规则关键词匹配"
        }

    def _query_enhancement(self, enhanced_query, intent_analysis):
        """进一步提问增强（在意图识别之后，针对特定工具优化检索语句）"""
        # 根据意图类型，对检索语句做针对性补充
        suffix_parts = []

        if intent_analysis.get("need_calc"):
            suffix_parts.append("赔偿计算标准 利息计算方法")

        if intent_analysis.get("need_web"):
            suffix_parts.append("最新规定 司法判例")

        # 如果没有需要补充的，直接返回
        if not suffix_parts:
            return {"enhanced_query": enhanced_query}

        enhanced = f"{enhanced_query} {' '.join(suffix_parts)}"
        self.logger(f"[查询增强] {enhanced}")
        return {"enhanced_query": enhanced}

    def _local_knowledge_tool(self, query, top_k=None):
        """本地知识库搜索工具"""
        # 优先使用 Chroma Server（如果配置了 CHROMA_SERVER_URL）
        if self.chroma_http and self.chroma_http_collection:
            return self._server_knowledge_tool(query, top_k=top_k)

        if self.vector_store is None:
            return []

        if top_k is None:
            top_k = self.config.TOP_K_RESULTS

        # 检索优化：对 query 去噪/提取法名&条号/多查询扩展/按 law_name 元数据过滤
        norm_query, law_name, article_no = self._normalize_retrieval_query(query)
        expanded_queries = self._build_retrieval_queries(query, norm_query, law_name, article_no)

        adaptive = bool(getattr(self.config, "RETRIEVAL_ADAPTIVE", True))
        max_top_k = int(getattr(self.config, "RETRIEVAL_MAX_TOP_K", max(top_k, 10)))
        min_unique = int(getattr(self.config, "RETRIEVAL_MIN_UNIQUE_RESULTS", max(4, top_k)))

        # 每个 query 取少量，合并后再截断到 top_k（提升 recall，降低跑偏）
        per_q_k = max(2, min(8, int(getattr(self.config, "RETRIEVAL_PER_QUERY_K", 4))))

        merged: Dict[str, Dict] = {}

        def _add_docs(docs_with_score):
            for doc, score in docs_with_score:
                key = f"{hash(doc.page_content)}"
                # 取更"好"的 score（Chroma score 可能是距离，越小越好；这里用 min 兼容）
                if key not in merged or score < merged[key]["score"]:
                    merged[key] = {"doc": doc, "score": score}

        # 1) 优先按 law_name 过滤（如果能从问题里识别出来）
        if law_name:
            where = {"law_name": law_name}
            for q in expanded_queries:
                try:
                    docs = self.vector_store.similarity_search_with_score(q, k=per_q_k, filter=where)
                except TypeError:
                    # 兼容旧版本 langchain_chroma 接口：filter 参数可能叫 where
                    docs = self.vector_store.similarity_search_with_score(q, k=per_q_k, where=where)
                _add_docs(docs)

        # 2) 再做不带过滤的检索补齐（避免识别错误导致召回下降）
        for q in expanded_queries:
            docs = self.vector_store.similarity_search_with_score(q, k=per_q_k)
            _add_docs(docs)

        # 合并后按 score 排序
        ranked = sorted(merged.values(), key=lambda x: x["score"])
        results = [{"content": it["doc"].page_content, "metadata": it["doc"].metadata, "score": it["score"]} for it in ranked]

        # Top-K 自适应：如果命中不足/多样性不足，则扩大候选池再 rerank 降噪
        if adaptive and (len(results) < min_unique):
            # 扩大每 query 的 k，再补一次检索
            per_q_k2 = max(per_q_k, min(20, max_top_k * 2))
            for q in expanded_queries:
                docs = self.vector_store.similarity_search_with_score(q, k=per_q_k2)
                _add_docs(docs)
            ranked2 = sorted(merged.values(), key=lambda x: x["score"])
            results = [{"content": it["doc"].page_content, "metadata": it["doc"].metadata, "score": it["score"]} for it in ranked2]

        # 轻量 rerank + 截断
        return self._adaptive_rerank_and_trim(query, results, top_k)

    def _server_knowledge_tool(self, query: str, top_k: Optional[int] = None):
        """
        通过 Chroma Server v2 HTTP API 检索（documents + metadatas + distances）。
        """
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS

        norm_query, law_name, article_no = self._normalize_retrieval_query(query)
        expanded_queries = self._build_retrieval_queries(query, norm_query, law_name, article_no)
        adaptive = bool(getattr(self.config, "RETRIEVAL_ADAPTIVE", True))
        max_top_k = int(getattr(self.config, "RETRIEVAL_MAX_TOP_K", max(top_k, 10)))
        min_unique = int(getattr(self.config, "RETRIEVAL_MIN_UNIQUE_RESULTS", max(4, top_k)))
        per_q_k = max(2, min(10, int(getattr(self.config, "RETRIEVAL_PER_QUERY_K", 4))))

        merged: Dict[str, Dict] = {}

        def _add_one(doc_text: str, meta: dict, dist: float):
            key = f"{hash(doc_text)}"
            if key not in merged or dist < merged[key]["score"]:
                merged[key] = {"content": doc_text, "metadata": meta or {}, "score": dist}

        for q in expanded_queries:
            qv = self.rag_system.embedding_model.embed_query(q)
            data = self.chroma_http.query(
                collection_id=self.chroma_http_collection.id,
                query_embeddings=[qv],
                n_results=per_q_k,
                include=["documents", "metadatas", "distances"],
            )
            docs = (data.get("documents") or [[]])[0]
            metas = (data.get("metadatas") or [[]])[0]
            dists = (data.get("distances") or [[]])[0]
            for i in range(min(len(docs), len(dists))):
                _add_one(docs[i], metas[i] if i < len(metas) else {}, dists[i])

        ranked = sorted(merged.values(), key=lambda x: x["score"])

        # 自适应补召回：扩大 n_results 再查一轮
        if adaptive and (len(ranked) < min_unique):
            per_q_k2 = max(per_q_k, min(30, max_top_k * 2))
            for q in expanded_queries:
                qv = self.rag_system.embedding_model.embed_query(q)
                data = self.chroma_http.query(
                    collection_id=self.chroma_http_collection.id,
                    query_embeddings=[qv],
                    n_results=per_q_k2,
                    include=["documents", "metadatas", "distances"],
                )
                docs = (data.get("documents") or [[]])[0]
                metas = (data.get("metadatas") or [[]])[0]
                dists = (data.get("distances") or [[]])[0]
                for i in range(min(len(docs), len(dists))):
                    _add_one(docs[i], metas[i] if i < len(metas) else {}, dists[i])
            ranked = sorted(merged.values(), key=lambda x: x["score"])

        # 轻量 rerank + 截断
        return self._adaptive_rerank_and_trim(query, ranked, top_k)

    def _normalize_retrieval_query(self, query: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        对检索 query 做去噪，并尽可能提取法名/条号。
        返回: (normalized_query, law_name, article_no)
        """
        q = (query or "").strip()

        # 去掉评估集/模板常见尾巴
        q = re.sub(r"请(简要)?(概括|总结).*$", "", q)
        q = re.sub(r"(主要)?规定了?什么.*$", "", q)
        q = re.sub(r"要点.*$", "", q)
        q = re.sub(r"[，。！？；]+$", "", q)
        q = re.sub(r"\s+", " ", q).strip()

        # 抽取《法名》
        law_name = None
        m = re.search(r"《([^》]{2,30})》", query)
        if m:
            law_name = m.group(1).strip()

        # 抽取"第X条"
        article_no = None
        m2 = re.search(r"(第[一二三四五六七八九十百千万0-9]+条)", query)
        if m2:
            article_no = m2.group(1)

        return q, law_name, article_no

    def _build_retrieval_queries(self, raw_query: str, norm_query: str, law_name: Optional[str], article_no: Optional[str]) -> List[str]:
        """
        多查询扩展：在不引入额外 LLM 成本的情况下，提高召回。
        """
        qs = []
        for x in [raw_query, norm_query]:
            x = (x or "").strip()
            if x and x not in qs:
                qs.append(x)

        # 如果能识别法名/条号，加入更"检索友好"的短 query
        if law_name and article_no:
            short = f"{law_name} {article_no}"
            if short not in qs:
                qs.insert(0, short)
        if law_name and norm_query and law_name not in norm_query:
            q2 = f"{law_name} {norm_query}"
            if q2 not in qs:
                qs.append(q2)
        if article_no and article_no not in norm_query:
            q3 = f"{article_no} {norm_query}"
            if q3 not in qs:
                qs.append(q3)

        # 最多保留 4 个，避免额外开销过大
        return qs[:4]

    def _simple_keyword_overlap(self, query: str, text: str) -> int:
        """
        极轻量的 rerank 特征：query 与 text 的关键片段重合度（中文用 2-gram + 条号/法名优先）。
        """
        q = (query or "").strip()
        t = (text or "").strip()
        if not q or not t:
            return 0

        tokens = set()
        # 2-gram
        for i in range(len(q) - 1):
            g = q[i : i + 2]
            if g.strip():
                tokens.add(g)
        # 条号、法名直接加权
        m = re.search(r"(第[一二三四五六七八九十百千万0-9]+条)", q)
        if m:
            tokens.add(m.group(1))
        m2 = re.search(r"《([^》]{2,30})》", q)
        if m2:
            tokens.add(m2.group(1))

        hit = 0
        for tok in tokens:
            if tok and tok in t:
                hit += 1
        return hit

    def _adaptive_rerank_and_trim(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        对候选进行降噪：优先 keyword overlap，再按距离/score（越小越好）排序。
        """
        for c in candidates:
            c["_overlap"] = self._simple_keyword_overlap(query, c.get("content", ""))
        ranked = sorted(candidates, key=lambda x: (-x.get("_overlap", 0), x.get("score", 0)))
        for c in ranked:
            c.pop("_overlap", None)
        return ranked[:top_k]

    def _web_search_tool(self, query, max_results=None):
        """网络搜索工具"""
        if max_results is None:
            max_results = self.config.MAX_RESULTS

        return self.web_searcher.search(query, max_results=max_results)

    def _run_calculation(self, question, history_str, local_results, web_results, final_query):
        """统一的法律赔偿计算器（流式/非流式共用）

        从历史对话提取金额/时间，从法律文本提取计算公式，执行计算。
        Returns: calc_result 字符串，或 None（无法计算时）
        """
        # 从历史对话中提取关键信息
        key_info = self._extract_key_info_from_history(history_str)

        # 从当前问题中提取时间信息（优先使用当前问题中的时间）
        current_time_info = self._extract_key_info_from_history(question)
        if current_time_info.get("time_period"):
            key_info["time_period"] = current_time_info["time_period"]
            self.logger(f"[计算器] 从当前问题中提取时间：{key_info['time_period']:.1f}个月")

        # 缺少金额或时间 → 尝试提取简单算式
        if not key_info["amount"] or not key_info["time_period"]:
            calc_expressions = re.findall(r'\d+[+\-*/]\d+', question)
            if calc_expressions:
                return self._calculator_tool(calc_expressions[0])
            missing = []
            if not key_info["amount"]:
                missing.append("金额")
            if not key_info["time_period"]:
                missing.append("时间")
            return f"无法提取计算所需的信息，缺少：{', '.join(missing)}"

        # 有金额和时间 → 尝试从法律文本中提取计算公式
        if not local_results and not web_results:
            self.logger(f"[计算器] 需要检索法律依据，正在检索...")
            calc_query = f"{final_query} 赔偿计算 利息计算 违约金计算"
            local_results = self._local_knowledge_tool(calc_query)
            if not local_results:
                web_results = self._web_search_tool(calc_query)

        calc_method = None
        if local_results or web_results:
            calc_method = self._extract_calculation_method(local_results, web_results, final_query)

        amount = key_info["amount"]
        time_period = key_info["time_period"]

        if calc_method and calc_method.get("has_explicit_method"):
            # 使用法律规定的计算方式
            rate = calc_method.get("rate")
            rate_type = calc_method.get("rate_type", "")
            method_desc = calc_method.get("calculation_method", "")
            legal_basis = calc_method.get("legal_basis", "")

            interest = 0
            additional_compensation = 0
            calculation_details = []

            if "年利率" in rate_type or "年利率" in method_desc:
                if rate:
                    if isinstance(rate, str) and "-" in rate:
                        rate_parts = rate.split("-")
                        min_r, max_r = float(rate_parts[0]), float(rate_parts[1])
                        avg_r = (min_r + max_r) / 2
                        interest = amount * (avg_r / 12) * time_period
                        calculation_details.append(
                            f"- 逾期利息（年利率{min_r*100:.1f}%-{max_r*100:.1f}%，取中间值{avg_r*100:.1f}%）：{interest:,.2f}元")
                        calculation_details.append(
                            f"  （范围：{amount*(min_r/12)*time_period:,.2f}元 - {amount*(max_r/12)*time_period:,.2f}元）")
                    else:
                        annual_rate = float(rate)
                        interest = amount * (annual_rate / 12) * time_period
                        calculation_details.append(
                            f"- 逾期利息（年利率{annual_rate*100:.1f}%）：{interest:,.2f}元")
            elif "月利率" in rate_type or "月利率" in method_desc:
                if rate:
                    monthly_rate = float(rate)
                    interest = amount * monthly_rate * time_period
                    calculation_details.append(
                        f"- 逾期利息（月利率{monthly_rate*100:.1f}%）：{interest:,.2f}元")
            elif "日利率" in rate_type or "日利率" in method_desc:
                if rate:
                    daily_rate = float(rate)
                    interest = amount * daily_rate * time_period * 30
                    calculation_details.append(
                        f"- 逾期利息（日利率{daily_rate*100:.3f}%）：{interest:,.2f}元")
            elif "比例" in rate_type or "%" in method_desc or "百分之" in method_desc or "加付赔偿金" in method_desc:
                if rate:
                    if isinstance(rate, str) and "-" in rate:
                        rate_parts = rate.split("-")
                        min_r, max_r = float(rate_parts[0]), float(rate_parts[1])
                        avg_r = (min_r + max_r) / 2
                        additional_compensation = amount * avg_r
                        calculation_details.append(
                            f"- 加付赔偿金（应付金额的{min_r*100:.0f}%-{max_r*100:.0f}%，取中间值{avg_r*100:.0f}%）：{additional_compensation:,.2f}元")
                        calculation_details.append(
                            f"  （范围：{amount*min_r:,.2f}元 - {amount*max_r:,.2f}元）")
                    else:
                        comp_rate = float(rate)
                        additional_compensation = amount * comp_rate
                        calculation_details.append(
                            f"- 加付赔偿金（应付金额的{comp_rate*100:.0f}%）：{additional_compensation:,.2f}元")
            elif "倍" in method_desc or "倍数" in rate_type:
                if rate:
                    multiplier = float(rate)
                    additional_compensation = amount * (multiplier - 1)
                    calculation_details.append(
                        f"- 赔偿金（按{multiplier}倍计算）：{additional_compensation:,.2f}元")

            total = amount + interest + additional_compensation
            calc_result = f"""计算说明（基于法律规定）：
- 本金：{amount:,}元
- 逾期时间：{time_period:.1f}个月
{chr(10).join(calculation_details)}
- 总计应支付：{total:,.2f}元

法律依据：{legal_basis if legal_basis else method_desc}
计算方式：{method_desc}
"""
            self.logger(f"[计算器] 完成（法律规定）：本金{amount:,}元，逾期{time_period:.1f}个月，总计{total:,.2f}元")
            return calc_result
        else:
            # 默认方式（年利率6%）
            self.logger(f"[计算器] 未找到明确法律计算方式，使用默认年利率6%")
            monthly_rate = 0.06 / 12
            interest = amount * monthly_rate * time_period
            total = amount + interest
            calc_result = f"""计算说明：
- 本金：{amount:,}元
- 逾期时间：{time_period:.1f}个月
- 逾期利息（年利率6%，默认标准）：{interest:,.2f}元
- 总计应支付：{total:,.2f}元

注：此计算为默认标准，具体利率可能因合同约定或法律规定而异。
"""
            self.logger(f"[计算器] 完成（默认）：本金{amount:,}元，逾期{time_period:.1f}个月，总计{total:,.2f}元")
            return calc_result

    def _extract_key_info_from_history(self, history_str):
        """从历史对话中提取关键信息（金额、时间等）"""
        key_info = {
            "amount": None,
            "time_period": None,
            "subject": None
        }

        if not history_str:
            return key_info

        # 提取金额（如"3万"、"30000"等）
        amount_patterns = [
            r'(\d+)\s*[万]',
            r'(\d+)\s*元',
            r'(\d+)\s*块',
            r'拖欠.*?(\d+)',
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, history_str)
            if match:
                amount_str = match.group(1)
                if '万' in match.group(0):
                    key_info["amount"] = int(amount_str) * 10000
                else:
                    key_info["amount"] = int(amount_str)
                break

        # 提取时间 — 先处理复合表达式（如"3个月15天"），再处理单一表达式
        total_months = 0.0
        matched_any_time = False

        # 1) 复合时间: "X个月Y天"
        compound = re.search(r'(\d+)\s*个?月\s*(\d+)\s*天', history_str)
        if compound:
            total_months = int(compound.group(1)) + int(compound.group(2)) / 30
            matched_any_time = True

        if not matched_any_time:
            # 2) 单一时间表达式
            single_patterns = [
                (r'半年', lambda m: 6),
                (r'(\d+)\s*个?月', lambda m: int(m.group(1))),
                (r'(\d+)\s*年', lambda m: int(m.group(1)) * 12),
                (r'(\d+)\s*天', lambda m: int(m.group(1)) / 30),
            ]
            for pattern, calc_fn in single_patterns:
                match = re.search(pattern, history_str)
                if match:
                    total_months = calc_fn(match)
                    matched_any_time = True
                    break

        if matched_any_time:
            key_info["time_period"] = total_months

        return key_info

    def _calculator_tool(self, expression):
        """计算器工具"""
        try:
            # 安全计算，避免执行恶意代码
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "无效的计算表达式"

            result = eval(expression, {"__builtins__": {}})
            return f"计算结果: {result}"
        except:
            return "计算失败，请检查表达式"

    def _extract_calculation_method(self, local_results, web_results, query):
        """从法律检索结果中提取赔偿计算方式"""
        # 合并所有检索结果
        all_content = []

        # 从本地知识库结果中提取
        for result in local_results[:3]:  # 只取前3条
            all_content.append(result.get("content", ""))

        # 从网络搜索结果中提取
        for result in web_results[:3]:  # 只取前3条
            all_content.append(result.get("content", ""))

        if not all_content:
            return None

        combined_content = "\n\n".join(all_content)

        # 使用LLM提取计算方式
        extraction_prompt = load_prompt("extract_calculation",
            combined_content=combined_content[:3000],
            query=query,
        )

        try:
            response = self._call_llm(extraction_prompt, temperature=0.1, max_tokens=800)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("has_explicit_method") and result.get("calculation_method"):
                    return result
        except Exception as e:
            self.logger(f"   [警告] 提取计算方式失败: {e}")

        return None

    def _aggregate_and_verify(self, local_results, web_results, enhanced_query, calc_result=None):
        """多信息聚合和交叉验证"""
        local_info = "\n".join([f"- {r['content'][:200]}..." for r in local_results[:3]]) if local_results else "无"
        web_info = "\n".join(
            [f"- {r['title']}: {r['content'][:200]}..." for r in web_results[:3]]) if web_results else "无"
        calc_info = calc_result if calc_result else "无"

        aggregation_prompt = load_prompt("aggregate_verify",
            enhanced_query=enhanced_query,
            local_info=local_info,
            web_info=web_info,
            calc_info=calc_info,
        )
        response = self._call_llm(aggregation_prompt, temperature=0.2, max_tokens=1500)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "consistent_info": [],
                    "conflicts": [],
                    "credible_sources": [],
                    "aggregated_answer": response
                }
        except:
            return {
                "consistent_info": [],
                "conflicts": [],
                "credible_sources": [],
                "aggregated_answer": response
            }

    def _build_evidence_snippets(self, local_results, web_results, max_local=3, max_web=3, max_chars=500):
        """
        构造"证据片段"给最终回答引用，减少幻觉/跑偏。
        输出为可直接拼到 prompt 的文本，带可引用的编号：
        - [本地1] ...
        - [网1] ...
        """
        snippets = []

        if local_results:
            for i, r in enumerate(local_results[:max_local], 1):
                content = (r.get("content") or "").strip()
                if not content:
                    continue
                meta = r.get("metadata") or {}
                law = meta.get("law_name") or "本地知识库"
                src = meta.get("source") or ""
                head = content[:max_chars].replace("\n", " ").strip()
                suffix = f"（{law}{' | ' + src if src else ''}）"
                snippets.append(f"- [本地{i}] {head}{suffix}")

        if web_results:
            for i, r in enumerate(web_results[:max_web], 1):
                title = (r.get("title") or "").strip()
                content = (r.get("content") or "").strip()
                url = (r.get("url") or "").strip()
                head = (content[:max_chars].replace("\n", " ").strip()) if content else ""
                label = f"{title}（{url}）" if url else (title or "网络来源")
                snippets.append(f"- [网{i}] {head}（{label}）")

        return "\n".join(snippets) if snippets else "（无可用证据片段）"

    def _generate_final_answer(self, aggregated_info, original_query, sources, history_str="", is_follow_up=False,
                               calc_result=None, evidence_snippets: str = ""):
        """生成最终回答"""
        # 只有当确实是后续问题时，才使用历史上下文
        if is_follow_up and history_str:
            history_context = f"\n历史对话上下文:\n{history_str}\n"
            follow_up_instruction = (
                "这是对之前问题的后续提问，必须结合历史对话中的信息（如之前提到的金额、时间、具体情况等）来回答。"
                "回答要连贯，体现对之前对话的理解和记忆。"
                "如果当前问题涉及计算（如赔偿金额），要基于历史对话中的具体数字进行计算。"
            )
        else:
            history_context = ""
            follow_up_instruction = (
                "这是一个全新的、独立的法律问题，与之前的对话无关。"
                "请只回答当前问题，不要提及或混合之前对话中的任何内容。"
            )

        # 动态构建板块说明
        has_calc = calc_result and calc_result != "无法提取计算所需的信息，请提供具体的金额和时间信息"
        calc_instruction = f"计算器结果（必须包含在回答中）:\n{calc_result}" if has_calc else ""

        sections = ["问题分析", "法律依据", "具体建议", "注意事项"]
        if has_calc:
            sections.insert(2, "计算结果")
        if is_follow_up:
            section_instruction = f"包括以下板块：{'、'.join(sections)}。如果是后续问题，要结合历史上下文进行详细说明。"
        else:
            section_instruction = f"包括以下板块：{'、'.join(sections)}。"

        final_prompt = load_prompt("final_answer",
            original_query=original_query,
            history_context=history_context,
            aggregated_answer=aggregated_info.get("aggregated_answer", ""),
            consistent_info="\n".join(aggregated_info.get("consistent_info", [])) or "无",
            credible_sources="\n".join(aggregated_info.get("credible_sources", [])) or "无",
            evidence_snippets=evidence_snippets or "（无）",
            calc_instruction=calc_instruction,
            follow_up_instruction=follow_up_instruction,
            section_instruction=section_instruction,
        )

        answer = self._call_llm(final_prompt, temperature=0.3, max_tokens=2000)

        # 如果有计算器结果，确保在回答中包含
        if calc_result and calc_result != "无法提取计算所需的信息，请提供具体的金额和时间信息":
            # 检查回答中是否包含计算结果的关键信息
            calc_keywords = ["本金", "利息", "总计", "应支付", "赔偿", "元"]
            has_calc_info = any(kw in answer for kw in calc_keywords)

            # 如果回答中没有包含计算结果，在适当位置添加
            if not has_calc_info:
                # 在"具体建议"部分之前或之后添加计算结果
                if "### 具体建议" in answer or "具体建议" in answer:
                    # 在具体建议部分之前添加
                    answer = answer.replace("### 具体建议", f"### 计算结果\n\n{calc_result}\n\n### 具体建议")
                    answer = answer.replace("具体建议", f"### 计算结果\n\n{calc_result}\n\n### 具体建议", 1)
                else:
                    # 在回答末尾添加
                    answer += f"\n\n### 计算结果\n\n{calc_result}"

        # 添加来源
        if sources:
            answer += "\n\n【参考来源】:\n" + "\n".join(sources)

        return answer

    def answer_question(self, question: str) -> str:
        """主回答方法"""
        # 构建历史上下文（包含更多历史信息）
        history_str = ""
        if self.chat_history:
            recent_history = self.chat_history[-3:]  # 增加到最近3轮对话
            history_parts = []
            for q, a in recent_history:
                history_parts.append(f"用户: {q}")
                history_parts.append(f"助手: {a[:200]}...")  # 增加答案长度
            history_str = "\n".join(history_parts)

        # 1. 初步提问增强（在意图识别之前）
        pre_enhanced = self._pre_enhancement(question, history_str)
        enhanced_query = pre_enhanced.get("enhanced_query", question)
        is_follow_up = pre_enhanced.get("is_follow_up", False)
        self.logger(f"[初步增强] 优化查询: {enhanced_query}")

        # 2. 意图识别和路由（基于增强后的问题）
        intent = self._router_analyze(enhanced_query, history_str)
        reasoning = intent.get('reasoning', '')
        self.logger(
            f"[意图分析] 需要本地: {intent.get('need_local', False)}, 网络: {intent.get('need_web', False)}, 计算: {intent.get('need_calc', False)}")
        if reasoning:
            self.logger(f"[推理过程] {reasoning}")

        # 3. 进一步提问增强（针对特定工具优化）
        final_enhanced = self._query_enhancement(enhanced_query, intent)
        final_query = final_enhanced.get("enhanced_query", enhanced_query)

        # 4. 调用工具
        local_results = []
        web_results = []
        calc_result = None

        if intent.get('need_local', False):
            local_results = self._local_knowledge_tool(final_query)
            self.logger(f"[本地搜索] 找到 {len(local_results)} 条结果")
            # 如果本地搜索结果很少或相关性低，建议补充网络搜索
            if len(local_results) == 0 and not intent.get('need_web', False):
                self.logger(f"[提示] 本地知识库未找到相关内容，建议启用网络搜索")
                intent['need_web'] = True

        if intent.get('need_web', False):
            search_query = intent.get('search_query', final_query)
            web_results = self._web_search_tool(search_query)
            self.logger(f"[网络搜索] 找到 {len(web_results)} 条结果")

        if intent.get('need_calc', False):
            self.logger(f"[计算器] 正在计算...")
            calc_result = self._run_calculation(question, history_str, local_results, web_results, final_query)

        # 5. 多信息聚合和交叉验证
        if intent.get('need_local', False) or intent.get('need_web', False):
            # 如果有计算器结果，也要传递给聚合验证
            aggregated = self._aggregate_and_verify(local_results, web_results, final_query, calc_result)
        else:
            aggregated = {"aggregated_answer": calc_result or "无法处理该问题"}

        # 6. 生成最终回答
        sources = []
        if web_results:
            sources.extend([r['url'] for r in web_results])
        if local_results:
            sources.append("本地知识库")

        evidence_snippets = self._build_evidence_snippets(local_results, web_results)
        final_answer = self._generate_final_answer(aggregated, question, sources, history_str, is_follow_up,
                                                   calc_result, evidence_snippets=evidence_snippets)

        # 保存到历史
        self.chat_history.append((question, final_answer))

        return final_answer

    def get_rag_data(self, question: str):
        """专门为评估获取 RAG 数据 (Question, Contexts, Answer)"""
        # 1. 前置处理
        history_str = "" # 评估通常针对独立问题
        pre_enhanced = self._pre_enhancement(question, history_str)
        enhanced_query = pre_enhanced.get("enhanced_query", question)
        
        # 2. 意图分析
        intent = self._router_analyze(enhanced_query, history_str)
        
        # 3. 检索
        contexts = []
        if intent.get('need_local', False):
            results = self._local_knowledge_tool(enhanced_query)
            contexts.extend([r['content'] for r in results])
        
        if intent.get('need_web', False):
            results = self._web_search_tool(intent.get('search_query', enhanced_query))
            contexts.extend([r['content'] for r in results])
            
        # 4. 生成回答
        answer = self.answer_question(question)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }

    def run(self):
        self.logger("智能法律咨询助手已启动！")

        while True:
            query = input("请输入您的法律问题 (输入'退出'结束): ").strip()
            if query.lower() == "退出":
                break
            self.logger("----------------------------------------")
            self.logger(f"用户: {query}")
            self.logger("正在分析和处理...")
            answer = self.answer_question(query)
            self.logger(answer)
            self.logger("----------------------------------------")


if __name__ == "__main__":
    app = LegalAgent()
    app.run()

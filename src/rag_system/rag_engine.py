import os
import requests as _requests
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from config.config import Config
from chromadb.config import Settings


class LegalRAGSystem:
    """法律问答RAG系统"""

    def __init__(self, config: Config, logger=None):
        self.config = config
        self.logger = logger if logger else print
        self.embedding_model = None
        self.vector_store = None
        self.qa_chain = None
        
    def initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 检查是否使用BGE模型
            if "bge" in self.config.EMBEDDING_MODEL.lower():
                # 使用本地下载的BGE模型
                if "BAAI/bge-large-zh-v1.5" in self.config.EMBEDDING_MODEL:
                    # 使用本地模型路径
                    model_path = os.path.abspath("models/bge-large-zh-v1.5")
                    self.logger(f"正在使用本地BGE模型: {model_path}")
                    
                    self.embedding_model = HuggingFaceBgeEmbeddings(
                        model_name=model_path,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                else:
                    # 如果是其他BGE模型，尝试从网络下载
                    self.logger(f"正在从网络下载BGE模型: {self.config.EMBEDDING_MODEL}")
                    self.embedding_model = HuggingFaceBgeEmbeddings(
                        model_name=self.config.EMBEDDING_MODEL,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
            else:
                self.embedding_model = SentenceTransformerEmbeddings(
                    model_name=self.config.EMBEDDING_MODEL
                )
            self.logger("嵌入模型初始化成功")
        except Exception as e:
            self.logger(f"嵌入模型初始化失败: {e}")
            # 如果失败，尝试使用原始模型名
            try:
                self.logger("尝试使用原始模型名重试...")
                self.embedding_model = HuggingFaceBgeEmbeddings(
                    model_name=self.config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger("嵌入模型初始化成功")
            except Exception as e2:
                self.logger(f"重试失败: {e2}")
            
    def create_vector_store(self, documents: List[Dict[str, str]]):
        """
        创建或加载向量存储
        
        Args:
            documents (List[Dict[str, str]]): 文档列表
        """
        if not self.embedding_model:
            self.initialize_embeddings()
            
        # 检查是否已经存在向量存储
        try:
            self.vector_store = Chroma(
                persist_directory=self.config.VECTOR_DB_PATH,
                embedding_function=self.embedding_model,
                client_settings=Settings(anonymized_telemetry=False)
            )
            
            # 检查向量存储是否为空
            if len(self.vector_store.get()['ids']) > 0:
                self.logger(f"向量存储加载成功，已有 {len(self.vector_store.get()['ids'])} 个文档片段")
                return
            else:
                self.logger("现有向量存储为空，将重新创建...")
                
        except Exception as e:
            self.logger(f"加载现有向量存储失败，将重新创建: {e}")
            
        # 如果向量存储不存在或为空，则创建新的
        try:
            # 将文档转换为LangChain Document格式
            langchain_docs = []
            for doc in documents:
                # 分割文档内容
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.CHUNK_SIZE,
                    chunk_overlap=self.config.CHUNK_OVERLAP,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
                )
                
                # 创建文档对象
                doc_obj = Document(
                    page_content=doc['content'],
                    metadata={
                        'law_name': doc['law_name'],
                        'date': doc['date'],
                        'source': doc['source']
                    }
                )
                
                # 分割文档
                split_docs = text_splitter.split_documents([doc_obj])
                langchain_docs.extend(split_docs)
            
            # 创建向量存储
            self.vector_store = Chroma.from_documents(
                documents=langchain_docs,
                embedding=self.embedding_model,
                persist_directory=self.config.VECTOR_DB_PATH
            )
            self.vector_store.persist()
            self.logger(f"向量存储创建成功，共处理 {len(langchain_docs)} 个文档片段")
        except Exception as e:
            self.logger(f"向量存储创建失败: {e}")
    
    def setup_qa_chain(self):
        """设置问答链"""
        if not self.vector_store:
            self.logger("请先创建向量存储")
            return
            
        try:
            # 创建检索器
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.MAX_RESULTS}
            )
            
            # 注意：在当前实现中，我们直接使用通义千问模型进行问答，
            # 而不是通过LangChain的QA链
            self.logger("问答链设置成功")
        except Exception as e:
            self.logger(f"问答链设置失败: {e}")
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        搜索相似文档
        
        Args:
            query (str): 查询文本
            k (int): 返回结果数量
            
        Returns:
            List[Tuple[Document, float]]: 相似文档和相似度分数
        """
        if not self.vector_store:
            self.logger("请先创建向量存储")
            return []
            
        try:
            # 执行相似性搜索
            docs = self.vector_store.similarity_search_with_score(
                query, 
                k=k
            )
            return docs
        except Exception as e:
            self.logger(f"文档搜索失败: {e}")
            return []
    
    def answer_question_with_llm(self, question: str, context: str = "") -> str:
        """
        使用配置的 LLM 回答法律问题（通过 OpenAI 兼容接口）

        Args:
            question (str): 用户问题
            context (str): 相关法律条文上下文

        Returns:
            str: 回答内容
        """
        from prompts import load_prompt
        prompt = load_prompt("rag_answer", context_text=context, question=question)

        try:
            url = f"{self.config.LLM_BASE_URL}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.LLM_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
            }
            resp = _requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM 调用失败：{str(e)}"
    
    def answer_question(self, question: str) -> str:
        """
        回答法律问题
        
        Args:
            question (str): 用户问题
            
        Returns:
            str: 回答内容
        """
        # 搜索相关文档
        similar_docs = self.search_similar_documents(question)
        
        if not similar_docs:
            return "抱歉，我没有找到相关的法律条文来回答您的问题。"
        
        # 构建上下文
        context_parts = []
        for i, (doc, score) in enumerate(similar_docs, 1):
            if score > self.config.SIMILARITY_THRESHOLD:
                context_parts.append(f"《{doc.metadata.get('law_name', '未知法律')}》:")
                context_parts.append(f"{doc.page_content[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        # 使用 LLM 生成回答
        try:
            return self.answer_question_with_llm(question, context)
        except Exception:
            pass
        else:
            # 如果通义千问模型不可用，则使用原来的简单回答方式
            # 构建回答
            answer_parts = []
            answer_parts.append("根据相关法律法规，我的回答如下：\n")
            
            for i, (doc, score) in enumerate(similar_docs, 1):
                if score > self.config.SIMILARITY_THRESHOLD:
                    answer_parts.append(f"{i}. 来自《{doc.metadata.get('law_name', '未知法律')}》:")
                    answer_parts.append(f"   相关内容: {doc.page_content[:500]}...")
                    answer_parts.append(f"   相似度: {score:.2f}\n")
            
            # 添加免责声明
            answer_parts.append("\n请注意：以上信息仅供参考，具体法律问题建议咨询专业律师。")
            
            return "\n".join(answer_parts)

# 使用示例
if __name__ == "__main__":
    config = Config()
    rag_system = LegalRAGSystem(config)
    
    # 初始化嵌入模型
    rag_system.initialize_embeddings()
    
    # 注意：这里需要先加载文档并创建向量存储才能使用完整功能
    self.logger("法律RAG系统初始化完成")
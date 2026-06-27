import os
import sys
import time
from src.data_processing.document_processor import LegalDocumentProcessor
from src.rag_system.rag_engine import LegalRAGSystem
from config.config import Config

def initialize_vector_database():
    """初始化向量数据库"""
    print("开始初始化向量数据库...")
    print("="*50)
    
    try:
        # 初始化配置
        config = Config()
        
        # 初始化文档处理器
        print("1. 正在加载法律文档...")
        processor = LegalDocumentProcessor(config.DATABASE_PATH)
        documents = processor.load_documents()
        print(f"   [OK] 成功加载 {len(documents)} 个法律文档")
        
        if len(documents) == 0:
            print("   [!] 未找到任何法律文档，请检查Database目录")
            return False
            
        # 显示加载的文档信息
        print("   加载的文档列表:")
        for i, doc in enumerate(documents):
            print(f"     {i+1}. {doc['law_name']} ({doc['date']})")
        
        # 初始化RAG系统
        print("\n2. 正在初始化RAG系统...")
        rag_system = LegalRAGSystem(config)
        rag_system.initialize_embeddings()
        
        if not rag_system.embedding_model:
            print("   [X] 嵌入模型初始化失败")
            return False
            
        print("   [OK] 嵌入模型初始化成功")
        
        # 创建向量存储
        print("\n3. 正在创建向量存储 (这可能需要几分钟时间)...")
        start_time = time.time()
        rag_system.create_vector_store(documents)
        end_time = time.time()
        
        if not rag_system.vector_store:
            print("   [X] 向量存储创建失败")
            return False
            
        print(f"   [OK] 向量存储创建成功 (耗时: {end_time - start_time:.2f} 秒)")
        print(f"   [OK] 知识库已保存到: {config.VECTOR_DB_PATH}")
        
        print("\n" + "="*50)
        print("向量数据库初始化完成！")
        print("现在可以运行主程序进行法律咨询了。")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n[X] 初始化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = initialize_vector_database()
    if not success:
        sys.exit(1)
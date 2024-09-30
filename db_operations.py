from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any

class DBOperations:
    def __init__(self, persist_directory, embedding_model='paraphrase-multilingual-MiniLM-L12-v2'):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        print(f"Loaded Chroma DB from {persist_directory}")

    def similarity_search(self, query: str, top_k: int = 3, absolute_threshold: float = 20.0) -> List[Dict[str, Any]]:
        # 获取结果
        results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
        
        # 应用绝对阈值
        filtered_results = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results
            if score <= absolute_threshold
        ]
        
        # 按相似度排序并限制结果数量
        filtered_results.sort(key=lambda x: x['score'])
        return filtered_results[:top_k]

    def get_relevant_documents(self, query: str, top_k: int = 3, absolute_threshold: float = 20.0) -> List[Dict[str, Any]]:
        return self.similarity_search(query, top_k, absolute_threshold)

    def get_collection_stats(self) -> int:
        return len(self.vectorstore.get()['ids'])

if __name__ == "__main__":
    db_ops = DBOperations("/root/demo930/chroma_db")
    print(f"Total documents in collection: {db_ops.get_collection_stats()}")
    
    query = "在抽蓄电站设计过程中，水文方向一般考虑哪些要素？"
    results = db_ops.get_relevant_documents(query, top_k=5, absolute_threshold=20.0)
    print(f"\n查询: {query}")
    if results:
        for i, doc in enumerate(results, 1):
            print(f"\n文档 {i}:")
            print(f"内容: {doc['content'][:100]}...")
            print(f"来源: {doc['metadata'].get('source', 'Unknown')}")
            print(f"相似度分数: {doc['score']:.4f}")
    else:
        print("没有找到匹配的结果或所有结果的相似度低于阈值。")
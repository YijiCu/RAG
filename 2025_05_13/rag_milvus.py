import os
import logging
import json
import torch
from pymilvus import MilvusClient
from BCEmbedding import EmbeddingModel

class VectorRetrieval:
    def __init__(
        self,
        model_dir="/app/rag/modeldir/bce-embedding-base_v1",
        db_path="./milvus_db/vector.db",
        collection_name="kms",
        device="cuda:1",
        doclink_path="/app/rag/kb/kms/doclink"
    ):
        """初始化向量检索类"""
        logging.info("初始化向量检索系统...")
        self.device = device
        self.doclink_path = doclink_path
        
        try:
            # 设置当前设备
            device_id = int(self.device.split(':')[1])
            torch.cuda.set_device(device_id)
            
            # 加载模型
            self.model = EmbeddingModel(
                model_name_or_path=model_dir,
                device=device
            )
            logging.info(f"成功在设备 {self.device} 上初始化Embedding模型")
            
            # 连接数据库
            try:
                self.client = MilvusClient(uri=db_path)
                self.collection_name = collection_name
                
                if self.client.has_collection(self.collection_name):
                    self.client.load_collection(self.collection_name)
                    logging.info(f"Collection {self.collection_name} 加载成功")
                else:
                    raise ValueError(f"Collection {collection_name} 不存在!")
            except Exception as e:
                raise RuntimeError(f"连接数据库失败: {str(e)}")
                
        except Exception as e:
            logging.error(f"初始化失败: {str(e)}")
            raise

    def find_doc_url(self, filename):
        """根据文件名在doclink目录下查找对应的URL"""
        try:
            # 确保文件名是解码后的中文
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')
            
            link_file_path = os.path.join(self.doclink_path, f"{filename}.txt")
            logging.info(f"尝试读取link文件: {link_file_path}")
            
            if not os.path.exists(link_file_path):
                logging.warning(f"找不到对应的link文件: {link_file_path}")
                return None
                
            with open(link_file_path, 'r', encoding='utf-8') as f:
                url = f.read().strip()
                logging.info(f"成功读取URL for {filename}: {url}")
            
            return url
            
        except Exception as e:
            logging.error(f"读取文档URL失败 - 文件名: {filename}, 错误: {str(e)}")
            return None

    def search(self, query, limit=5):
        """执行向量检索并返回带URL的结果"""
        try:
            logging.info(f"开始执行向量检索 - 查询: {query}, 限制数量: {limit}")
            
            query_vector = self.model.encode([query])[0].tolist()
            logging.info("成功生成查询向量")
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=limit,
                output_fields=["text", "metadata"]
            )
            
            if not results or not results[0]:
                logging.warning("未找到相关结果")
                return None
                
            logging.info(f"检索到 {len(results[0])} 条结果")
            
            # 处理搜索结果
            for idx, result in enumerate(results[0], 1):
                logging.info(f"\n--- 结果 {idx} ---")
                
                # 记录原始结果
                logging.info(f"原始结果: {json.dumps(result, ensure_ascii=False)}")
                
                # 处理元数据
                metadata = result.get('entity', {}).get('metadata') if 'entity' in result else result.get('metadata')
                if metadata:
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            logging.warning(f"metadata解析失败: {metadata}")
                            continue
                            
                    logging.info(f"元数据: {json.dumps(metadata, ensure_ascii=False)}")
                    
                    # 获取文件名和URL
                    if 'source' in metadata:
                        filename = os.path.basename(metadata['source'])
                        url = self.find_doc_url(filename)
                        if url:
                            result['url'] = url
                            
                # 记录文本内容
                text = result.get('entity', {}).get('text') if 'entity' in result else result.get('text')
                if text:
                    text_preview = text[:200] + "..." if len(text) > 200 else text
                    logging.info(f"文本内容预览:\n{text_preview}")
            
            return results
                
        except Exception as e:
            logging.error(f"搜索过程出错: {str(e)}", exc_info=True)
            return None

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'client'):
            try:
                self.client.close()
                logging.info("数据库连接已关闭")
            except Exception as e:
                logging.error(f"关闭数据库连接时出错: {str(e)}")

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # 初始化检索系统
        retriever = VectorRetrieval()
        
        # 获取用户输入
        query = input("请输入搜索问题: ")
        
        # 执行搜索
        retriever.search(query)
        
    except Exception as e:
        logging.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()

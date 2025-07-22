import time
import tempfile
import os
import io
import requests
import json
import logging
from docx import Document
import pdfplumber
from rag_milvus import VectorRetrieval
import torch
import sseclient

class UnifiedAssistant:
    def __init__(self):
        """初始化统一助手"""
        # API配置
        self.base_url = "***" #外部展示时覆盖
        self.api_key = "***" #外部展示时覆盖
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 系统提示词
        self.prompt_template = """你是xxxxx的Deepseek办公助手，请根据用户上传材料（若有）以及企业内部知识（若有）对用户问题进行解答。
用户问题：{user_input}
用户上传材料：{user_file}
企业内部知识：{knowledge}
请回答："""
        
        # GPU设置
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(1)
                logging.info("已设置默认GPU为: cuda:1")
                
            # 初始化RAG系统
            self.retriever = VectorRetrieval(
                model_dir="/app/rag/modeldir/bce-embedding-base_v1",
                db_path="./milvus_db/vector.db",
                collection_name="kms",
                device="cuda:1"
            )
            self.rag_available = True
            logging.info("RAG系统初始化成功")
            
        except Exception as e:
            logging.error(f"初始化失败: {e}")
            self.rag_available = False

    def read_file(self, file):
        """统一的文件读取接口"""
        try:
            filename = getattr(file, 'name', 'Unknown')
            logging.info(f"开始处理文件: {filename}")
            
            # 获取文件扩展名
            file_ext = os.path.splitext(filename)[1].lower()
            
            # 根据文件类型选择相应的处理方法
            if file_ext == '.pdf':
                return self.read_pdf(file)
            elif file_ext == '.docx':
                return self.read_docx(file)
            elif file_ext == '.doc':
                return self.read_doc(file)
            elif file_ext == '.txt':
                return self.read_txt(file)
            else:
                logging.error(f"不支持的文件类型: {file_ext}")
                return None
                
        except Exception as e:
            logging.error(f"读取文件失败: {str(e)}")
            return None

    def read_pdf(self, file):
        """读取PDF文件内容"""
        try:
            # 获取文件内容
            if hasattr(file, 'getvalue'):
                pdf_data = io.BytesIO(file.getvalue())
            else:
                pdf_data = file
                
            text_content = []
            with pdfplumber.open(pdf_data) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(text)
            
            content = "\n".join(text_content)
            logging.info("PDF文件内容提取成功")
            return content
            
        except Exception as e:
            logging.error(f"读取PDF文件失败: {str(e)}")
            return None

    def read_doc(self, file):
        """读取旧版 Word (.doc) 文件内容"""
        try:
            import subprocess
            
            # 如果是上传的文件对象，需要先保存到临时文件
            if hasattr(file, 'getvalue'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
                    temp_file.write(file.getvalue())
                    temp_path = temp_file.name
            else:
                temp_path = file
                
            try:
                # 使用 antiword 提取文本
                result = subprocess.run(['antiword', temp_path], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"Antiword 处理失败: {result.stderr}")
                    return None
                    
                text = result.stdout
                
                # 清理和格式化文本
                text = text.replace('\r', '\n')
                text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
                
                logging.info("DOC文件内容提取成功")
                return text
                
            finally:
                # 如果使用了临时文件，需要删除它
                if hasattr(file, 'getvalue') and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logging.error(f"读取DOC文件失败: {str(e)}")
            return None

    def read_docx(self, file):
        """读取Word (.docx) 文件内容"""
        try:
            if hasattr(file, 'getvalue'):
                doc = Document(io.BytesIO(file.getvalue()))
            else:
                doc = Document(file)
            
            # 提取文本
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            content = "\n".join(text_content)
            logging.info("DOCX文件内容提取成功")
            return content
            
        except Exception as e:
            logging.error(f"读取DOCX文件失败: {str(e)}")
            return None

    def read_txt(self, file):
        """读取TXT文件内容"""
        try:
            # 获取文件内容
            if hasattr(file, 'getvalue'):
                content = file.getvalue().decode('utf-8')
            else:
                content = file.read().decode('utf-8')
            
            logging.info("TXT文件内容提取成功")
            return content
            
        except UnicodeDecodeError:
            try:
                # 如果UTF-8解码失败，尝试使用GBK
                if hasattr(file, 'getvalue'):
                    content = file.getvalue().decode('gbk')
                else:
                    content = file.read().decode('gbk')
                logging.info("TXT文件内容提取成功（GBK编码）")
                return content
            except Exception as e:
                logging.error(f"读取TXT文件失败: {str(e)}")
                return None
        except Exception as e:
            logging.error(f"读取TXT文件失败: {str(e)}")
            return None

    def truncate_text(self, text, max_length, add_ellipsis=True):
        """智能截断文本到指定长度，尽量保持完整句子和段落"""
        if not text or len(text) <= max_length:
            return text
        
        # 首先按段落分割
        paragraphs = text.split('\n')
        truncated = []
        current_length = 0
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # 如果单个段落就超过了最大长度
            if len(para) > max_length:
                # 按句子分割
                sentences = para.split('。')
                for sentence in sentences:
                    if current_length + len(sentence) + 1 > max_length:
                        break
                    truncated.append(sentence + '。')
                    current_length += len(sentence) + 1
            else:
                # 如果添加整个段落后不超过最大长度
                if current_length + len(para) + 1 <= max_length:
                    truncated.append(para)
                    current_length += len(para) + 1
                else:
                    break
        
        result = '\n'.join(truncated)
        if add_ellipsis and result != text:
            result = result.rstrip('。\n') + '...'
        
        return result

    def get_rag_knowledge(self, query, limit=5):
        """获取RAG检索结果"""
        if not self.rag_available:
            return None, []
            
        try:
            results = self.retriever.search(query, limit=limit)
            if not results or not results[0]:
                return None, []
                
            knowledge_parts = []
            doc_links = []
            
            for item in results[0]:
                try:
                    metadata = item.get('entity', {}).get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    similarity = 1 - item.get('distance', 0)
                    text = item.get('entity', {}).get('text', '')
                    
                    if metadata and text:
                        knowledge_parts.append(
                            f"相关度{similarity:.2f}的内容：\n{text}\n"
                            f"(来源：{metadata.get('title', '未知文档')})"
                        )
                        
                        if 'url' in item:
                            doc_links.append({
                                "title": metadata.get('title', '未知文档'),
                                "url": item['url']
                            })
                except Exception as e:
                    logging.warning(f"处理搜索结果时出错: {e}")
                    continue
            
            return "\n\n".join(knowledge_parts), doc_links
            
        except Exception as e:
            logging.error(f"RAG检索失败: {e}")
            return None, []

    def process_stream(self, response, streaming_callback=None):
        """处理流式响应"""
        try:
            full_response = ""
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        event_type = data.get("event")
                        
                        # 检查错误状态
                        if event_type == "error":
                            logging.error(f"Stream error: {data.get('message')}")
                            break
                        
                        # 检查消息结束事件
                        if event_type == "message_end":
                            logging.info("收到message_end事件，响应生成完成")
                            return full_response
                        
                        # 处理新的文本块
                        if event_type == "message" and "answer" in data:
                            chunk = data["answer"]
                            if chunk:  # 确保chunk不为空
                                full_response += chunk
                                if streaming_callback:
                                    streaming_callback(full_response)
                                
                    except json.JSONDecodeError:
                        continue
                        
            logging.info("事件流结束")
            return full_response
                    
        except Exception as e:
            logging.error(f"处理流式响应时出错: {str(e)}")
            return None

    def chat(self, user_input, doc_content=None, use_rag=False, streaming_callback=None):
        try:
            # 准备对话内容
            file_content = doc_content if doc_content else "无上传文件"
            knowledge, doc_links = "无相关知识", []
            
            if use_rag and self.rag_available:
                knowledge, doc_links = self.get_rag_knowledge(user_input)
                knowledge = knowledge if knowledge else "未找到相关知识"
            
            # 计算系统提示词和用户问题的长度
            base_prompt = self.prompt_template.format(
                user_input=user_input,
                user_file="",  # 临时空内容用于计算基础长度
                knowledge=""
            )
            base_length = len(base_prompt)
            MAX_LENGTH = 4400
            
            # 计算可用于文件内容的最大长度
            available_length = MAX_LENGTH - base_length - 100  # 预留一些空间给格式化字符
            if file_content and file_content != "无上传文件":
                # 使用更大的限制，因为没有使用RAG
                if not use_rag:
                    available_length = min(available_length, 4000)  # 给文件内容分配更多空间
                file_content = self.truncate_text(file_content, available_length)
                logging.info(f"处理后的文件内容长度: {len(file_content)}")
            
            # 组装最终提示词
            prompt = self.prompt_template.format(
                user_input=user_input,
                user_file=file_content,
                knowledge=knowledge
            )
            
            logging.info(f"最终提示词长度: {len(prompt)}")
            
            # 准备请求
            payload = {
                "inputs": {},
                "query": prompt,
                "response_mode": "streaming",
                "user": "test_user_1"
            }
            
            # 发送请求时修改超时设置
            with requests.post(
                f"{self.base_url}/chat-messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=(30, 120)  # 连接超时30秒，读取超时120秒
            ) as response:
                response.raise_for_status()
                
                # 处理响应
                full_response = self.process_stream(response, streaming_callback)
                if full_response:  # 改为检查响应内容是否为空
                    logging.info(f"获取响应成功，长度: {len(full_response)}")
                    return True, full_response, doc_links
                
                logging.error("未能获取有效响应")
                return False, "生成回答失败", []
                
        except requests.Timeout:
            error_msg = "请求超时"
            logging.error(error_msg)
            return False, error_msg, []
        except Exception as e:
            error_msg = f"聊天过程出错: {str(e)}"
            logging.error(error_msg)
            return False, error_msg, []

    def cleanup(self):
        """清理资源"""
        if self.rag_available and hasattr(self, 'retriever'):
            try:
                self.retriever.cleanup()
                logging.info("资源清理完成")
            except Exception as e:
                logging.error(f"清理资源时出错: {e}")

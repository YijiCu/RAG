import os
import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.schema.runnable import RunnablePassthrough
from db_operations import DBOperations
from typing import Any, List, Optional
from pydantic import Field, BaseModel

# 自定义LLM类，用于封装InternLM2模型
class InternLM2LLM(LLM):
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    # 实现模型推理逻辑
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if stop:
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text[:generated_text.index(s)]

        return generated_text[len(prompt):]

    @property
    def _identifying_params(self) -> dict:
        return {"name": "InternLM2LLM"}

    @property
    def _llm_type(self) -> str:
        return "internlm2"

# RAG模型主类
class RAGModel:
    def __init__(self, model_dir, chroma_db_dir, embedding_model='paraphrase-multilingual-MiniLM-L12-v2'):
        # 设置环境变量
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
        os.environ['TRUST_REMOTE_CODE'] = '1'

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # 初始化数据库操作
        self.db_ops = DBOperations(chroma_db_dir, embedding_model)
        print(f"Loaded Chroma DB from {chroma_db_dir}")

        # 创建LLM实例
        self.llm = InternLM2LLM(self.model, self.tokenizer)

        # 定义提示模板
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""#Role: 上海院清洁能源行业知识专家

    ##Profile: 你是上海院(上海勘测设计研究院）的知识专家，专门负责回答清洁能源相关问题。你的回答应该基于提供的背景知识，并结合你的专业理解。

    ##Rules: Rules: 请绝对遵守Rules中的规则。
    1. 仔细阅读背景知识并回答最相关的信息，请不要回答你判断认为不相关的背景知识。
    2. 请只回答一次，不要重复回答，回答中不要提到“背景知识”。

    ##背景知识：
    {context}

    ##用户问题：
    {query}

    ##回答："""
        )

        # 创建问答链
        self.qa_chain = self.prompt_template | self.llm

        # 初始化缓存
        self.cache = {}

    # 生成响应的方法
    def generate_response(self, query, top_k=3, score_threshold=0.85):
        # 检查缓存
        if query in self.cache:
            print("使用缓存的响应")
            return self.cache[query]

        # 检索相关文档
        relevant_docs = self.db_ops.get_relevant_documents(query, top_k, absolute_threshold=10.0)
        if not relevant_docs:
            return "抱歉，我没有找到与您问题相关的足够准确的信息。请尝试重新表述您的问题或询问其他方面的问题。"

        # 构建上下文
        context = "\n".join([f"内容: {doc['content']}\n相似度: {doc['score']:.4f}" for doc in relevant_docs])
        
        print(f"检索到 {len(relevant_docs)} 个相关文档")
        for doc in relevant_docs:
            print(f"相似度: {doc['score']:.4f}, 内容: {doc['content'][:100]}...")

        # 生成响应
        response = self.qa_chain.invoke({"context": context, "query": query})
        print(f"LLM 响应: {response}")

        # 缓存响应
        self.cache[query] = response

        return response

    # 异步流式生成响应的方法
    async def generate_response_stream(self, query, top_k=3, score_threshold=0.85):
        if query in self.cache:
            response = self.cache[query]
        else:
            relevant_docs = self.db_ops.get_relevant_documents(query, top_k, absolute_threshold=10.0)
            if not relevant_docs:
                response = "抱歉，我没有找到与您问题相关的足够准确的信息。请尝试重新表述您的问题或询问其他方面的问题。"
            else:
                context = "\n".join([f"内容: {doc['content']}\n相似度: {doc['score']:.4f}" for doc in relevant_docs])
                response = self.qa_chain.invoke({"context": context, "query": query})
                self.cache[query] = response

        # 延迟2秒
        await asyncio.sleep(2)

        # 逐字输出
        for token in response:
            yield token
            await asyncio.sleep(1/50) 

if __name__ == "__main__":
    model_dir = "/root/demo930/modeldir/internlm2_5-20b-chat-w4a16-4bit"
    chroma_db_dir = "/root/demo930/chroma_test"
    rag_model = RAGModel(model_dir, chroma_db_dir)

    # 测试流式输出
    async def test_stream():
        query = input("\n请输入您的清洁能源相关问题：")
        print("\n清洁能源专家回答：")
        async for chunk in rag_model.generate_response_stream(query):
            print(chunk, end='', flush=True)
        print("\n")

    asyncio.run(test_stream())
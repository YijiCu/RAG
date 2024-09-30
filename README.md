# 基于InternLM2.5的清洁能源行业知识专家 #1408
上海AI Lab书生·浦语训练营第三期项目

团队名称：还没想好捏

**本项目基于InternLM实现清洁能源行业知识专家与 InternLM Tutorial 的链接。** 

**https://github.com/InternLM/Tutorial** 

欢迎大家加入InternLM大家庭一起学习大模型！！作为一名从0开始的小白，在InternLM课程的帮助下我一步步也终于是成功做出了一点项目成果！期待更多的人来这里一起提升自己！

### 项目简介
智能风电专家 —— 基于RAG的行业知识大模型
本项目致力于打造一个智能、高效的风电行业知识专家系统，通过结合检索增强生成(RAG)技术和大语言模型，为用户提供准确、专业的风电领域知识服务。项目主要特点如下：
 🌟 基于InternLM构建的大模型与检索系统深度融合
 🚀 采用RAG技术，实现快速知识更新和模型迭代
 📚 整合500份专业风电领域知识，构建高质量知识库
 🎯 优化召回算法，确保高准确率和相关性
 ⚡ 利用开源工具栈，实现快速响应和高性能
 💼 专注于风电行业应用，提供精准的专业知识问答
 
本项目通过创新的RAG架构，有效解决了传统AI模型在专业领域知识更新和维护方面的挑战。通过简单更换底座大模型，即可快速适应AI技术的迭代，大幅降低企业在AI应用层的成本和维护难度。
我们的目标是为风电行业提供一个智能、可靠的知识助手，助力工程师们更高效地获取和应用专业知识。

### 项目地址
https://github.com/YijiCu/RAG

### 视频地址
https://www.bilibili.com/video/BV1FfWde5EQL/

### 项目架构图
![image](https://github.com/YijiCu/New_energy_llm/blob/main/config/%E9%A1%B9%E7%9B%AE%E6%9E%B6%E6%9E%84%E5%9B%BENEL.png)

### 项目运行图
![image](https://github.com/YijiCu/New_energy_llm/blob/main/config/%E6%95%88%E6%9E%9C%E5%9B%BE1.png)
![image](https://github.com/YijiCu/New_energy_llm/blob/main/config/%E6%95%88%E6%9E%9C%E5%9B%BE2.png)

### 项目文件介绍

1. **chroma_db/**
   - 存放知识库文件，为向量化处理后的知识库文件。
   - 使用开源向量数据库工具 [Chroma](https://docs.trychroma.com/)。
   - 注意：由于 GitHub 大小限制，处理后的文件无法上传，请使用 `getdatabase.py` 脚本自行处理数据库文件。

2. **config/**
   - 存放项目的图片文件。

3. **getdatabase.py**
   - 处理数据库文件脚本。
   - 使用说明：
     - 请自行下载数据库文件，并在脚本中提供文件路径。
     - 使用的 Embedding 模型：paraphrase-multilingual-MiniLM-L12-v2。
   - 功能：
     - 支持处理 .txt/.docx/.pptx/.pdf 格式的文本。
     - 自动清除乱码文本。
     - 暂不支持 .doc、.ppt 和扫描版 .pdf 文件处理（后续会更新）。
   - 当前切块方式：
     - 对整个文档进行简单切段处理，1000个 token 为一段，前后各 200 token 的重复区间。
     - 后续计划更新更智能的文本切段处理方法。

4. **db_operations.py**
   - 数据库操作核心脚本。
   - 负责知识库的检索、向量化、更新等操作。

5. **rag_model.py**
   - RAG 模型核心脚本。
   - 功能：
     - 负责模型与向量知识库的链接、推理等操作。
     - 支持更换底座模型（需相应配置）。
   - 使用的大模型：
     - 底座：InternLM2.5-20B-chat
     - 优化：使用 LMDeploy 进行 4-bit 精度量化，提高响应速度。
   - 特性：
     - 增加缓存功能，减少推理时间。
     - 使用 asyncio 预设流式输出功能，模拟流式输出效果。

6. **chatui.py**
   - 界面脚本，负责界面的搭建和交互操作。

## 如何运行

1. 创建虚拟环境并安装依赖：
   ```
   conda create -n [your_env_name] python=3.11
   conda activate [your_env_name]
   pip install -r requirements.txt
   ```

2. 处理数据库文件：
   ```
   python getdatabase.py
   ```

3. 运行聊天界面：
   ```
   streamlit run chatui.py
   ```

4. 打开开发机返回的端口连接，即可开始测试。

## 欢迎交流！

如有任何问题或建议，欢迎随时与我们联系。
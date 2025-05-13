import os
import logging
import streamlit as st
from PIL import Image
import base64
from model_api import UnifiedAssistant

class UIConfig:
    """UI配置类"""
    PAGE_TITLE = "xxx大模型AI办公助手"
    PAGE_ICON = "🔍"
    BACKGROUND_COLOR = "#F3F6FB"
    USER_ICON_PATH = '/app/rag/config/提问.png'
    AI_ICON_PATH = '/app/rag/config/模型回复机器人.png'
    DEEPSEEK_ICON_PATH = '/app/rag/config/小钢笔.png'
    SHANGHAI_ICON_PATH = '/app/rag/config/书本.png'
    WELCOME_MESSAGE = "xxx小伙伴您好，请输入您的问题，我可以帮您进行报告分析、生成等各种办公作业，并且我还在持续更新！"
    
    # 选中状态
    SELECTED_BG_COLOR = "#3F70FF"
    SELECTED_TEXT_COLOR = "#fff"
    SELECTED_FONT_WEIGHT = "bold" 
    SELECTED_FONT_SIZE = "18px"
    
    # 未选中状态
    UNSELECTED_BG_COLOR = "#fff"
    UNSELECTED_TEXT_COLOR = "#4D4B72"
    UNSELECTED_FONT_WEIGHT = "normal"
    UNSELECTED_FONT_SIZE = "18px"
    
    # 思考文字样式
    THINKING_BG_COLOR = "#FAFDFF"
    THINKING_TEXT_COLOR = "#8b8b8b"
    THINKING_FONT_SIZE = "14px"
    THINKING_BORDER_COLOR = "#D5DBDE"

    @staticmethod
    def load_image(path: str):
        """加载图片"""
        try:
            if os.path.exists(path):
                return Image.open(path)
            return None
        except Exception as e:
            logging.error(f"Error loading image from {path}: {str(e)}")
            return None

def initialize_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 修改点1：默认为DeepSeek模式，对应rag_enabled为False
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "deepseek"  # 可选值: "deepseek" 或 "shanghai_rag"
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = UnifiedAssistant()
            logging.info("UnifiedAssistant 初始化成功")
        except Exception as e:
            logging.error(f"UnifiedAssistant 初始化失败: {e}")
            st.error("系统初始化失败，请刷新页面重试")

def setup_page():
    """设置页面配置"""
    st.set_page_config(
        page_title=UIConfig.PAGE_TITLE,
        page_icon=UIConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # 使用h1标签直接设置标题，以便更精确控制样式
    st.markdown(f"<h1 style='margin-bottom: 0; font-size: 28px;'>{UIConfig.PAGE_TITLE}</h1>", unsafe_allow_html=True)
    st.caption("基于DeepSeek R1定制化开发")

def setup_background(background_color):
    """设置背景样式"""
    st.markdown(
        f"""
        <style>
        /* 全局背景色 */
        .stApp {{
            background-color: {background_color} !important;
        }}
        
        /* 标题样式 */
        .block-container > div:first-child h1 {{
            margin-bottom: 0 !important;
            margin-top: 0 !important;
            padding-top: 20px !important;
            padding-bottom: 5px !important;
        }}
        
        /* 输入框样式 */
        .stChatInput,
        .stChatInput > div,
        .stChatInput input {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 20px !important;
        }}
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {{
            background-color: #fafafa !important;
            border-right: 1px solid #eee !important;
        }}
        
        /* 聊天气泡样式 */
        .st-emotion-cache-1c7y2kd,
        [data-testid="stChatMessage"] {{    
            background-color: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px !important;
            margin-bottom: 15px !important;
        }}
        
        /* 用户消息气泡 */
        [data-testid="stChatMessageContent"][data-message-type="user"] {{
            background-color: white !important;
        }}
        
        /* AI响应气泡 */
        [data-testid="stChatMessageContent"][data-message-type="assistant"] {{
            background-color: rgba(227, 237, 250, 0.5) !important;
        }}
        
        details[class*="st-emotion-cache"] {{
            background-color: rgba(227, 237, 250, 0.5) !important;
        }}
        
        /* 思考文字样式 */
        .thinking-content {{
            background-color: {UIConfig.THINKING_BG_COLOR};
            color: {UIConfig.THINKING_TEXT_COLOR};
            padding: 10px;
            border-radius: 5px;
            font-size: {UIConfig.THINKING_FONT_SIZE};
            border-left: 5px solid {UIConfig.THINKING_BORDER_COLOR};
            margin: 10px 0;
        }}
        
        /* 侧边栏标题样式 */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            font-size: 18px !important;
            color: #333;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }}
        
        /* 修改底部输入框样式 */
        .stChatInput {{
            border: 1px solid #e0e0e0 !important;
            margin-bottom: 20px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def setup_sidebar():
    """设置侧边栏"""
    with st.sidebar:
        st.title("📚 工作台")
        
def setup_sidebar():
    """设置侧边栏"""
    with st.sidebar:
        # 增大工作台标题字体
        st.markdown('<h1 style="font-size: 24px;">📚 工作台</h1>', unsafe_allow_html=True)
        
        # 模式选择按钮，上下布局，使用小齿轮图标
        st.markdown("### ⚙️ 模式选择")
        
        # DeepSeek办公助手按钮 - 默认选中，添加钢笔图标
        deepseek_selected = st.session_state.current_mode == "deepseek"
        if st.button(
            "🖋️ DeepSeek AI办公助手", 
            key="deepseek_button",
            use_container_width=True,
            type="primary" if deepseek_selected else "secondary"
        ):
            if not deepseek_selected:  # 只有当前未选中时才触发状态变更
                st.session_state.current_mode = "deepseek"
                st.session_state.rag_enabled = False
                st.rerun()
            
        # 上海院知识库问答助手按钮，添加书本图标
        shanghai_selected = st.session_state.current_mode == "shanghai_rag"
        if st.button(
            "📚 上海院知识库问答助手", 
            key="shanghai_button",
            use_container_width=True,
            type="primary" if shanghai_selected else "secondary"
        ):
            if not shanghai_selected:  # 只有当前未选中时才触发状态变更
                st.session_state.current_mode = "shanghai_rag"
                st.session_state.rag_enabled = True
                st.rerun()
        
        # 设置按钮样式
        st.markdown(f"""
        <style>
        /* 模式选择按钮样式 */
        /* 选中状态 - 蓝色背景 */
        .stButton [data-testid="baseButton-primary"] {{
            background-color: {UIConfig.SELECTED_BG_COLOR} !important;
            color: {UIConfig.SELECTED_TEXT_COLOR} !important;
            font-weight: {UIConfig.SELECTED_FONT_WEIGHT} !important;
            font-size: {UIConfig.SELECTED_FONT_SIZE} !important;
            border-radius: 5px !important;
            border: none !important;
            padding: 10px !important;
            min-height: 50px !important;
            box-shadow: none !important;
            transition: all 0.3s ease !important;
        }}
        
        /* 未选中状态 - 白色背景 */
        .stButton [data-testid="baseButton-secondary"] {{
            background-color: {UIConfig.UNSELECTED_BG_COLOR} !important;
            color: {UIConfig.UNSELECTED_TEXT_COLOR} !important;
            font-weight: {UIConfig.UNSELECTED_FONT_WEIGHT} !important;
            font-size: {UIConfig.UNSELECTED_FONT_SIZE} !important;
            border-radius: 5px !important;
            border: none !important;
            padding: 10px !important;
            min-height: 50px !important;
            box-shadow: none !important;
            transition: all 0.3s ease !important;
        }}
        
        /* 确保按钮内的图标和文字正确对齐 */
        .stButton button div {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }}
        
        /* 增大侧边栏中的所有标题 */
        [data-testid="stSidebar"] h3 {{
            font-size: 20px !important;
            margin-top: 20px !important;
            margin-bottom: 10px !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # 文件处理
        st.markdown("### 📁 文件处理")
        
        # 文件上传器的key要保持一致
        uploaded_file = st.file_uploader(
            "上传文件",
            type=["txt", "pdf", "doc", "docx"],
            help="上传文件进行问答",
            key="file_uploader"  # 添加固定的key
        )
        
        # 处理文件上传和移除
        if uploaded_file is not None:
            # 检查是否需要重新处理文件
            current_file_name = getattr(uploaded_file, 'name', None)
            if 'last_processed_file' not in st.session_state or \
               st.session_state.last_processed_file != current_file_name:
                
                st.write("文件信息：")
                st.write(f"- 文件名: {uploaded_file.name}")
                st.write(f"- 文件类型: {uploaded_file.type}")
                st.write(f"- 文件大小: {uploaded_file.size / 1024:.2f} KB")
                
                if hasattr(st.session_state, 'assistant'):
                    # 处理文件内容
                    doc_content = st.session_state.assistant.read_file(uploaded_file)
                    if doc_content:
                        st.session_state.current_doc_content = doc_content
                        st.session_state.last_processed_file = current_file_name
                        st.success("文件处理成功！")
                    else:
                        st.error("文件处理失败")
        else:
            # 文件被移除时清理相关状态
            if 'current_doc_content' in st.session_state:
                del st.session_state.current_doc_content
            if 'last_processed_file' in st.session_state:
                del st.session_state.last_processed_file
        
        # 清除历史按钮 - 使用独立样式，不受模式选择影响
        st.markdown("### 🗑️ 清除历史")
        if st.button("清除对话历史", key="clear_history_button", use_container_width=True):
            st.session_state.messages = []
            # 清除文件相关的所有状态
            if 'current_doc_content' in st.session_state:
                del st.session_state.current_doc_content
            if 'last_processed_file' in st.session_state:
                del st.session_state.last_processed_file
            st.rerun()
            
        # 添加清除历史按钮的独立样式
        st.markdown("""
        <style>
        /* 清除历史按钮样式 - 蓝色背景但独立控制 */
        button:has(div:contains("清除对话历史")) {
            background-color: #3F70FF !important;
            color: white !important;
            border: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

def handle_file_change():
    """处理文件变更的回调函数"""
    if 'current_doc_content' in st.session_state:
        del st.session_state.current_doc_content
        logging.info("文件已移除，清除文档内容")

def merge_doc_links(doc_links):
    """合并来自同一文档的链接"""
    if not doc_links or not isinstance(doc_links, list):
        return []
        
    # 使用字典来合并相同文档的链接
    merged = {}
    for doc in doc_links:
        if isinstance(doc, dict):
            title = doc.get('title')
            url = doc.get('url')
            if title and url:
                # 如果这个文档已经存在，跳过（只保留第一次出现）
                if title not in merged:
                    merged[title] = url
                    
    # 转换回列表格式
    return [{"title": title, "url": url} for title, url in merged.items()]

def display_chat_history():
    """显示聊天历史"""
    # 显示欢迎消息
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar=UIConfig.load_image(UIConfig.AI_ICON_PATH)):
            st.write(UIConfig.WELCOME_MESSAGE)
            
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(
            message["role"],
            avatar=UIConfig.load_image(UIConfig.USER_ICON_PATH if message["role"] == "user" else UIConfig.AI_ICON_PATH)
        ):
            # 处理思考内容的显示格式
            content = message["content"]
            if message["role"] == "assistant" and "<think>" in content and "</think>" in content:
                try:
                    # 分离思考内容和普通内容
                    parts = content.split("<think>", 1)  # 只分割第一个出现的标签
                    before_think = parts[0]
                    
                    if len(parts) > 1:
                        remaining = parts[1].split("</think>", 1)  # 只分割第一个出现的结束标签
                        think_content = remaining[0]
                        after_think = remaining[1] if len(remaining) > 1 else ""
                        
                        # 显示格式化后的内容
                        if before_think.strip():
                            st.markdown(before_think)
                        
                        # 显示思考内容（使用自定义CSS样式）
                        if think_content.strip():
                            st.markdown(f'<div class="thinking-content">{think_content}</div>', unsafe_allow_html=True)
                        
                        if after_think.strip():
                            st.markdown(after_think)
                    else:
                        # 如果分割失败，显示原始内容
                        st.markdown(content)
                except Exception as e:
                    # 如果处理过程中出错，回退到显示原始内容
                    logging.error(f"处理思考内容时出错: {str(e)}")
                    st.markdown(content)
            else:
                st.markdown(content)
            
            # 如果是助手消息且有相关文档，显示文档链接
            if message["role"] == "assistant" and "relevant_docs" in message:
                docs = merge_doc_links(message["relevant_docs"])
                if docs:
                    with st.expander("📑 参考文档来源", expanded=True):
                        for idx, doc in enumerate(docs, 1):
                            st.markdown(f"**文档 {idx}**: [{doc['title']}]({doc['url']})")

def handle_user_input():
    """处理用户输入"""
    if not hasattr(st.session_state, 'assistant'):
        st.error("系统未正确初始化，请刷新页面重试")
        return

    if prompt := st.chat_input("请输入您的问题"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 显示用户消息
        with st.chat_message("user", avatar=UIConfig.load_image(UIConfig.USER_ICON_PATH)):
            st.markdown(prompt)
        
        # 创建新的助手消息容器
        with st.chat_message("assistant", avatar=UIConfig.load_image(UIConfig.AI_ICON_PATH)):
            try:
                # 创建消息占位符和等待提示占位符
                message_placeholder = st.empty()
                waiting_placeholder = st.empty()
                
                # 在等待占位符中显示等待提示
                with waiting_placeholder:
                    waiting_placeholder.markdown("🤔 排队等待回答中...")
                
                # 用一个变量来标记是否已经开始接收响应
                response_started = False
                
                def streaming_callback(text):
                    nonlocal response_started
                    if not response_started:
                        # 清除等待提示
                        waiting_placeholder.empty()
                        response_started = True
                    
                    # 处理思考内容的显示格式
                    try:
                        if "<think>" in text and "</think>" in text:
                            # 分离思考内容和普通内容
                            parts = text.split("<think>", 1)  # 只分割第一个出现的标签
                            before_think = parts[0]
                            
                            if len(parts) > 1:
                                remaining = parts[1].split("</think>", 1)  # 只分割第一个出现的结束标签
                                think_content = remaining[0]
                                after_think = remaining[1] if len(remaining) > 1 else ""
                                
                                # 显示格式化后的内容
                                html_content = ""
                                if before_think.strip():
                                    html_content += before_think
                                
                                # 显示思考内容（使用自定义CSS样式）
                                if think_content.strip():
                                    html_content += f'<div class="thinking-content">{think_content}</div>'
                                
                                if after_think.strip():
                                    html_content += after_think
                                    
                                message_placeholder.markdown(html_content, unsafe_allow_html=True)
                            else:
                                # 如果分割失败，显示原始内容
                                message_placeholder.markdown(text)
                        else:
                            message_placeholder.markdown(text)
                    except Exception as e:
                        # 如果处理过程中出错，回退到显示原始内容
                        logging.error(f"处理流式思考内容时出错: {str(e)}")
                        message_placeholder.markdown(text)
                
                # 处理文档内容和RAG
                doc_content = st.session_state.get('current_doc_content')
                success, full_response, doc_links = st.session_state.assistant.chat(
                    prompt,
                    doc_content=doc_content,
                    use_rag=st.session_state.rag_enabled,
                    streaming_callback=streaming_callback
                )
                
                # 确保等待提示被清除
                waiting_placeholder.empty()
                
                # 显示完整响应
                if success:
                    # 处理思考内容的显示格式（确保在流式显示结束后也能正确显示）
                    try:
                        if "<think>" in full_response and "</think>" in full_response:
                            # 分离思考内容和普通内容
                            parts = full_response.split("<think>", 1)  # 只分割第一个出现的标签
                            before_think = parts[0]
                            
                            if len(parts) > 1:
                                remaining = parts[1].split("</think>", 1)  # 只分割第一个出现的结束标签
                                think_content = remaining[0]
                                after_think = remaining[1] if len(remaining) > 1 else ""
                                
                                # 显示格式化后的内容
                                html_content = ""
                                if before_think.strip():
                                    html_content += before_think
                                
                                # 显示思考内容（使用自定义CSS样式）
                                if think_content.strip():
                                    html_content += f'<div class="thinking-content">{think_content}</div>'
                                
                                if after_think.strip():
                                    html_content += after_think
                                    
                                message_placeholder.markdown(html_content, unsafe_allow_html=True)
                            else:
                                # 如果分割失败，显示原始内容
                                message_placeholder.markdown(full_response)
                        else:
                            message_placeholder.markdown(full_response)
                    except Exception as e:
                        # 如果处理过程中出错，回退到显示原始内容
                        logging.error(f"处理最终思考内容时出错: {str(e)}")
                        message_placeholder.markdown(full_response)
                    
                    # 合并并显示文档链接
                    merged_doc_links = merge_doc_links(doc_links)
                    if merged_doc_links:
                        with st.expander("📑 参考文档来源", expanded=True):
                            for idx, doc in enumerate(merged_doc_links, 1):
                                st.markdown(f"**文档 {idx}**: [{doc['title']}]({doc['url']})")
                    
                    # 保存消息到历史记录（保存合并后的文档链接）
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "relevant_docs": merged_doc_links
                    })
                else:
                    st.error(full_response)

            except Exception as e:
                error_msg = f"处理出错: {str(e)}"
                logging.error(error_msg)
                st.error(error_msg)

def main():
    """主函数"""
    try:
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 初始化会话状态
        initialize_session_state()
        
        # 设置页面
        setup_page()
        
        # 设置背景：修改点2，使用背景色值而非背景图片
        setup_background(UIConfig.BACKGROUND_COLOR)
        
        # 设置侧边栏
        setup_sidebar()
        
        # 显示聊天历史
        display_chat_history()
        
        # 处理用户输入
        handle_user_input()
        
    except Exception as e:
        logging.error(f"程序运行出错: {str(e)}")
        st.error("系统出错，请刷新页面重试")

if __name__ == "__main__":
    main()

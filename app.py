import streamlit as st
import os
import json
import time
import base64
from PIL import Image
import numpy as np
import asyncio
from rag_model import RAGModel

st.set_page_config(page_title="清洁能源行业知识专家", page_icon="🦜🔗")
st.title("清洁能源行业知识专家")
st.caption("清洁能源行业勘测设计领域垂类大模型")

#初始化大模型
@st.cache_resource
def initialize_rag_model():
    model_dir = "./modeldir/internlm2_5-20b-chat-w4a16-4bit"
    chroma_db_dir = "./chroma_db"
    return RAGModel(model_dir, chroma_db_dir)

rag_model = initialize_rag_model()

#设置背景图片
background_path = "./config/wallpaper.png"
if os.path.exists(background_path):
    st.markdown(
        f"""
        <style>
        [data-testid="stApp"]{{
            background: url(data:image/png;base64,{base64.b64encode(open(background_path, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(f"Background image not found at {background_path}")

# 加载ui图标
user_icon_path = './config/工程师.png'
ai_icon_path = './config/风机.png'

if os.path.exists(user_icon_path) and os.path.exists(ai_icon_path):
    usericon = Image.open(user_icon_path)
    aiicon = Image.open(ai_icon_path)
else:
    st.warning("Icon images not found. Using default icons.")
    usericon = None
    aiicon = None

async def chat_ui():
    state = st.session_state
    
    if "message_history" not in state:
        state.message_history = []
    
    if "messages" not in state:
        st.session_state.messages = [{"role": "assistant", "avatar": np.array(aiicon) if aiicon else None, "content": "工程师您好，请输入您的问题！"}]
    
    for message in state.message_history:
        if message["role"] != "system":
            avatar = np.array(aiicon) if message["role"] == "assistant" and aiicon else np.array(usericon) if usericon else None
            st.chat_message(message["role"], avatar=avatar).write(message["content"])

    user_input = st.chat_input("请输入您的问题")

    if user_input:
        state.message_history.append({"role": "user", "content": user_input})
        st.chat_message("user", avatar=np.array(usericon) if usericon else None).write(user_input)

        with st.chat_message("assistant", avatar=np.array(aiicon) if aiicon else None):
            message_placeholder = st.empty()
            full_response = ""

            async for token in rag_model.generate_response_stream(user_input):
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        state.message_history.append({"role": "assistant", "content": full_response})

def init_chatbot():
    state = st.session_state
    if "message_history" not in state:
        state.message_history = []
        welcome_message = "工程师您好，请输入您的问题！"
        state.messages = [{"role": "assistant", "avatar": np.array(aiicon) if aiicon else None, "content": welcome_message}]
        state.message_history.append({"role": "assistant", "content": welcome_message})

if __name__ == "__main__":
    init_chatbot()
    asyncio.run(chat_ui())
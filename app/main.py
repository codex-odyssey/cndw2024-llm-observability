import os, uuid, logging
from agent.agent import Agent

import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere

from traceloop.sdk import Traceloop


logger = logging.getLogger(name=__name__)
Traceloop.init(
    # デモ用なので、batch processorではなく即時でトレースデータを送る
    disable_batch=True,
    # アプリケーションの名前
    app_name="CNDW2024 Session Bot",
    # 独自属性の追加
    resource_attributes={"env": "demo", "version": "1.0.0"},
)

openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

st.title("❄️ CNDW2024 Session Bot ❄️")
st.caption(
    """
CNDW2024のセッションをデータソースとして使用したアプリケーションです。
関心のあるセッションを探す際や気になるセッションの概要を知りたい時に便利に使えます。
"""
)

# サイドバー関連
with st.sidebar.container():
    with st.sidebar:
        st.sidebar.markdown("### LLM関連パラメータ")
        model_name = st.sidebar.selectbox(
            label="Model Name",
            options=["gpt-4o-mini", "command-r-plus"],
        )
        max_tokens = st.sidebar.slider(
            label="Max Tokens",
            min_value=128,
            max_value=2048,
            value=1024,
            step=128,
            help="LLMが出力する最大のトークン長",
        )
        temperature = st.sidebar.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="モデルの出力のランダム性",
        )
        st.sidebar.markdown("### 検索関連パラメータ")
        top_k = st.sidebar.slider(
            label="Top K",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="関連情報の取得数",
        )
        use_reranker = st.sidebar.radio(
            label="Reranker",
            options=[True, False],
            horizontal=True,
            help="Rerankerを使用するか（※使用には、CohereのAPI Keyが必要です）",
        )
        top_n = st.sidebar.slider(
            label="Top N",
            min_value=1,
            max_value=10,
            value=3,
            help="Rerankerで取得した情報を何件に絞り込むか",
        )

if model_name == "gpt-4o-mini":
    chat_model = ChatOpenAI(
        api_key=openai_api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
elif model_name == "command-r-plus":
    chat_model = ChatCohere(
        cohere_api_key=cohere_api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
else:
    logger.error("Unsetted model name")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("何が聞きたいですか？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    agent = Agent(llm=chat_model)
    with st.chat_message("assistant"):
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        if "run_id" not in st.session_state:
            st.session_state["run_id"] = str(uuid.uuid4())
        #stream = chain.stream(input=prompt, config={"run_id": st.session_state["run_id"]})
        result = agent.run(question=prompt)
        logger.warning(result)
        #response = st.write_stream(stream=stream)
        st.markdown(result.content)
        logger.warning(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": result.content})

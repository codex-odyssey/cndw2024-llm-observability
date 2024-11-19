import os, uuid, logging
import vector_store as vs

import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.rerank import CohereRerank

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

vector_store = vs.initialize(model_name=model_name)
retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

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

# Rerankerの使用フラグが有効（デフォルト）の場合は、CohereのRerankerを用いて、
# 取得した情報を関連度順に並び替えた後に、指定件数分のみ採用する
if use_reranker == True:
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, model="rerank-multilingual-v3.0", top_n=top_n
    )
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

prompt = """
以下の質問をコンテキストに基づいて、答えてください。

## コンテキスト
{context}

## 質問
{question}
"""

chain = (
    {"question": RunnablePassthrough(), "context": retriever}
    | PromptTemplate.from_template(prompt)
    | chat_model
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("何が聞きたいですか？"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]
        st.session_state["run_id"] = run_id = str(uuid.uuid4())
        stream = chain.stream(input=prompt, config={"run_id": run_id})
        response = st.write_stream(stream=stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

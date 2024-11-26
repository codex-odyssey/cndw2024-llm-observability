
from agent.agent import Agent

import os, logging
from langchain_openai.chat_models import ChatOpenAI
from langchain_cohere.chat_models import ChatCohere

from traceloop.sdk import Traceloop
Traceloop.init(
    # デモ用なので、batch processorではなく即時でトレースデータを送る
    disable_batch=True,
    # アプリケーションの名前
    app_name="CNDW2024 Session Bot",
    # 独自属性の追加
    resource_attributes={"env": "demo", "version": "1.0.0"},
)

model_name = "gpt-4o-mini"
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
temperature = .7
max_tokens = 1024

logger = logging.getLogger(name=__name__)

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

agent = Agent(llm=chat_model)

def exec(q:str):
    print(q)
    agent.run(question=q)

exec("Go言語をよく触るインフラエンジニアです。オススメのセッションを教えて？")
#exec("YAMLをよく書くインフラエンジニアです。オススメのセッションを教えてください")
#exec("ワクワクするセッションは？")
#exec("オススメのセッションを教えてください。")
#exec("OpenTelemetryに関連するセッションを教えてー")
#exec("LLMに関連するセッションを教えて")
#exec("SREにオススメのセッションを教えて")
#exec("認知的負荷に悩んでいます。オススメのセッションを教えてください")
#exec("プラットフォームエンジニアにオススメのセッションを教えてください")
#exec("SIerに所属しています。オススメのセッションを教えてください")

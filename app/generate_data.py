
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

print("歌手に関連するセッションを教えて！")
agent.run(question="歌手に関連するセッションを教えて！")
print("インフラエンジニアにおすすめは？")
agent.run(question="インフラエンジニアにおすすめは？")
print("androidエンジニアにおすすめは？")
agent.run(question="androidエンジニアにおすすめは？")
print("山芋たべたい！")
agent.run(question="山芋たべたい！")
print("ワクワクするセッションは？")
agent.run(question="ワクワクするセッションは？")
print("つらいよー。スライドが全然できてないの...")
agent.run(question="つらいよー。スライドが全然できてないの...")
print("オススメのセッションを教えてー")
agent.run(question="オススメのセッションを教えてー")
print("OpenTelemetryに関連するセッションを教えてー")
agent.run(question="OpenTelemetryに関連するセッションを教えてー")
print("はらへったー")
agent.run(question="はらへったー")

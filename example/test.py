import os
from dotenv import find_dotenv, load_dotenv
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage


_ = load_dotenv(find_dotenv())

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["TRACELOOP_BASE_URL"] = "http://localhost:4318"

from traceloop.sdk import Traceloop
from langchain_openai.chat_models import ChatOpenAI

Traceloop.init()

chat = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o-mini",
)
result=chat.invoke("資料作成の終わりが少し見えてきました！")

from traceloop.sdk.decorators import workflow

@workflow("hello")
def hello(s:str):
  print(s)

hello("Hello world")
print(result)

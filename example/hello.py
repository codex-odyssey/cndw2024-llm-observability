from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
import os

#Traceloop.init(app_name="joke_generation_service")

@workflow(name="joke_creation")
def create_joke():
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "opentelemetryに関するジョークを教えて"}],
  )

  return completion.choices[0].message.content


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

from langchain_core.prompts import ChatPromptTemplate
#print(create_joke())
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "あなたはCloud Nativeの分野に詳しい専門家です。",
    ),
    (
      "human",
      f"下記の質問に関連するCloud Nativeな分野の単語を1個つだけ教えてください。\n\n"
      "また、解答は単語のみをCSV形式で出力してください。\n\n"
      "たとえば、質問が「Go言語」であれば「kubernetes」と回答してください。"
      "質問:{question}",
    ),
  ]
)
chain = prompt | chat_model

print(chain.invoke("OpenTelemetryに関連するセッションを教えて").content)

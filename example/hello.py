from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
import os

Traceloop.init(app_name="joke_generation_service")

@workflow(name="joke_creation")
def create_joke():
  client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": "opentelemetryに関するジョークを教えて"}],
  )

  return completion.choices[0].message.content

#print(create_joke())


from langchain_openai import ChatOpenAI
from langchain_community.tools import HumanInputRun

from traceloop.sdk import Traceloop

Traceloop.init(app_name="simple_agent")

model = ChatOpenAI(model="gpt-4o")

tools = [HumanInputRun()]

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer

system_message = "あなたは有用なアシスタントです。"
memory = MemorySaver()
langgraph_agent_executor = create_react_agent(
  model, tools,state_modifier=system_message, checkpointer=memory
)
config = {"configurable": {"thread_id": "test-thread"}}

res = langgraph_agent_executor.invoke(
  {
    "messages": [
      ("user","天気を教えて"),
    ]
  },
  config,
)

print(res)

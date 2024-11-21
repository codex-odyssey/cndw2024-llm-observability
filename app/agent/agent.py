
from langchain_openai.chat_models import ChatOpenAI
from typing import Annotated, Any, Optional
import operator
from typing import Annotated, Any, Optional

from langchain_core.prompts import PromptTemplate

from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import vector_store as vs

# セッションを表すデータモデル
class Session(BaseModel):
  title: str = Field(..., description="セッションのタイトル")
  abstract: str = Field(..., description="セッションの要約")
  relevance_score: float = Field(..., description="セッションのrelevance_score")

class Sessions(BaseModel):
  sessions: list[Session] = Field(
    default_factory=list, description="セッションのリスト"
  )

from langchain_cohere.rerank import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

class SessionRetriever:
  def __init__(self, llm: ChatOpenAI, k: int = 3):
    self.llm = llm.with_structured_output(Sessions)

    vector_store = vs.initialize(model_name="gpt-4o-mini")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # CohereのRerankerを用いて、取得した情報を関連度順に並び替えた後に、指定件数分のみ採用する
    compressor = CohereRerank(
        model="rerank-multilingual-v3.0", top_n=k
    )
    self.retriever  = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

  def run(self, words: str) -> Sessions:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "Docutmentの情報を要約する専門家です。",
        ),
        (
          "human",
          f"以下のセッションの情報をまとめてください。\n\n"
          "セッション情報:{sessions}\n\n"
          "各セッションには、タイトルと要約とrelevance_scoreを含めてください。",
        ),
      ]
    )
    chain = { "sessions" : self.retriever} | prompt | self.llm
    return chain.invoke(words)

class Respondent:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm

  def run(self, context:str, related_words:str, question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          f"あなたは有能なアシストです。",
        ),
        (
          "human",
          f"セッション情報に基づいて、質問に答えてください。\n\n"
          "「質問に関連する単語」を考慮しても良いですが、その場合は「直接関連するセッションはありませんでしたが、関連しそうなセッションを検索してみました。」と回答に一文入れてください。\n\n"
          "## セッション情報\n\n"
          "{context}\n\n"
          "## 質問\n\n"
          "{question}\n\n"
          "## 質問に関連する単語\n\n"
          "{related_words}",
        ),
      ]
    )
    chain = prompt | self.llm #| StrOutputParser
    return chain.invoke({"context":context, "related_words":related_words , "question":question})

class KeywordGenerator:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm

  def run(self, question: str, k: int = 5) -> AIMessage:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "あなたはCloud Nativeの分野に詳しい専門家です。",
        ),
        (
          "human",
          f"下記の質問に関連するCloud Nativeな分野の単語を{k}個教えてください。\n\n"
          "また、2単語以上の場合、単語はカンマで区切って出力してください。\n\n"
          "質問:{question}",
        ),
      ]
    )
    chain = prompt | self.llm #| StrOutputParser
    return chain.invoke({"question":question,"k":k})

class State(BaseModel):
    question: str = Field(..., description="ユーザーからの質問")
    sessions: Sessions = Field(
      default_factory= lambda: Sessions(sessions=[]), description="取得したセッションのリスト"
    )
    keywords: str = Field(default="", description="questionに関連するキーワード")
    result: str = Field(default="", description="回答")
    iteration: int = Field(default=0, description="反復回数")
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )

class Agent:
    def __init__(self, llm: ChatOpenAI):
        # 各種ジェネレータの初期化
        self.keyword_generator = KeywordGenerator(llm=llm)
        self.respondent = Respondent(llm=llm)
        self.session_retriever = SessionRetriever(llm=llm)

        # グラフの作成
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # グラフの初期化
        workflow = StateGraph(State)

        # 各ノードの追加
        workflow.add_node("keyword_generator", self._keyword_generator)
        workflow.add_node("respondent", self._respondent)
        workflow.add_node("session_retriever", self._session_retriever)
        workflow.add_node("evaluate_information", self._evaluate_information)

        # エントリーポイントの設定
        workflow.set_entry_point("session_retriever")

        # ノード間のエッジの追加
        workflow.add_edge("keyword_generator", "session_retriever")
        workflow.add_edge("session_retriever", "evaluate_information")

        # 条件付きエッジの追加
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: state.is_information_sufficient or state.iteration > 3,
            {True: "respondent", False: "keyword_generator"},
        )
        workflow.add_edge("respondent", END)

        # グラフのコンパイル
        return workflow.compile()

    def _keyword_generator(self, state: State) -> dict[str, Any]:
        keywords: AIMessage = self.keyword_generator.run(state.question, state.iteration)
        return {
            "keywords": keywords.content,
        }

    def _respondent(self, state: State) ->  dict[str, Any]:
        result: str = self.respondent.run(
            str(state.sessions), state.keywords, state.question,
        )
        return {"result": result}

    def _session_retriever(self, state: State) -> dict[str, Any]:
        sessions: Sessions = self.session_retriever.run(
            state.question if state.keywords == "" else state.keywords,
        )
        return {"sessions": sessions}

    def _evaluate_information(self, state: State) -> dict[str, Any]:
        return {
            "is_information_sufficient": any(s.relevance_score >= 0.5 for s in state.sessions.sessions),
            "iteration": state.iteration + 1,
        }

    def run(self, question: str) -> str:
        # 初期状態の設定
        initial_state = State(question=question)
        # グラフの実行
        final_state = self.graph.invoke(initial_state)

        return final_state["result"]

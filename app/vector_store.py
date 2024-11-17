import os, logging
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(name=__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")


def initialize(model_name: str) -> FAISS:
    """Initialize Vector Stoare(FAISS)."""
    if model_name == "gpt-4o-mini":
        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key, model="text-embedding-ada-002"
        )
        logger.info("Use OpenAI Embedding model")
    elif model_name == "command-r-plus":
        embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key, model="embed-multilingual-v3.0"
        )
        logger.info("Use Cohere Embedding model")
    else:
        logger.error("Unsetted model name")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    documents = CSVLoader(file_path="./data/cndw2024_accepted_sessions.csv").load()
    vector_store.add_documents(documents=documents)
    return vector_store

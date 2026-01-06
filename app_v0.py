# Import libraries
import os 
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load .env variables into the environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

VIDEO_ID = "Gfr50f6ZBvo"
QUESTION = (
    "Is the topic of nuclear fusion discussed in this video? "
    "If yes, what exactly was discussed?"
)

# --- YouTube transcript (new API) ---
try:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(VIDEO_ID, languages=["en"])  # FetchedTranscript
    transcript_list = fetched.to_raw_data()              # [{text,start,duration}, ...]
    transcript = " ".join(x["text"] for x in transcript_list).strip()
except TranscriptsDisabled:
    print("No captions available for this video.")
    transcript = ""
except Exception as e:
    print(f"Failed to fetch transcript: {e}")
    transcript = ""

if not transcript:
    raise SystemExit("Cannot proceed: no transcript retrieved.")

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.create_documents([transcript])

# --- Local embeddings (FastEmbed uses onnxruntime CPU we pinned) ---
embeddings = FastEmbedEmbeddings()

# --- Vector store / retriever ---
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- Prompt & LLM (Groq) ---
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a concise, helpful assistant. Use ONLY the context to answer.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "If the answer isn't in the context, say you can't find it."
    ),
)

# Options: "llama-3.1-8b-instant" (fast) or "llama-3.1-70b-versatile" (stronger)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})
parser = StrOutputParser()
main_chain = parallel | prompt | llm | parser

# --- Run ---
answer = main_chain.invoke(QUESTION)
print("\n=== Answer ===\n")
print(answer)

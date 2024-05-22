from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# https://python.langchain.com/v0.2/docs/tutorials/rag/#indexing-load
# https://github.com/ollama/ollama/tree/main/examples/langchain-python-rag-document

loader = TextLoader("./data/ququ.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)
prompt = hub.pull("rlm/rag-prompt")
model = 'mistral'
embedding = OllamaEmbeddings(model=model)

try:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
except:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding,  persist_directory="./chroma_db")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = Ollama(model=model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

while True:
    query = input("\nQuery: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(rag_chain.invoke(query))


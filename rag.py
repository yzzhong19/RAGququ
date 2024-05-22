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
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# https://python.langchain.com/v0.2/docs/tutorials/rag/#indexing-load
# https://github.com/ollama/ollama/tree/main/examples/langchain-python-rag-document

loader = TextLoader("./data/ququ.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
splits = text_splitter.split_documents(docs)
prompt = hub.pull("rlm/rag-prompt")
model = 'mistral'
embedding = OllamaEmbeddings(model=model)
print(len(splits))
try:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
except Exception as e:
    print('writing vector database')
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

    system_prompt = (
        "你是一个回答问题的助手。请使用以下检索到的内容来回答问题。如果你不知道答案，就说你不知道。请用中文回答"
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    print(response["answer"])
    #print(response['context'])
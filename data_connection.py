#!pip install GitPython

# Document loaders
from langchain.document_loaders import GitLoader

def file_filter(file_path):
    return file_path.endswith(".mdx")

loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loader.load()
print(len(raw_docs))

# Document transformers

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

docs = text_splitter.split_documents(raw_docs)
print(len(docs))

# !pip install tiktoken

# Text embedding models
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# query = "AWSのS3からデータを読み込むためのDocumentLoaderはありますか？"

# vector = embeddings.embed_query(query)
# print(len(vector))
# # print(vector)

# !pip install chromadb==0.4.15

from langchain.vectorstores import Chroma

db = Chroma.from_documents(docs, embeddings)

# Retrievers

retriever = db.as_retriever()

query = "AWSのS3からデータを読み込むためのDocumentLoaderはありますか？"

content_docs = retriever.get_relevant_documents(query)
print(f"lent = {len(content_docs)}")

first_doc = content_docs[0]
print(f"metadata = {first_doc.metadata}")

print("^^^^^^^^^^^^^^")
print(first_doc.page_content)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.run(query)
print(result)
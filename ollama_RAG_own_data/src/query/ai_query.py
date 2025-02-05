import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

def get_data_from_db():

    connection_string = "postgresql+psycopg2://" + os.getenv("POSTGRES_DB_USER") + ":" + os.getenv("POSTGRES_DB_PASSWORD") + "@" + os.getenv(
        "POSTGRES_DB_HOST") + ":" + os.getenv("POSTGRES_DB_PORT") + "/" + os.getenv("POSTGRES_DB_NAME")

    vector_db = PGVector(
        embeddings=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL")),
        connection=connection_string,
        collection_name=os.getenv("POSTGRES_DB_COLLECTION_NAME")
    )

    # create model
    llm = ChatOllama(model="llama3.2")

    query_template = PromptTemplate(
        input_variables=["question"],
        template="""
        You are an expert AI language model specialized in Biology. Your task is to methodically generate five distinct, semantically diverse reformulations of the user’s original question to optimize retrieval of biologically relevant documents from a vector database. Each reformulated question must:

        1. Explore alternate angles of the original query (e.g., molecular, ecological, evolutionary, or systems-level perspectives).

        2. Incorporate domain-specific terminology, synonyms, and hierarchical relationships (e.g., taxa, biochemical pathways).

        3. Vary syntactic structures (e.g., hypothesis-driven, definition-based, comparative, or cause-effect framing) to broaden search scope.

        Prioritize questions that collectively address potential gaps in keyword-based similarity metrics, such as:

        * Terminological variations (e.g., ‘apoptosis’ vs. ‘programmed cell death’).

        * Cross-disciplinary context (e.g., linking genetic mechanisms to organismal phenotypes).

        * Granularity shifts (e.g., cellular processes vs. population-level impacts).
        Ensure all questions adhere to rigorous academic standards and contextual breadth for robust biomedical or life sciences document retrieval.
         Original question: {question}
        """,
    )

    retriver = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm=llm,
        prompt=query_template
    )

    # RAG prompt
    template = """
    Answer the question based ONLY on the following context:{context}
    Question: {question}
    """

    # create prompt
    prompt = ChatPromptTemplate.from_template(template)

    # create chain
    chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    res = chain.invoke(input=("What is the document about?"))
    print(f"\n Q1. What is the document about? \n Answer: {res}\n")
    res = chain.invoke(input=("What are the key points of the document that student should know very well?"))
    print(f"\n Q2. What are the key points of the document that student should know very well? \n Answer: {res}\n")


def main():
    load_dotenv()
    get_data_from_db()


if __name__ == "__main__":
    main()

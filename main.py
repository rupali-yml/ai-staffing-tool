import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from langchain.retrievers import MergerRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

llm_prompt_template = """You are an HR in a medium sized company. You will be given employees details,
  project details and employee reviews in the context.

  Your tasks will be to recommend team structure and team size along with employee names for a given project.
  You will be given the following details - Project Name, Project description, Duration,
  Budget, Start Date and Platforms to be built, in the Question.

  Your recommendations should align with the required skills for the project and reviews. For instance,
  individuals proficient in front-end technologies could be matched with web-related tasks, while those skilled in
  Java or NodeJS might be suitable for backend development. Similarly, those experienced in machine learning
  could contribute to ML-related tasks.

  You should calculate a matching score for each team member based on their skills and experience, aiming to
  assign those with the highest compatibility to the project.

  \nQuestion: {question}
  \nContext: {context}

  Answer should be in the following format or consists of these many details.

  Replace the word "allocation percentage" of
  each individual with the appropriate "allocation" value from context.

  Example Answer:
      Project Name: Gen AI POC
      Project Description:
      This is a proof-of-concept project aimed at exploring the capabilities of Gen AI.
      Duration: 6 Months
      Budget: $0.25M
      Start Date: 06-30-2024
      Platforms to be built:
        - Web
        - Backend
      Engineering Team:
          - Team Structure:
            - Frontend Lead (FE Lead) - 1 Name - Suchak Mihir Dinkarray (100% allocation)
            - Frontend Software Engineer (FE SE) - 1 Name - Raghav Sharma (100% allocation)
            - Backend Lead (BE Lead) - 1 Name - Srijita Thakur (50% allocation percentage)
            - Backend Associate Software Engineer (BE ASE) - 1 Name - Bandhan Roy (25% allocation)
            - Quality Engineer (QA SSE) - 1 Name - Aishwarya Chandrakant Madiwal (50% allocation)
            - Engineering Manager - 1 Name - Vinay S (25% allocation)
            - Project Manager - 1 Name - Maryam Fatima (100% allocation)
            - Product Manager - 1 Name - Pallavi Tandan (50% allocation)
            - Director of Engineering - 1 Name - Shubhang Krishnamurthy Vishwamitra (100% allocation)

          - Total Team Size: 9
      Tech Stacks:
        - Python
        - Langchain
        - NodeJS
        - React
        - Next JS
        - MongoDB

  \nAnswer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_open_api_key():
    f = open('secrets/google_api_key.txt')
    api_key = f.read()
    print(api_key)
    os.environ['GOOGLE_API_KEY'] = api_key


def allocate_staff(
        project_name: str,
        project_description: str,
        duration: str,
        budget: str,
        start_date: str,
        platforms: str,
        skills_preferred: str
):
    get_open_api_key()
    # Loading the files
    csv_loader = CSVLoader("./data/csv_data/SkillMatrixDummy.csv")
    skill_matrix_document = csv_loader.load()

    csv_loader = CSVLoader("./data/csv_data/generate_dummy_project_details_csv_file")
    project_details_document = csv_loader.load()

    loader = PyPDFLoader("./data/csv_data/Feedbacks 2-10.pdf")
    feedback_review_document = loader.load()

    # Creating the vector stores
    feedback_text_dir = "./db/feedback_text_db"
    skill_matrix_text_dir = "./db/skill_matrix_csv_text_db"
    project_details_text_dir = "./db/project_details_text_db"

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    skill_matrix_vector_store = Chroma.from_documents(
        documents=skill_matrix_document,
        embedding=embeddings,
        persist_directory=skill_matrix_text_dir
    )
    skill_matrix_vector_store.persist()
    skill_matrix_vector_store_disk = Chroma(persist_directory=skill_matrix_text_dir, embedding_function=embeddings)
    skill_matrix_retriever = skill_matrix_vector_store_disk.as_retriever(search_kwargs={"k": 1})

    project_details_vector_store = Chroma.from_documents(
        documents=project_details_document,
        embedding=embeddings,
        persist_directory=project_details_text_dir
    )
    project_details_vector_store_disk = Chroma(persist_directory=project_details_text_dir,
                                               embedding_function=embeddings)
    project_details_retriever = project_details_vector_store_disk.as_retriever(search_kwargs={"k": 1})

    feedback_text_vector_store = Chroma.from_documents(
        documents=feedback_review_document,
        embedding=embeddings,
        persist_directory=feedback_text_dir
    )
    feedback_text_vector_store_disk = Chroma(persist_directory=feedback_text_dir, embedding_function=embeddings)
    feedback_text_retriever = feedback_text_vector_store_disk.as_retriever(search_kwargs={"k": 1})

    # Merging the retriever
    filter_ordered_cluster = EmbeddingsClusteringFilter(
        embeddings=embeddings,
        num_clusters=3,
        num_closest=1,
    )

    merged_retriever = MergerRetriever(
        retrievers=[skill_matrix_retriever, feedback_text_retriever, project_details_retriever])

    pipeline = DocumentCompressorPipeline(transformers=[
        filter_ordered_cluster,
    ])
    compression_retriever_reordered = ContextualCompressionRetriever(
        base_compressor=pipeline, base_retriever=merged_retriever
    )

    # Loading the llm
    llm = ChatGoogleGenerativeAI(model="gemini-pro",
                                 temperature=0.1, top_p=0.85)

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    rag_chain = (
            {"context": compression_retriever_reordered | format_docs, "question": RunnablePassthrough()}
            | llm_prompt
            | llm
            | StrOutputParser()
    )

    # Invoking the llm
    response = rag_chain.invoke(f"""
            Using the given context and data create an engineering team structure with their names and allocation percentage and the relevant tech stacks.
            Engineering team structure consists of the name, title of engineer and percentage of allocation of engineer into the project. Make sure you find the right person for the right role.
            Project Name: {project_name}
            Project Description: {project_description}
            Duration: {duration}
            Budget: {budget}
            Start Date: {start_date}
            Platforms to be built:
              {platforms}
            Skills Preferred: {skills_preferred}
        """)

    return response


@app.post('/assign_staff_to_a_project')
async def predict(project_name, project_description, duration, budget, start_date, platforms,
                  skills_preferred) -> Any:
    print("I am here")
    return allocate_staff(project_name, project_description, duration, budget, start_date, platforms, skills_preferred)


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)

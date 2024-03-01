from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.llms import GPT4All
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain

sentiment_text_dir = "./db/sentiment_text_db"
skill_Matrix_text_dir = "./db/csv_text_db"
project_details_text_dir = "./db/project_details_text_db"


def text_splitter_method():
    return RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )


def embedding_function_method():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def main():
    # Sentiment analysis reader
    loader = PyPDFLoader("./data/csv_data/Feedbacks 2-10.pdf")
    text_splitter = text_splitter_method()
    embedding_function = embedding_function_method()

    sentiment_text = ""
    for page in loader.load_and_split(text_splitter=text_splitter):
        sentiment_text += page.page_content + "\n"

    sentiment_split_text = text_splitter.create_documents([sentiment_text])

    sentiment_text_db = Chroma.from_documents(documents=sentiment_split_text,
                                              embedding=embedding_function,
                                              persist_directory=sentiment_text_dir)

    sentiment_text_db.persist()

    sentiment_vectorstore = Chroma(
        persist_directory=sentiment_text_dir,
        embedding_function=embedding_function
    )

    sentiment_retriever = sentiment_vectorstore.as_retriever(search_kwargs={"k": 1})

    print(sentiment_retriever.get_relevant_documents("YML0002-Ananya"))

    # skill matrix reader
    with open("./data/csv_data/SkillMatrixDummy.csv", "r", encoding="utf-8") as file:
        file_content = file.read()
    skill_matrix_split_text = text_splitter.create_documents([file_content])

    skill_matrix_text_db = Chroma.from_documents(documents=skill_matrix_split_text,
                                                 embedding=embedding_function,
                                                 persist_directory=skill_Matrix_text_dir)

    skill_matrix_text_db.persist()

    skill_matrix_vectorstore = Chroma(
        persist_directory=skill_Matrix_text_dir,
        embedding_function=embedding_function
    )

    skill_matrix_retriever = skill_matrix_vectorstore.as_retriever(search_kwargs={"k": 1})

    # project matrix reader
    with open("./data/csv_data/generate_dummy_project_details_csv_file", "r", encoding="utf-8") as file:
        file_content = file.read()
    project_details_split_text = text_splitter.create_documents([file_content])

    project_details_text_db = Chroma.from_documents(documents=project_details_split_text,
                                                    embedding=embedding_function,
                                                    persist_directory=project_details_text_dir)

    project_details_text_db.persist()

    project_details_vectorstore = Chroma(
        persist_directory=project_details_text_dir,
        embedding_function=embedding_function
    )

    project_details_retriever = project_details_vectorstore.as_retriever(search_kwargs={"k": 1})

    llm = GPT4All(model="./models/orca-mini-3b-gguf2-q4_0.gguf", backend="llama")

    prompt_template = """You are an HR person. Suggest employees for the upcoming
    project in the question, taking help from the context. Employees should belong
    to skill_matrix_retriever.
    \nQuestion: {question} 
    \nContext: {context} 
    \nAnswer:"""

    model_prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

    llm_chain = LLMChain(prompt=model_prompt, llm=llm, verbose=True)

    suggestions = llm_chain.invoke(
        {
            'question': f"Suggest one employee for machine learning project."
                        f"Project Name: Langchain POC, "
                        f"Project Description: This is a proof-of-concept project aimed at exploring the capabilities "
                        f"of Langchain."
                        f"Duration: 2 Months "
                        f"Budget: $0.25M "
                        f"Start Date: 03-01-2024 "
                        f"Platforms to be built: "
                        f"  - Web (preferably angular) "
                        f"  - Backend (preferably java) "
                        f"  - Machine Learning Langchain "
                        f"Skills Preferred: Java, python, machine learning ",
            # 'question': f"Suggest total team size, Engineering team structure with their names and allocation "
            # f"percentage" f"and Tech stacks. Engineering team structure consists of title of engineer and
            # percentage of " f"allocation of" f"engineer into the project. " f"Project Name: Langchain POC,
            # " f"Project Description: This is a proof-of-concept project aimed at exploring the capabilities " f"of
            # Langchain." f"Duration: 2 Months " f"Budget: $0.25M " f"Start Date: 03-01-2024 " f"Platforms to be
            # built: " f"  - Web (preferably angular) " f"  - Backend (preferably java) " f"  - Machine Learning
            # Langchain " f"Skills Preferred: Java, python, machine learning ",
            'context': [sentiment_retriever, skill_matrix_retriever, project_details_retriever]})

    print(f"Suggestions are: \n {suggestions}")


# if __name__ == "__main__":
#     main()

import csv
import pandas as pd
import openai
import os

from gpt4all import GPT4All
from langchain.agents import AgentType
from langchain.chains import LLMChain
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# def read_csv(csv_path):
#     data = []
#     with open(csv_path, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         for row in csv_reader:
#             data.append(row)
#     return data

template = """You are an HR person.
        
        You will be given the following details - Project Name, Project description, Duration, 
        Budget, Start Date and Platforms to be built in the Question.
        
        Suggest the Name of employee from skill_matrix_all_df_data which is a pandas dataframe passed in df argument, 
        also check the availability of any
        particular employee on the basis of its allocation to another project which can be found in 
        project_details_df_data, which is also a pandas dataframe passed in df argument.
        
        Your recommendations should align with the required skills for the project. For instance, 
        individuals proficient in front-end technologies could be matched with web-related tasks, while those skilled in 
        Java or NodeJS might be suitable for backend development. Similarly, those experienced in machine learning 
        could contribute to ML-related tasks.

        You should calculate a matching score for each team member based on their skills and experience, aiming to 
        assign those with the highest compatibility to the project.
        
        Answer should be in the following format or consists of these many details.
        
        AnswerFormat: 
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
                  - Backend Lead (BE Lead) - 1 Name - Srijita Thakur (50% allocation)
                  - Backend Associate Software Engineer (BE ASE) - 1 Name - Bandhan Roy (100% allocation)
                  - Quality Engineer (QA SSE) - 1 Name - Aishwarya Chandrakant Madiwal (100% allocation)
                  - Engineering Manager - 1 Name - Vinay S (50% allocation)
                  - Project Manager - 1 Name - Maryam Fatima (100% allocation)
                  - Product Manager - 1 Name - Pallavi Tandan (50% allocation)
                  - Director of Engineering - 1 Name - Shubhang Krishnamurthy Vishwamitra (25% allocation)
                
                - Total Team Size: 9
            Tech Stacks:
              - Python
              - Langchain
              - NodeJS
              - React
              - Next JS
              - MongoDB
        
        Engineering Manager, Project Manager, Product Manager and Director of Engineering must be there 
        in every project. Print the Total Team Size correctly. Give names from skill matrix all dataframe 
        and allocation %. Give complete answer.
        """


def main():
    try:
        skill_matrix_all_csv_path = 'data/csv_data/Skill Matrix_All.csv'
        skill_matrix_all_data = pd.read_csv(skill_matrix_all_csv_path)
        # print(skill_matrix_all_data.info())

        project_details_csv_path = 'data/csv_data/generate_dummy_project_details_csv_file'
        project_details_csv_data = pd.read_csv(project_details_csv_path)
        # print(project_details_csv_data.info())
        prompt_question = """
            Project Name: Langchain POC
            Project Description:
            This is a proof-of-concept project aimed at exploring the capabilities of Langchain.
            Duration: 2 Months
            Budget: $0.25M
            Start Date: 03-01-2024
            Platforms to be built:
             - Web (preferably angular)
             - Backend (preferably java)
             - Machine Learning Langchain
            Suggest total team size, Engineering team structure with their names and allocation percentage
            and Tech stacks. Engineering team structure consists of title of engineer and percentage of allocation of
            engineer into the project.
        """

        get_project_allocation_suggestion(skill_matrix_all_data, project_details_csv_data, prompt_question)
    except Exception as e:
        print(e)


def get_open_api_key():
    f = open('secrets/openapi_key.txt')
    api_key = f.read()
    print(api_key)
    os.environ['OPENAI_API_KEY'] = api_key
    openai.api_key = os.getenv('OPENAI_API_KEY')


def get_project_allocation_suggestion(skill_matrix_all_df_data, project_details_df_data, prompt_question):
    get_open_api_key()
    # prompt = PromptTemplate(template=template, input_variables=["prompt_question"])
    callbacks = [StreamingStdOutCallbackHandler()]

    # llm_model = GPT4All(model_name="gpt4all-falcon-newbpe-q4_0.gguf", // hugging face gemma
    #                     model_path="./models/",
    #                     n_threads=4)

    llm_model = "gpt-3.5-turbo-0613"
    llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=2, model=llm_model)

    # chat_agent = create_pandas_dataframe_agent(
    #     llm,
    #     [skill_matrix_all_df_data, project_details_df_data],
    #     verbose=True,
    #     agent_type=AgentType.OPENAI_FUNCTIONS,
    #     prefix=template,
    #     handle_parsing_errors=True
    # )
    chat_agent = create_csv_agent(
        llm,
        ["./data/csv_data/generate_dummy_project_details_csv_file", "./data/csv_data/Skill Matrix_All.csv"],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=template,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    response = chat_agent.run(prompt_question)
    # llm_chain = LLMChain(prompt=prompt, llm=llm_model, verbose=True)
    # response = llm_chain.invoke()
    return response


if __name__ == "__main__":
    main()

import os
from constants import openai_key
from langchain_community.llms import  OpenAI
import streamlit as st  
from langchain.prompts import PromptTemplate        # for Prompt Engineering
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = openai_key
st.title("My first Lanchain demo using OpenAI...by Rajeev Khoodeeram")
input_text = st.text_input("Search your prefered topic...")

# Using the Prompt template 
myfirst_prompt  =  PromptTemplate(
    input_variables=["name"],
    template="What is the definition of {name}?" 
    )



## initialise your openai
llm = OpenAI()
llm = LLMChain(llm=llm, prompt =  myfirst_prompt, verbose =  True)
chain = llm

if input_text:
    st.write(chain.invoke(input_text))

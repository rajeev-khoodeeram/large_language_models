'''
This program shows how to use from SimpleSequentialChain
to execute a sequence of chains in a simple sequential manner.
The output is in the  format of json which displays only  the result from the second prompt
The first prompt is ignored or simple displayed in the terminal
'''

import os
from constants import openai_key
from langchain_community.llms import  OpenAI
import streamlit as st  
from langchain.prompts import PromptTemplate        # for Prompt Engineering
from langchain.chains import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.sequential import SequentialChain


os.environ["OPENAI_API_KEY"] = openai_key
st.title("My first Lanchain demo using OpenAI...by Rajeev Khoodeeram")
input_text = st.text_input("Search your prefered topic...")

## initialise your openai
llm = OpenAI()

# Using the Prompt template , but this time is with  a more complex prompt
myfirst_prompt  =  PromptTemplate(
    input_variables=["name"],
    template="Who is {name}?" 
    )
chain1  = LLMChain(llm=llm, prompt =  myfirst_prompt, verbose =  True, output_key="person")


mysecond_prompt  =  PromptTemplate(
    input_variables=["person"],
    template="When was {person} born ?" 
    )
chain2  = LLMChain(llm=llm, prompt =  mysecond_prompt, verbose =  True, output_key="dob")


mythird_prompt  =  PromptTemplate(
    input_variables=["dob"],
    template="List of events happening during that period --- {dob} ?" 
    )
chain3  = LLMChain(llm=llm, prompt =  mythird_prompt, verbose =  True, output_key="description")

# mainChain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
mainChain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['name'], 
                            output_variables=['person','dob', 'description'], verbose=True)

if input_text:
    #  execute the chain when SimpleSequentialChain is used
    st.write(mainChain.invoke(input_text))
    #  execute the chain when SequentialChain is used
    # st.write(mainChain({'name':input_text}))
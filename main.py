import os
from constants import openai_key
import streamlit as st  
from langchain_community.llms import OpenAI

os.environ["OPENAI_API_KEY"] = openai_key

st.title("My first Lanchain demo using OpenAI...by Rajeev Khoodeeram")
input_text = st.text_input("Search your prefered topic...")

## initialise your openai
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))

import os
tesseract_path = "/usr/bin/tesseract"  # Replace with the actual path to tesseract
os.environ["PATH"] += os.pathsep + tesseract_path


from uuid import uuid4
import openai
import streamlit as st
import shutil

from tempfile import NamedTemporaryFile
import tempfile

import os

# Get the current working directory
cwd = os.getcwd()

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Multi-Modal RAG App`PDF`')

st.sidebar.subheader('Text Summarization Model')
time_hist_color = st.sidebar.selectbox('Summarize by', ('gpt-4-turbo', 'gemini-1.5-pro-latest'))

st.sidebar.subheader('Image Summarization Model')
immage_sum_model = st.sidebar.selectbox('Summarize by', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest'))

#st.sidebar.subheader('Embedding Model')
#embedding_model = st.sidebar.selectbox('Select data', ('OpenAIEmbeddings', 'GoogleGenerativeAIEmbeddings'))

st.sidebar.subheader('Response Generation Model')
generation_model = st.sidebar.selectbox('Select data', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest'))

#st.sidebar.subheader('Line chart parameters')
#plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
max_concurrecy = st.sidebar.slider('Maximum Concurrency', 3, 4, 5)

st.sidebar.markdown('''
---
Multi-Modal RAG App with Multi Vector Retriever
''')

#st.write(tables)


#from langchain_google_vertexai import ChatVertexAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai
from unstructured.partition.pdf import partition_pdf




uploaded_file = st.file_uploader(label = "Upload your file",type="pdf")
if uploaded_file is not None:
    temp_file="./temp.pdf"
    with open(temp_file,"wb") as file:
        file.write(uploaded_file.getvalue())

    image_path = "./"
    pdf_elements = partition_pdf(
        temp_file,
        chunking_strategy="by_title",
        #chunking_strategy="basic",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
    )
    

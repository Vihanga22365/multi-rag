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

    # Categorize elements by type
    def categorize_elements(_raw_pdf_elements):
      """
      Categorize extracted elements from a PDF into tables and texts.
      raw_pdf_elements: List of unstructured.documents.elements
      """
      tables = []
      texts = []
      for element in _raw_pdf_elements:
          if "unstructured.documents.elements.Table" in str(type(element)):
              tables.append(str(element))
          elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
              texts.append(str(element))
      return texts, tables
    
    texts, tables = categorize_elements(pdf_elements)


symbol = st.text_input("Enter the question")

#st.write("Current working directory:", cwd)


pr = st.button("Generate")
if pr ==True:
  unique_id = uuid4().hex[0:8]
  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_PROJECT"] = "multi_model_rag_mvr"
  os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
  

  # Generate summaries of text elements
  def generate_text_summaries(texts, tables, summarize_texts=False):
      """
      Summarize text elements
      texts: List of str
      tables: List of str
      summarize_texts: Bool to summarize texts
      """

      # Prompt
      prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
      These summaries will be embedded and used to retrieve the raw text or table elements. \
      Give a concise summary of the table or text that is well-optimized for retrieval. Table \
      or text: {element} """

      prompt = PromptTemplate.from_template(prompt_text)
      empty_response = RunnableLambda(
          lambda x: AIMessage(content="Error processing document")
      )
      # Text summary chain

      if time_hist_color == 'gpt-4-turbo':
        model = ChatOpenAI(
          temperature=0, model= "gpt-4-turbo", max_tokens=1024)

      else:
        model = ChatGoogleGenerativeAI(
            #temperature=0, model="gemini-pro", max_output_tokens=1024
            temperature=0, model="gemini-1.5-pro-latest", max_output_tokens=1024
        )


          #temperature=0, model="gemini-1.5-pro-latest", max_output_tokens=1024
      #)


      summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

      # Initialize empty summaries
      text_summaries = []
      table_summaries = []

      # Apply to text if texts are provided and summarization is requested
      if texts and summarize_texts:
          text_summaries = summarize_chain.batch(texts, {"max_concurrency": max_concurrecy})
      elif texts:
          text_summaries = texts

      # Apply to tables if tables are provided
      if tables:
          table_summaries = summarize_chain.batch(tables, {"max_concurrency":max_concurrecy})

      return text_summaries, table_summaries


  # Get text, table summaries
  text_summaries, table_summaries = generate_text_summaries(
      texts, tables, summarize_texts=True
  )

  import uuid
  import base64
  from langchain.schema.messages import HumanMessage, SystemMessage

  import base64
  import os

  from langchain_core.messages import HumanMessage

  def encode_image(image_path):
      """Getting the base64 string"""
      with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode("utf-8")

  def image_summarize(img_base64, prompt):
      """Make image summary"""
      if immage_sum_model == 'gpt-4-vision-preview':
        model = ChatOpenAI(
          temperature=0, model=immage_sum_model, max_tokens=1024)
      else:
        #model = ChatGoogleGenerativeAI(model="gemini-pro-vision", max_output_tokens=1024)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", max_output_tokens=1024)

      msg = model(
          [
              HumanMessage(
                  content=[
                      {"type": "text", "text": prompt},
                      {
                          "type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                      },
                  ]
              )
          ]
      )
      return msg.content
  
  def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

  def generate_img_summaries(path):
      """
      Generate summaries and base64 encoded strings for images
      path: Path to list of .jpg files extracted by Unstructured
      """

      # Store base64 encoded images
      img_base64_list = []

      # Store image summaries
      image_summaries = []

      # Prompt
      prompt = """You are an assistant tasked with summarizing images for retrieval. \
      These summaries will be embedded and used to retrieve the raw image. \
      Give a concise summary of the image that is well optimized for retrieval."""

      # Apply to images
      for img_file in sorted(os.listdir(path)):
          if img_file.endswith(".jpg"):
              img_path = os.path.join(path, img_file)
              base64_image = encode_image(img_path)
              img_base64_list.append(base64_image)
              image_summaries.append(image_summarize(base64_image, prompt))

      return img_base64_list, image_summaries

  fpath = "/content/figures"
  
  create_directory_if_not_exists(fpath)
  # Image summaries
  img_base64_list, image_summaries = generate_img_summaries(fpath)

  import uuid

  from langchain.embeddings import VertexAIEmbeddings
  from langchain.retrievers.multi_vector import MultiVectorRetriever
  from langchain.schema.document import Document
  from langchain.storage import InMemoryStore
  from langchain.vectorstores import Chroma
  from langchain_google_genai import GoogleGenerativeAIEmbeddings




  def create_multi_vector_retriever(
      vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
  ):
      """
      Create retriever that indexes summaries, but returns raw images or texts
      """

      # Initialize the storage layer
      store = InMemoryStore()
      id_key = "doc_id"

      # Create the multi-vector retriever
      retriever = MultiVectorRetriever(
          vectorstore=vectorstore,
          docstore=store,
          id_key=id_key,
      )
      # Helper function to add documents to the vectorstore and docstore
      def add_documents(retriever, doc_summaries, doc_contents):
          doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
          summary_docs = [
              Document(page_content=s, metadata={id_key: doc_ids[i]})
              for i, s in enumerate(doc_summaries)
          ]
          retriever.vectorstore.add_documents(summary_docs)
          retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

      # Add texts, tables, and images
      # Check that text_summaries is not empty before adding
      if text_summaries:
          add_documents(retriever, text_summaries, texts)
      # Check that table_summaries is not empty before adding
      if table_summaries:
          add_documents(retriever, table_summaries, tables)
      # Check that image_summaries is not empty before adding
      if image_summaries:
          add_documents(retriever, image_summaries, images)
      return retriever


  vectorstore = Chroma(
      collection_name="mm_rag_mistral",
        embedding_function=OpenAIEmbeddings()
  )


  #vectorstore = Chroma(
  #collection_name="mm_rag_mistral",
   # embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  #)

  # Create retriever
  retriever_multi_vector_img = create_multi_vector_retriever(
      vectorstore,
      text_summaries,
      texts,
      table_summaries,
      tables,
      image_summaries,
      img_base64_list,
  )

  import io
  import re

  from IPython.display import HTML, display
  from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
  from PIL import Image
  from langchain.chat_models import ChatOpenAI


  def looks_like_base64(sb):
      """Check if the string looks like base64"""
      return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


  def is_image_data(b64data):
      """
      Check if the base64 data is an image by looking at the start of the data
      """
      image_signatures = {
          b"\xFF\xD8\xFF": "jpg",
          b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
          b"\x47\x49\x46\x38": "gif",
          b"\x52\x49\x46\x46": "webp",
      }
      try:
          header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
          for sig, format in image_signatures.items():
              if header.startswith(sig):
                  return True
          return False
      except Exception:
          return False

  def resize_base64_image(base64_string, size=(128, 128)):
      """
      Resize an image encoded as a Base64 string
      """
      # Decode the Base64 string
      img_data = base64.b64decode(base64_string)
      img = Image.open(io.BytesIO(img_data))

      # Resize the image
      resized_img = img.resize(size, Image.LANCZOS)

      # Save the resized image to a bytes buffer
      buffered = io.BytesIO()
      resized_img.save(buffered, format=img.format)
      # Encode the resized image to Base64
      return base64.b64encode(buffered.getvalue()).decode("utf-8")

  #context creation
  def split_image_text_types(docs):
      """
      Split base64-encoded images and texts
      """
      b64_images = []
      texts = []
      for doc in docs:
          # Check if the document is of type Document and extract page_content if so
          if isinstance(doc, Document):
              doc = doc.page_content
          if looks_like_base64(doc) and is_image_data(doc):
              doc = resize_base64_image(doc, size=(1300, 600))
              b64_images.append(doc)
          else:
              texts.append(doc)
      if len(b64_images) > 0:
          return {"images": b64_images[:1], "texts": []}
      return {"images": b64_images, "texts": texts}


  #response generation
  def img_prompt_func(data_dict):
      """
      Join the context into a single string
      """
      formatted_texts = "\n".join(data_dict["context"]["texts"])
      messages = []

      # Adding the text for analysis
      text_message = {
          "type": "text",
          "text": (
              "You are an AI scientist tasking with providing factual answers.\n"
              "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
              "Use this information to provide answers related to the user question. \n"
              "Final answer should be easily readable and structured. \n"
              f"User-provided question: {data_dict['question']}\n\n"
              "Text and / or tables:\n"
              f"{formatted_texts}"
          ),
      }
      messages.append(text_message)
      # Adding image(s) to the messages if present
      if data_dict["context"]["images"]:
          for image in data_dict["context"]["images"]:
              image_message = {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{image}"},
              }
              messages.append(image_message)
      return [HumanMessage(content=messages)]


  def multi_modal_rag_chain(_retriever):
      """
      Multi-modal RAG chain
      """

      # Multi-modal LLM
      if generation_model == 'gemini-1.5-pro-latest':
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",max_output_tokens=1024)
      else:
        try:
          model = ChatOpenAI(model="gpt-4-vision-preview", max_output_tokens=1024)
        except Exception as e:
          model = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)

      #model = ChatOpenAI(model="gpt-4-vision-preview", openai_api_key = OPENAI_API_KEY, max_tokens=1024)

      # RAG pipeline
      chain = (
          {
              "context": _retriever | RunnableLambda(split_image_text_types),
              "question": RunnablePassthrough(),
          }
          | RunnableLambda(img_prompt_func)
          | model
          | StrOutputParser()
      )

      return chain


  # Create RAG chain
  chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

  query = symbol
  docs = retriever_multi_vector_img.get_relevant_documents(query, limit=1)

  #len(docs)
  #docs

  import base64
  from PIL import Image
  from io import BytesIO

  from IPython.display import display, Markdown



  markdown_text = chain_multimodal_rag.invoke(query)
  #display(Markdown(markdown_text))
  st.write(markdown_text)


  from IPython import display
  from IPython.display import HTML, display as ipy_display


  def load_image(_image_file):
    img = Image.open(_image_file)
    return img

  found_image = False  # Flag variable to track if an image has been found

  for i in range(len(docs)):
      if docs[i].startswith('/9j') and not found_image:
          display.display(HTML(f'<img src="data:image/jpeg;base64,{docs[i]}">'))

          base64_image = docs[i]
          image_data = base64.b64decode(base64_image)

          # Display the image
          #img = Image.open(BytesIO(image_data))
          #img.show()
          #img = load_image(image_data)
          st.image(image_data)
          found_image = True  # Set the flag to True to indicate that an image has been found


      #elif not docs[i].startswith('/9j'):
          # Display the document in the notebook
          #ipy_display(docs[i])

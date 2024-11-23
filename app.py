from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import fitz  # PyMuPDF
from prompt import *


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text("text")
    return text



# Response Format For my LLM Model
def Summarization_chain(input_text, tone, word_count):
    # Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=1, api_key=GOOGLE_API_KEY)  
    
    # Define the prompt
    PROMPT_TEMPLATE = PROMPT  # Imported
    prompt = PromptTemplate(
            input_variables=["text", "tone", "word_count"], # input in prompt
            template=PROMPT_TEMPLATE,
        )
      
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate response
    response = llm_chain.run({"text": input_text, "tone": tone, "word_count": word_count})
    return response


# Streamlit app
st.set_page_config(page_title="PDF Summarizer")
st.header("PDF Summarizer")

# PDF file upload
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Parameters
column_1, column_2 = st.columns([5, 5])

# Tone selection
with column_1:
    tone = st.selectbox("Select the tone", ["Formal", "Informal", "Friendly", "Professional"])

# Word count selection
with column_2:
    word_count = st.text_input("Number Of Words")

# Summarize button
if st.button("Summarize") and pdf_file:
    if word_count.isdigit():
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Ensure there's content to summarize
        if pdf_text.strip():
            response = Summarization_chain(input_text=pdf_text, tone=tone, word_count=int(word_count))
            st.write(" The Summary is : \n \n ", response)

            


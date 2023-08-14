
import os
import tempfile

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title='GPT Paper Reader')

# set text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=20,
    length_function=len
)

# get password
open_api_key = st.text_input('OpenAI API Token: ', type='password')

if open_api_key != '':
    os.environ['OPENAI_API_KEY'] = open_api_key

    # get model
    llm_model_name = st.radio('gpt-3.5-turbo', ['gpt-3.5-turbo', 'gpt-4'])
    temperature = st.number_input('temperature', min_value=0.0, value=0.7)
    max_tokens = st.number_input('max_tokens', min_value=1, value=600)
    llm_model = ChatOpenAI(temperature=temperature, model_name=llm_model_name, max_tokens=max_tokens)

    # Get uploaded file
    uploaded_pdffile = st.file_uploader('Upload a file (.pdf)')

    # action
    st.text('What to do?')
    action = st.radio('Summarize', ['Summarize', 'Question & Answering'])

    if (uploaded_pdffile is not None):
        pdfbytes = tempfile.NamedTemporaryFile()
        tempfilename = pdfbytes.name
        pdfbytes.write(uploaded_pdffile.read())

        st.text('Reading the file...')
        loader = PyPDFLoader(tempfilename)
        pages = loader.load_and_split(text_splitter=text_splitter)
        st.text('...done!')

        if action == 'Summarize':
            if st.button('Summarize'):
                chain = load_summarize_chain(llm=llm_model, chain_type='map_reduce')
                response = chain.run(pages)

                st.markdown(response)
        elif action == 'Question & Answering':
            st.text('Handling the file...')
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(pages, embeddings)
            st.text('...done!')

            question = st.text_area('Ask a question:')
            to_ask = st.button('Ask')
            if to_ask:
                retriever = db.as_retriever()
                qa = RetrievalQA.from_chain_type(
                    llm=llm_model,
                    chain_type='stuff',
                    retriever=retriever,
                    return_source_documents=False
                )
                response_json = qa({'query': question})
                st.markdown(response_json['result'])

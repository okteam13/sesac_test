__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="jocoding", floating=True, width=221)

#제목
st.title("ChatPDF")
st.write("---")

#OpenAI KEY 입력 받기
openai_key = st.text_input('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', type="password")

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])
st.write("---")

#PDF 파일 처리 
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() #임시 디렉토리 만들어서 저장 
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name) #업로드한 파일을 넣어줌 
    with open(temp_filepath, "wb") as f: #파일을 열어줌 
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath) #읽어온 PDF불러옴
    pages = loader.load_and_split()  #페이지 단위로 나누어서 pages 변수에 넣어줌 
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None: # 
    pages = pdf_to_document(uploaded_file)

    #Split, 텍스트 스플릿 
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding 진행 
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into Chroma, 실행때마다 매번 임베딩을 한다 
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!") #헤더 부분 출력 
    question = st.text_input('질문을 입력하세요')

    if st.button('질문하기'): #질문하기 버튼 실행, GPT 모델, 랭체인 이용하여 입력한 질문에 대한 답변  
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])

            
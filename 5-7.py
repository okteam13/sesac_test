from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

#Loader
loader = PyPDFLoader('unsu.pdf') 

# 업로드 대상PDF 
pages = loader.load_and_split() #PDF 쪼개기

#Split,텍스트 쪼개기
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 300,  ##몇글자 단위로 쪼갤건가
    chunk_overlap  = 20, #겹치는 부분 어느정도로 할건가
    length_function = len, #어느 길이로
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages) #쪼갠 텍스트 저장 

#Embedding
embeddings_model = OpenAIEmbeddings() #오픈AI 임베딩 모델 로드 

# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model) #임베딩한걸 크로마 DB에 저장 

#Question
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
#question = "김첨지가 누구야"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever()) #db에서 유사도 기반으로 답변 뱉어달라 
result = qa_chain({"query": question})
print(result)

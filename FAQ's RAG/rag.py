from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain


doc = PyPDFLoader("FAQ_SWAYAM.pdf")
pg = doc.load_and_split()
document = ""
llm = Ollama()



vector_db = FAISS.from_texts(pg[0].page_content,OllamaEmbeddings(base_url="http://localhost:11434",model="llama3",))    

prompt = ChatPromptTemplate.from_template("""
        Answer the following question in a simpler way and make it in a bullet points. 
         If you don't have proper answer then use this context. 
         <context>
         {context}
         </context>
         Question: {input}   
        """)
chain = create_stuff_documents_chain(llm,prompt)

retrevier = vector_db.as_retriever()
answer_chain = create_retrieval_chain(retrevier,chain)

question = "What is SWAYAM"

ans = answer_chain.invoke({"input":question})

print(ans['answer'])


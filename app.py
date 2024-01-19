import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    The chatbot needs to take user queries/input and return responses based on content from these PDF documents. The creativity used to respond to user queries/questions would be important.
    When no relevant results are found in the PDF, it should rely on content from an LLM to frame the response. If there is still no appropriate answer, it should provide a user-friendly response.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Specify the folder path directly
    pdf_folder = "D:\pdfchatbot\pdfs"
    pdf_docs = [os.path.join(pdf_folder, pdf_file) for pdf_file in os.listdir(pdf_folder) if pdf_file.endswith(".pdf")]

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

    # Update chat history
    chat_history.append({"User": user_question, "Bot": response["output_text"]})

    return chat_history

def main():
    st.set_page_config("Chat PDF")
    st.header("PDF Chat Bot")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Retrieve chat history from the session state
    chat_history = st.session_state.get("chat_history", [])

    if user_question:
        chat_history = user_input(user_question, chat_history)

    # Display chat history in the sidebar
    with st.sidebar:
        st.title("Chat History:")
        for entry in chat_history:
            st.text(f"User: {entry['User']}")
            st.text(f"Bot: {entry['Bot']}")
            st.text("----")

    # Save the updated chat history to the session state
    st.session_state.chat_history = chat_history

if __name__ == "__main__":
    main()

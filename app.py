import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import PyPDF2

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Job Recommender", page_icon="🤖", layout="wide")

# --- API KEY ---
# Aapko Streamlit Deploy karte waqt iski key secrets me daalni hogi.
# Abhi ke liye hum isko sidebar me input karwa lenge.
st.sidebar.title("Configuration")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# --- CORE LOGIC (Copied from Colab) ---

@st.cache_resource
def load_models_and_data():
    """This function loads the embedding model and job data, and creates the vector store."""
    # Load embedding model
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # Job Data
    jobs_data_text = """
    Job Title: Python Developer Intern
    Company: Tech Innovations Inc.
    Location: Bangalore
    Job Type: Internship
    Description: Looking for a Python developer intern to work on backend systems. Must know Python, Django, and basic database concepts. Good opportunity to learn about scalable systems.

    Job Title: Frontend Developer
    Company: Creative Solutions
    Location: Pune
    Job Type: Full-time
    Description: We need an experienced Frontend Developer proficient in React, HTML, CSS, and JavaScript. You will be responsible for building beautiful and responsive user interfaces. 3+ years of experience required.

    Job Title: Data Science Intern
    Company: Data Insights Co.
    Location: Work from Home
    Job Type: Internship
    Description: A great opportunity for students to get hands-on experience in data analysis, machine learning, and visualization. Required skills: Python, Pandas, Matplotlib, and basic knowledge of machine learning algorithms.

    Job Title: UI/UX Designer
    Company: Creative Solutions
    Location: Mumbai
    Job Type: Full-time
    Description: We are seeking a talented UI/UX designer to create amazing user experiences. The ideal candidate should have an eye for clean and artful design, possess superior UI/UX skills and be able to translate high-level requirements into interaction flows. Proficiency in Figma, Sketch, or Adobe XD is a must.

    Job Title: Backend Engineer - Java
    Company: FinTech Secure
    Location: Bangalore
    Job Type: Full-time
    Description: We are hiring a Backend Engineer with strong experience in Java, Spring Boot, and Microservices. You will be building the backbone of our financial services platform. Experience with AWS and Docker is a plus.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([jobs_data_text])
    vector_store = FAISS.from_documents(documents, hf_embeddings)
    return vector_store

def get_rag_chain(api_key):
    """Creates and returns the RAG chain."""
    llm = ChatGroq(api_key=api_key, model_name="Llama3-8b-8192")
    prompt_template = """
    You are a friendly and helpful career advisor for students.
    Your task is to recommend suitable jobs from the provided context based on the user's details.
    For each recommended job, provide a short, 2-3 line explanation of why it is a good match.
    If no jobs match, just say "I'm sorry, I couldn't find any suitable jobs in my current database."

    Context:
    {context}

    User's Details:
    {input}

    Your Recommendations:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- STREAMLIT UI ---
st.title("🤖 AI-Powered Job Recommendation System")
st.markdown("Discover your next career opportunity! [cite: 2, 4]")

# Load data and models
vector_store = load_models_and_data()

# Choose input method
input_method = st.radio("Choose your input method:", ("Enter Details Manually", "Upload Resume"))

user_query = ""

if input_method == "Enter Details Manually":
    st.subheader("Enter your details manually: [cite: 7]")
    skills = st.text_area("Your Skills (e.g., Python, React, Data Analysis)")
    experience = st.text_area("Your Experience (e.g., Fresher, 1 year in Java)")
    preferences = st.text_input("Your Preferences (e.g., Internship in Bangalore)")
    
    if st.button("Get Job Recommendations"):
        if not groq_api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        elif not skills:
            st.warning("Please enter your skills.")
        else:
            user_query = f"Skills: {skills}. Experience: {experience}. Preferences: {preferences}."

elif input_method == "Upload Resume":
    st.subheader("Upload your Resume (PDF only): [cite: 7]")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Get Job Recommendations"):
        if not groq_api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        elif uploaded_file is not None:
            try:
                resume_text = extract_text_from_pdf(uploaded_file)
                user_query = f"Based on this resume, find a suitable job: {resume_text}"
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
        else:
            st.warning("Please upload your resume.")

# --- Generate and Display Recommendations ---
if user_query:
    with st.spinner("Finding the best jobs for you..."):
        try:
            retrieval_chain = get_rag_chain(groq_api_key)
            response = retrieval_chain.invoke({"input": user_query})
            st.subheader("Here are your personalized recommendations: ")
            st.markdown(response["answer"])
        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your API key.")
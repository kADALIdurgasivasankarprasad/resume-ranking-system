import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import string
import fitz  # PyMuPDF for PDF processing
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import torch
from io import BytesIO

# Download NLTK Data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Load Pre-trained BERT model for embeddings
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

# Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Read from the uploaded file
    text = " ".join(page.get_text("text") for page in doc)
    return text.strip()

# Function to Clean Resume Text
def clean_text(text):
    """Clean text by removing URLs, mentions, special characters, and extra spaces."""
    text = re.sub(r'http\S+\s*', ' ', text)  # Remove URLs
    text = re.sub(r'RT|cc', ' ', text)  # Remove RT and cc
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # Remove mentions
    text = re.sub(f'[{string.punctuation}]', ' ', text)  # Remove punctuation
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Function to Generate WordCloud
def generate_wordcloud(text):
    return WordCloud(width=800, height=400, background_color='white').generate(text)

# Function to Convert Text to BERT Embeddings
def get_bert_embeddings(text_list):
    return bert_model.encode(text_list, convert_to_tensor=True)

# Streamlit UI
st.set_page_config(page_title="Resume Ranking System", page_icon="📄")
st.title("📄 Resume Ranking System")
st.write("Upload multiple PDFs containing resumes, input a job description, and rank resumes based on relevance.")

# Upload PDF Files
uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

# Job Description Input
job_description = st.text_area("Enter the Job Description Here")

# Number of Resumes Required Input
num_resumes = st.number_input("Number of Resumes Required", min_value=1, step=1, value=5)

# Compare & Rank Button
if st.button("Compare & Rank Resumes"):
    if uploaded_files and job_description:
        resumes = []
        file_names = []
        
        # Extract and Clean Text from PDFs
        for pdf in uploaded_files[:num_resumes]:  
            text = extract_text_from_pdf(pdf)
            cleaned_text = clean_text(text)
            resumes.append(cleaned_text)
            file_names.append(pdf.name)
        
        resumeDataSet = pd.DataFrame({"File Name": file_names, "Resume": resumes})
        
        # Generate BERT Embeddings for Job Description and Resumes
        all_texts = [job_description] + resumes  
        embeddings = get_bert_embeddings(all_texts)

        # Compute Cosine Similarity using BERT Embeddings
        job_desc_embedding = embeddings[0]  # First embedding is the job description
        resume_embeddings = embeddings[1:]  # Remaining rows are resumes

        similarity_scores = torch.nn.functional.cosine_similarity(job_desc_embedding, resume_embeddings, dim=-1)
        similarity_scores = (similarity_scores.cpu().numpy() * 100).round(2)  # Convert to percentage

        # Add Similarity Scores to DataFrame
        resumeDataSet['Similarity Score (%)'] = similarity_scores
        
        # Sort Resumes by Highest Similarity
        ranked_resumes = resumeDataSet.sort_values(by="Similarity Score (%)", ascending=False)

        # Display Results
        st.subheader("📌 Ranked Resumes")
        st.dataframe(ranked_resumes)
        
        # Download as Excel
        excel_buffer = BytesIO()
        ranked_resumes.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        st.download_button(label="Download as Excel", data=excel_buffer, file_name="ranked_resumes.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Generate Word Clouds
        st.subheader("📊 Word Clouds")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Job Description")
            job_wc = generate_wordcloud(job_description)
            plt.figure(figsize=(5, 5))
            plt.imshow(job_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        
        with col2:
            st.write("### Resumes")
            all_resumes_text = " ".join(resumes)
            resume_wc = generate_wordcloud(all_resumes_text)
            plt.figure(figsize=(5, 5))
            plt.imshow(resume_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)

    else:
        st.error("Please upload resumes and enter a job description before ranking.")
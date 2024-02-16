"""
app.py
"""
import pandas as pd
import streamlit as st

from skills_extraction import (
    extract_free_text_skills,
    ssg_skills_extraction
    )
from skills_similarity_search import similarity_search
from v1_ssg_sea import 

st.set_page_config(layout="wide")

st.title("Demo Skill Extraction")
with st.expander("About"):
    st.info("This is a two step process: \n \n 1. Extract skills from free text using LLM \n 2. Map these skills to SSG Skills Taxonomy using Semantic Search")

col1, col2, col3 = st.columns(3)

text = col1.text_area("Enter text to extract skills from", height=200)
gpt_model = col1.selectbox("Select GPT Model for Step 1", 
                           ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"])

if col1.button("Extract Skills"):
    
    col2.subheader("Step 1: LLM Extracted Skills")
    extracted_skills = extract_free_text_skills(gpt_model, text)
    col2.write(extracted_skills)

    col2.subheader("Step 2: Mapped to SSG")
    # load skills taxo - can replace with vector db
    df = pd.read_csv("data/skills_taxonomy_embeddings_v2.csv")
    col2.write(similarity_search(df, extracted_skills))
        
    col3.subheader("Baseline: SSG-SEA")
    ssg_sea_output = ssg_skills_extraction(text)
   
    col3.write(ssg_sea_output)

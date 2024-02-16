"""
load_skills.py
"""

import streamlit as st
from openai import OpenAI
from tqdm import tqdm

tqdm.pandas()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embeddings(text: str):
    """
    Get embeddings
    """
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding

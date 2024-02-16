"""
embed_skills_taxo.py
"""
import pandas as pd

from embeddings import get_embeddings


def build_skills_csv(file):
    """
    Build skills csv with embeddings
    """
    df = pd.read_excel(file)
    df["embedding"] = df["Skills"].progress_apply(get_embeddings)
    df_embeddings = df["embedding"].apply(pd.Series).add_prefix("embedding_")
    df = pd.concat([df["Skills"], df_embeddings], axis=1)
    df.to_csv(f"{file}_embedding_v2.csv", index=False)


build_skills_csv("data/skills_taxonomy.xlsx")

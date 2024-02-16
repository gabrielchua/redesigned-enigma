"""
similarity_search.py
"""
import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import get_embeddings

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def similarity_search(df, free_text_skills):
    """
    Search for similar skills
    """
    matched_skills = {}

    # loop through keys in extracted_skills
    matched_skills = []
    for skill in free_text_skills:
        # get embedding
        skill_embedding = get_embeddings(skill)
        skill_embedding = np.array(skill_embedding).reshape(1, -1)
        # calculate cosine similarity
        similarity_scores = cosine_similarity(df.iloc[:, 1:], skill_embedding)
        similarity_scores = similarity_scores.flatten()
        top_k_index = np.argsort(-similarity_scores)[:5]
        top_k_matched_skills = []
        for index in top_k_index:
            top_k_matched_skills.append(df.iloc[index, 0])
        # rerank the top_k
        best_match = _reranker(skill, top_k_matched_skills)
        matched_skills.append(best_match)

    return matched_skills


def _reranker(original_skill, candidate_skill_list):
    """
    Rerank the candidate skills
    """
    rerank_scores = []
    for candidate_skill in candidate_skill_list:
        rerank_scores.append(cross_encoder.predict([original_skill, candidate_skill]))
    rerank_scores = np.array(rerank_scores).flatten()
    best_match_index = np.argmax(rerank_scores)
    return candidate_skill_list[best_match_index]

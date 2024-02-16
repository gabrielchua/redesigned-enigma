"""
llm.py
"""
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from ssg_sea.extract_skills import extract_skills

azure_client = instructor.patch(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))

class IdentifiedSkills(BaseModel):
    """
    Extracted Skills
    """

    skills: list[str] = Field(min_items=0, max_items=30, description="Skills.")
    tools: list[str] = Field(
        min_items=0, max_items=30, description="Tools/Software/Frameworks."
    )

def extract_free_text_skills(gpt_model, text):
    """
    Identify skills
    """
    response = azure_client.chat.completions.create(
        model=gpt_model,
        response_model=IdentifiedSkills,
        messages=[
            {
                "role": "system",
                "content": "Extract the skills and tools from the following job description. EXTRACT VERBATIM AND DO NOT REPHRASE. Ignore text that do not describe the job responsibilities (e.g. admin instructions on application)",
            },
            {"role": "user", "content": text},
        ],
        seed=0,
        temperature=0,
    )

    extracted_skills = response.skills + response.tools

    return extracted_skills

def ssg_skills_extraction(text: str):
    """
    Extract skills using SSG-SEA
    """
    ssg_sea_output = extract_skills(text)

    skill_extract_list = []

    for skill in ssg_sea_output['extractions'].values():
        skill_extract = {}
        skill_extract['title'] = skill['skill_title']
        skill_extract['type'] = skill['skill_type']
        skill_extract_list.append(skill_extract)

    hard_skills = []
    soft_skills = []
    tools = []

    for skill in skill_extract_list:
        if skill['type'] == "TSC":
            hard_skills.append(skill['title'])
        elif skill['type'] == "CCS":
            soft_skills.append(skill['title'])
        else:
            tools.append(skill['title'])

    ssg_sea_output_flatted = {"hard_skills": hard_skills,
                                "soft_skills": soft_skills,
                                "tools": tools}

    return ssg_sea_output_flatted
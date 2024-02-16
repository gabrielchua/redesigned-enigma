"""
llm.py
"""
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

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

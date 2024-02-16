"""
v1_ssg_sea.py
"""
from ssg_sea.extract_skills import extract_skills

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

"""
Placement Agent — recommends workshops, internships, and placements.

- For students with CGPA 7.0-7.5 and no arrears
- Matches based on eligibility CGPA and skill alignment
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from factory import get_llm
from db import get_placements
from logger import setup_logger

logger = setup_logger("PlacementAgent")


# Removed module-level get_llm, now using factory.get_llm


def _find_eligible_opportunities(student: dict) -> list[dict]:
    """Find placement opportunities the student is eligible for."""
    placements = get_placements()
    eligible = []

    all_subjects = student["subjects"]["strong"] + student["subjects"].get("weak", [])

    for p in placements:
        if student["cgpa"] >= p["eligibility_cgpa"]:
            skill_overlap = set(p.get("skills_required", [])) & set(all_subjects)
            eligible.append({
                "opportunity": p,
                "matching_skills": list(skill_overlap),
                "skill_match_score": len(skill_overlap),
            })

    eligible.sort(key=lambda x: x["skill_match_score"], reverse=True)
    matches = eligible[:5]
    logger.info(f"Found {len(matches)} eligible opportunities for {student['name']}")
    return matches


def recommend_opportunities(student: dict) -> str:
    """Recommend workshops, internships, and placements for the student."""
    logger.info(f"Recommending opportunities for {student['name']} (CGPA: {student['cgpa']})")
    matches = _find_eligible_opportunities(student)

    if not matches:
        return f"No placement opportunities currently match {student['name']}'s profile. We recommend building skills through online courses and personal projects."

    opp_info = json.dumps([{
        "type": m["opportunity"]["type"],
        "company": m["opportunity"]["company"],
        "role": m["opportunity"]["role"],
        "description": m["opportunity"]["description"],
        "matching_skills": m["matching_skills"],
        "deadline": m["opportunity"]["deadline"],
    } for m in matches], indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Placement Advisory agent. Help the student understand which opportunities 
are best suited for them. Prioritize by relevance and explain why each opportunity is a good fit. 
Include preparation tips. Be encouraging and practical. Keep the response concise but informative."""),
        ("human", """Student Profile:
- Name: {name}
- CGPA: {cgpa}
- Department: {department}
- Year: {year}
- Strong Subjects: {strong_subjects}
- Placement Status: {placed}

Eligible Opportunities:
{opp_info}

Recommend the best opportunities for this student with explanations and preparation tips."""),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    result = chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "department": student["department"],
        "year": student["year"],
        "strong_subjects": ", ".join(student["subjects"]["strong"]),
        "placed": "Already placed" if student["placed"] else "Not yet placed",
        "opp_info": opp_info,
    })

    return result

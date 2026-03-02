"""
Mentor Agent — matches students with mentors based on academic profile.

- CGPA > 75% (7.5): Match top 3 domain experts aligned to strong subjects
- CGPA < 70% (7.0) or has arrears: Match top 3 mentors for weak/arrear subjects
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from factory import get_llm
from db import get_mentors
from logger import setup_logger

logger = setup_logger("MentorAgent")


# Removed module-level get_llm, now using factory.get_llm


def _find_matching_mentors(student: dict, match_type: str) -> list[dict]:
    """Find mentors that match the student's needs."""
    mentors = get_mentors()
    available_mentors = [m for m in mentors if m["availability"]]

    if match_type == "domain_expert":
        # For high CGPA: match on strong subjects
        target_subjects = student["subjects"]["strong"]
    else:
        # For low CGPA / arrears: match on weak subjects and arrears
        target_subjects = list(set(student["subjects"]["weak"] + student.get("arrears", [])))

    scored = []
    for mentor in available_mentors:
        overlap = set(mentor["core_subjects"]) & set(target_subjects)
        if overlap:
            scored.append({
                "mentor": mentor,
                "matching_subjects": list(overlap),
                "score": len(overlap),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    matches = scored[:3]
    logger.info(f"Found {len(matches)} matches for {student['name']} (Type: {match_type})")
    return matches


def match_domain_experts(student: dict) -> str:
    """For CGPA > 7.5: Match top 3 domain expert mentors."""
    logger.info(f"Matching domain experts for {student['name']} (CGPA: {student['cgpa']})")
    matches = _find_matching_mentors(student, "domain_expert")

    if not matches:
        return f"No domain expert mentors currently available for {student['name']}'s areas of strength."

    mentor_info = json.dumps([{
        "name": m["mentor"]["name"],
        "domain": m["mentor"]["domain"],
        "expertise": m["mentor"]["expertise_level"],
        "matching_subjects": m["matching_subjects"],
    } for m in matches], indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Smart Mentor matching agent. A high-performing student needs domain expert mentors 
to further excel in their strong areas. Generate a personalized recommendation explaining why each 
mentor is a great fit. Be encouraging and specific about how the mentorship can help the student grow.
Keep the response concise but informative."""),
        ("human", """Student Profile:
- Name: {name}
- CGPA: {cgpa}
- Department: {department}
- Strong Subjects: {strong_subjects}

Top Matching Mentors:
{mentor_info}

Generate a personalized recommendation for each mentor match, explaining why they're ideal for this student."""),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "department": student["department"],
        "strong_subjects": ", ".join(student["subjects"]["strong"]),
        "mentor_info": mentor_info,
    })

    return result


def match_core_subject_mentors(student: dict) -> str:
    """For CGPA < 7.0 or has arrears: Match top 3 mentors for improvement areas."""
    logger.info(f"Matching core subject mentors for {student['name']} (CGPA: {student['cgpa']})")
    matches = _find_matching_mentors(student, "core_subject")

    if not matches:
        return f"No mentors currently available for {student['name']}'s areas needing improvement."

    mentor_info = json.dumps([{
        "name": m["mentor"]["name"],
        "domain": m["mentor"]["domain"],
        "expertise": m["mentor"]["expertise_level"],
        "matching_subjects": m["matching_subjects"],
    } for m in matches], indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Smart Mentor matching agent. A student needs academic support to improve in 
their weak areas and clear arrears. Generate a personalized improvement plan with mentor recommendations.
Be supportive, empathetic, and action-oriented. Include specific study strategies for each weak area.
Keep the response concise but informative."""),
        ("human", """Student Profile:
- Name: {name}
- CGPA: {cgpa}
- Department: {department}
- Weak Subjects: {weak_subjects}
- Arrears: {arrears}

Top Matching Mentors:
{mentor_info}

Generate a personalized improvement plan with mentor recommendations for each weak area."""),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "department": student["department"],
        "weak_subjects": ", ".join(student["subjects"]["weak"]),
        "arrears": ", ".join(student.get("arrears", [])) or "None",
        "mentor_info": mentor_info,
    })

    return result

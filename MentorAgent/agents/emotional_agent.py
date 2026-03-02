"""
Emotional Agent — analyzes student emotional state and provides support.

- Triggered when student is placed/attendance < 75/failing or expresses distress
- Analyzes stress level (low/medium/high)
- Low stress → Self-improvement plan
- High stress → Flag for parent & mentor notification
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import GOOGLE_API_KEY, MODEL_NAME
from logger import setup_logger

logger = setup_logger("EmotionalAgent")


def get_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
    )


def analyze_emotion(student: dict, user_message: str) -> dict:
    """Analyze the emotional state of a student based on their message and academic context."""
    logger.info(f"Analyzing emotion for {student['name']}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an empathetic emotional analysis agent for a student mentoring system.
Analyze the student's emotional state based on their message and academic context.

You MUST respond in this exact JSON format and nothing else:
{{
    "stress_level": "low" or "medium" or "high",
    "emotional_state": "brief description of detected emotions",
    "key_concerns": ["list", "of", "concerns"],
    "confidence": 0.0 to 1.0
}}

Factors to consider:
- Academic pressure (low CGPA, arrears, attendance issues)
- Explicit emotional language in the message
- Overall academic trajectory
- Social and personal indicators"""),
        ("human", """Student Profile:
- Name: {name}
- CGPA: {cgpa}
- Attendance: {attendance}%
- Arrears: {arrears}
- Placed: {placed}
- Emotional Notes: {emotional_notes}

Student's Message: "{message}"

Analyze their emotional state and return JSON."""),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    result = chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "attendance": student["attendance_pct"],
        "arrears": ", ".join(student.get("arrears", [])) or "None",
        "placed": "Yes" if student["placed"] else "No",
        "emotional_notes": student.get("emotional_notes", "No prior notes"),
        "message": user_message,
    })

    try:
        # Clean up the response — strip markdown code fences if present
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # Remove first line
            cleaned = cleaned.rsplit("```", 1)[0]  # Remove closing fence
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return {
            "stress_level": "medium",
            "emotional_state": "Unable to fully assess — defaulting to supportive response",
            "key_concerns": ["academic_pressure"],
            "confidence": 0.3,
        }


def generate_support_response(student: dict, emotion_analysis: dict, user_message: str) -> str:
    """Generate an appropriate support response based on emotional analysis."""
    stress_level = emotion_analysis.get("stress_level", "medium")
    logger.info(f"Generating support for {student['name']} (Stress: {stress_level})")

    if stress_level == "high":
        return _generate_high_stress_response(student, emotion_analysis, user_message)
    else:
        return _generate_improvement_plan(student, emotion_analysis, user_message)


def _generate_high_stress_response(student: dict, emotion_analysis: dict, user_message: str) -> str:
    """For high stress: provide immediate support and flag for parent/mentor notification."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate student counselor. The student is experiencing HIGH STRESS.
Your response should:
1. Acknowledge their feelings with empathy
2. Provide immediate coping strategies
3. Reassure them that help is available
4. Mention that their mentor and parents will be informed for additional support (frame this positively)
5. Suggest professional counseling resources if appropriate

Be warm, supportive, and non-judgmental. Keep the response concise but caring."""),
        ("human", """Student: {name} (CGPA: {cgpa}, Attendance: {attendance}%)
Emotional Analysis: {analysis}
Their message: "{message}"

Provide a supportive response."""),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "attendance": student["attendance_pct"],
        "analysis": json.dumps(emotion_analysis),
        "message": user_message,
    })


def _generate_improvement_plan(student: dict, emotion_analysis: dict, user_message: str) -> str:
    """For low/medium stress: encourage with a self-improvement plan."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an encouraging academic advisor. The student has low to moderate stress.
Your response should:
1. Acknowledge their situation positively
2. Create a practical self-improvement plan with small, achievable goals
3. Highlight their strengths and potential
4. Suggest time management and study strategies
5. Be motivating and optimistic

Keep the response concise, actionable, and uplifting."""),
        ("human", """Student: {name} (CGPA: {cgpa}, Attendance: {attendance}%)
Strong Subjects: {strong}
Weak Subjects: {weak}
Emotional Analysis: {analysis}
Their message: "{message}"

Create an encouraging self-improvement plan."""),
    ])

    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({
        "name": student["name"],
        "cgpa": student["cgpa"],
        "attendance": student["attendance_pct"],
        "strong": ", ".join(student["subjects"]["strong"]),
        "weak": ", ".join(student["subjects"]["weak"]),
        "analysis": json.dumps(emotion_analysis),
        "message": user_message,
    })

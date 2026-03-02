"""
Orchestrator — LangChain Agent that routes queries to the right sub-agent.

Uses Gemini as the LLM with custom tools for:
1. Looking up student data
2. Evaluating student academic standing (flow-diagram rules)
3. Mentor matching
4. Placement recommendations
5. Emotional support
"""

import json
from factory import get_llm
from langchain_core.tools import tool
from logger import setup_logger

logger = setup_logger("Orchestrator")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# from factory import get_llm handled LLM setup
from db import (
    get_student, get_all_students, get_conversation, save_conversation,
)
from agents.mentor_agent import match_domain_experts, match_core_subject_mentors
from agents.placement_agent import recommend_opportunities
from agents.emotional_agent import analyze_emotion, generate_support_response


# ──────────────────────────────────────────────
# Define Tools for the LangChain Agent
# ──────────────────────────────────────────────

@tool
def lookup_student(student_id: str) -> str:
    """Look up a student's full profile by their student ID (e.g. S001, S002)."""
    logger.info(f"Tool Call: lookup_student | ID: {student_id}")
    student = get_student(student_id)
    if student:
        return json.dumps(student, indent=2)
    return f"No student found with ID: {student_id}"


@tool
def list_all_students() -> str:
    """List all students in the system."""
    logger.info("Tool Call: list_all_students")
    students = get_all_students()
    summary = []
    for s in students:
        summary.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "cgpa": s["cgpa"],
            "attendance_pct": s["attendance_pct"],
            "arrears_count": len(s.get("arrears", [])),
            "placed": s["placed"],
        })
    return json.dumps(summary, indent=2)


@tool
def evaluate_student_status(student_id: str) -> str:
    """Evaluate academic standing and determine action path."""
    logger.info(f"Tool Call: evaluate_student_status | ID: {student_id}")
    student = get_student(student_id)
    if not student:
        return f"No student found with ID: {student_id}"

    evaluation = {
        "student_id": student["student_id"],
        "name": student["name"],
        "cgpa": student["cgpa"],
        "attendance_pct": student["attendance_pct"],
        "arrears": student.get("arrears", []),
        "placed": student["placed"],
        "flags": [],
        "recommended_actions": [],
    }

    # Rule 1: Attendance < 75%
    if student["attendance_pct"] < 75:
        evaluation["flags"].append("LOW_ATTENDANCE")
        evaluation["recommended_actions"].append(
            "NOTIFY_PARENT_MENTOR: Attendance is below 75%. Intimate parent and mentor."
        )

    # Rule 2: CGPA > 7.5 (75%)
    if student["cgpa"] > 7.5:
        evaluation["flags"].append("HIGH_PERFORMER")
        evaluation["recommended_actions"].append(
            "MENTOR_DOMAIN_EXPERT: CGPA > 7.5. Use get_mentor_domain_experts tool to match top 3 domain expert mentors."
        )

    # Rule 3: CGPA between 7.0 and 7.5
    if 7.0 <= student["cgpa"] <= 7.5:
        if student.get("arrears"):
            evaluation["flags"].append("HAS_ARREARS")
            evaluation["recommended_actions"].append(
                "MENTOR_CORE_SUBJECTS: Has arrears despite decent CGPA. Use get_mentor_core_subjects tool for improvement plan."
            )
        else:
            evaluation["flags"].append("PLACEMENT_ELIGIBLE")
            evaluation["recommended_actions"].append(
                "PLACEMENT_AGENT: CGPA 7.0-7.5 with no arrears. Use get_placement_recommendations tool."
            )

    # Rule 4: CGPA < 7.0
    if student["cgpa"] < 7.0:
        evaluation["flags"].append("NEEDS_ACADEMIC_SUPPORT")
        evaluation["recommended_actions"].append(
            "MENTOR_CORE_SUBJECTS: CGPA < 7.0. Use get_mentor_core_subjects tool for core subject mentoring."
        )

    # Rule 5: Emotional support trigger
    if student["placed"] or student["attendance_pct"] < 75 or student["cgpa"] < 5.0:
        evaluation["flags"].append("EMOTIONAL_CHECK_NEEDED")
        evaluation["recommended_actions"].append(
            "EMOTIONAL_AGENT: Student may need emotional support. Use emotional_support tool if they express distress."
        )

    return json.dumps(evaluation, indent=2)


@tool
def get_mentor_domain_experts(student_id: str) -> str:
    """Match high-performing student with domain experts."""
    logger.info(f"Tool Call: get_mentor_domain_experts | ID: {student_id}")
    student = get_student(student_id)
    if not student:
        return f"No student found with ID: {student_id}"
    return match_domain_experts(student)


@tool
def get_mentor_core_subjects(student_id: str) -> str:
    """Match struggling student with core subject mentors."""
    logger.info(f"Tool Call: get_mentor_core_subjects | ID: {student_id}")
    student = get_student(student_id)
    if not student:
        return f"No student found with ID: {student_id}"
    return match_core_subject_mentors(student)


@tool
def get_placement_recommendations(student_id: str) -> str:
    """Get placement/workshop recommendations."""
    logger.info(f"Tool Call: get_placement_recommendations | ID: {student_id}")
    student = get_student(student_id)
    if not student:
        return f"No student found with ID: {student_id}"
    return recommend_opportunities(student)


@tool
def emotional_support(student_id: str, user_message: str) -> str:
    """Provide emotional support to a student."""
    logger.info(f"Tool Call: emotional_support | ID: {student_id}")
    student = get_student(student_id)
    if not student:
        return f"No student found with ID: {student_id}"

    emotion_analysis = analyze_emotion(student, user_message)
    support_response = generate_support_response(student, emotion_analysis, user_message)

    actions = []
    stress_level = emotion_analysis.get("stress_level", "medium")
    if stress_level == "high":
        actions.append(f"⚠️ HIGH STRESS ALERT: Notification sent to parent ({student['email']}) and assigned mentor.")
        actions.append(f"Recommended: Schedule counseling session for {student['name']}.")

    result = {
        "emotion_analysis": emotion_analysis,
        "support_response": support_response,
        "actions_taken": actions,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
# Build the LangChain Agent
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are SmartMentor AI — an intelligent student mentoring system for a college.
You help with academic guidance, mentor matching, placement support, and emotional well-being.

WORKFLOW — follow this process for every student-related query:
1. First, identify the student (by ID or name). Use list_all_students if needed to find them.
2. Use lookup_student to get their full profile.
3. Use evaluate_student_status to understand their academic standing and what actions to take.
4. Based on the evaluation, call the appropriate tool(s):
   - HIGH CGPA (>7.5): Use get_mentor_domain_experts for advanced mentoring
   - LOW CGPA (<7.0) or ARREARS: Use get_mentor_core_subjects for academic support
   - MID CGPA (7.0-7.5) + NO ARREARS: Use get_placement_recommendations
   - EMOTIONAL DISTRESS or AT-RISK: Use emotional_support
5. Synthesize all information into a clear, helpful response.

IMPORTANT RULES:
- Always be empathetic and supportive in your responses.
- When a student seems stressed or expresses difficulty, ALWAYS use the emotional_support tool.
- You may use multiple tools for a single query if the situation warrants it.
- Include specific, actionable recommendations.
- Reference actual data from the tools (mentor names, opportunity details, etc.).
"""

TOOLS = [
    lookup_student,
    list_all_students,
    evaluate_student_status,
    get_mentor_domain_experts,
    get_mentor_core_subjects,
    get_placement_recommendations,
    emotional_support,
]


def create_agent():
    """Create and return the LangGraph agent using the factory LLM."""
    llm = get_llm(temperature=0.4)
    agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT)
    return agent


def build_messages_from_history(history: list[dict]) -> list:
    """Convert stored conversation history to LangChain message objects."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


async def run_agent(question: str, conversation_id: str, student_id: str | None = None) -> dict:
    """Run the orchestrator agent with conversation memory."""
    logger.info(f"Agent Execution Start | Conv: {conversation_id} | Student: {student_id}")
    agent = create_agent()

    # Load conversation history
    history = get_conversation(conversation_id)
    messages = build_messages_from_history(history)

    # If student_id is provided, prepend it to the question for context
    full_question = question
    if student_id:
        full_question = f"[Student ID: {student_id}] {question}"

    messages.append(HumanMessage(content=full_question))

    # Run the agent
    logger.info("Invoking LangGraph agent...")
    result = await agent.ainvoke({"messages": messages})
    logger.info("Agent invocation complete")

    # Extract the final response
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    final_response = ai_messages[-1].content if ai_messages else "I couldn't process that request."

    # Determine which agent was used based on tool calls
    tool_calls_made = []
    for m in result["messages"]:
        if hasattr(m, "tool_calls") and m.tool_calls:
            tool_calls_made.extend([tc["name"] for tc in m.tool_calls])

    agent_used = _determine_agent_used(tool_calls_made)
    actions = _extract_actions(tool_calls_made)
    logger.info(f"Execution Summary | Agent: {agent_used} | Actions: {actions}")

    # Save conversation history
    history.append({"role": "user", "content": full_question})
    history.append({"role": "assistant", "content": final_response})
    save_conversation(conversation_id, history)

    # Get student summary if a student was involved
    student_summary = None
    if student_id:
        student = get_student(student_id)
        if student:
            student_summary = {
                "name": student["name"],
                "cgpa": student["cgpa"],
                "attendance": student["attendance_pct"],
                "placed": student["placed"],
            }

    return {
        "response": final_response,
        "agent_used": agent_used,
        "actions_taken": actions,
        "student_summary": student_summary,
    }


def _determine_agent_used(tool_calls: list[str]) -> str:
    """Determine which primary agent handled the request."""
    if "emotional_support" in tool_calls:
        return "Emotional Agent"
    elif "get_mentor_domain_experts" in tool_calls:
        return "Mentor Agent (Domain Experts)"
    elif "get_mentor_core_subjects" in tool_calls:
        return "Mentor Agent (Core Subjects)"
    elif "get_placement_recommendations" in tool_calls:
        return "Placement Agent"
    elif "evaluate_student_status" in tool_calls:
        return "Orchestrator (Evaluation)"
    elif "lookup_student" in tool_calls or "list_all_students" in tool_calls:
        return "Orchestrator (Lookup)"
    else:
        return "Orchestrator (General)"


def _extract_actions(tool_calls: list[str]) -> list[str]:
    """Extract human-readable actions from tool calls."""
    actions = []
    action_map = {
        "lookup_student": "Looked up student profile",
        "list_all_students": "Listed all students",
        "evaluate_student_status": "Evaluated academic standing",
        "get_mentor_domain_experts": "Matched with domain expert mentors",
        "get_mentor_core_subjects": "Matched with core subject mentors",
        "get_placement_recommendations": "Generated placement recommendations",
        "emotional_support": "Provided emotional support & analysis",
    }
    seen = set()
    for tc in tool_calls:
        if tc in action_map and tc not in seen:
            actions.append(action_map[tc])
            seen.add(tc)
    return actions

# Smart Mentor Agent — AI-Powered Student Success System

An intelligent, multi-agent conversational system designed to provide personalized academic guidance, placement support, and emotional well-being monitoring for college students.

---

## 🌟 Pitch: Why Smart Mentor?

In modern education, students often feel overwhelmed by academic pressure, lack of personalized career guidance, and emotional stress. **Smart Mentor** bridges this gap by acting as a 24/7 digital advisor that:
- **Intelligently routes** students to the right resources based on their real-time academic profile.
- **Matches top performers** with domain experts to push their boundaries.
- **Identifies struggling students** early and provides targeted academic intervention.
- **Proactively monitors emotional health**, ensuring no student feels alone during tough times.

---

## 🚀 Key Features

### 1. 🧠 Autonomous Orchestration (The Brain)
Using **LangChain and LangGraph**, our system intelligently analyzes user queries and student data. It doesn't just answer; it **reasons** about the student's academic standing and selects the most appropriate sub-agent for the task.

### 2. 🎓 Specialized AI Agents
- **Mentor Agent**: 
    - *High-Performers (>7.5 CGPA)*: Matches with top-tier domain experts in their strong subjects.
    - *Academic Support (<7.0 CGPA)*: Matches with core subject specialists and creates personalized improvement plans.
- **Placement Agent**: Recommends workshops, internships, and placement opportunities tailored to the student's skills and eligibility (7.0-7.5 CGPA).
- **Emotional Agent**: High-empathy analyzer that detects academic or personal stress. In high-stress scenarios, it automatically alerts parents and assigned mentors for human intervention.


## 🛠️ Technical Stack

- **Framework**: FastAPI (Backend API)
- **AI Core**: LangChain & LangGraph
- **LLM**: Google Gemini (Directly integrated via `langchain-google-genai`)
- **Data Layer**: JSON-based persistent storage (No external database setup required for PoC)
- **Environment**: Python 3.10+

---

## 📂 Project Structure

```text
MentorAgent/
├── main.py              # FastAPI Application Entry Point
├── orchestrator.py      # Core AI Agent Routing (LangGraph)
├── agents/              # Specialized Agent Modules
│   ├── mentor_agent.py
│   ├── placement_agent.py
│   └── emotional_agent.py
├── data/                # JSON Database (Students, Mentors, Placements)
│   ├── students.json
│   ├── mentors.json
│   └── placements.json
├── db.py                # Database helper functions
├── logger.py            # Centralized logging configuration
├── config.py            # System configuration & .env loader
└── .env                 # API Credentials (Protected)
```

---

## 🚦 Getting Started

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create or edit the `.env` file and add your Google Gemini API key:
```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the Server
```bash
uvicorn main.py:app --reload --port 8000
```

### 4. Interactive Docs
Visit `http://localhost:8000/docs` to test the API directly through the Swagger UI.

---

## 👥 For Developers
We use a **ReAct (Reason + Act) pattern** for the orchestrator. If you wish to add new tools or agents, simply define a new tool in `orchestrator.py` and decorate it with `@tool`.

## 📈 For Business Users
This PoC demonstrates the power of **Agentic AI** in education. By automating the first line of student support, institutions can scale their mentorship programs while ensuring high-quality, data-driven outcomes for every student.

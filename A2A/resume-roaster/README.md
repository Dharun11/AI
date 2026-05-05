# Resume Roaster: Multi-Agent A2A Resume Reviewer

A production-style, multi-agent resume review system that scores a resume from two lenses in parallel:

- Engineer Lens: technical depth, impact, architecture quality
- HR Lens: recruiter-first screening, ATS clarity, timeline consistency

Both agents are exposed through A2A-style HTTP+JSON endpoints, orchestrated in parallel, and visualized in a Streamlit UI.

---

## Why this project stands out

This is not a single prompt script. It demonstrates practical agent engineering patterns:

- Multi-agent architecture with clear separation of concerns
- A2A protocol-style service boundaries between agents
- Parallel fan-out and merge orchestration
- Tool-based reasoning:
  - Engineer: LangChain ReAct + tool calling
  - HR: Semantic Kernel ReAct loop + deterministic tool observations
- Structured outputs and validation via Pydantic
- End-to-end UI for demos and portfolio sharing

---

## Project structure

```text
resume-roaster/
  agents/
    engineer_agent/
      a2a_server.py
      executor.py
      main.py
      tools.py
    hr_agent/
      a2a_server.py
      executor.py
      main.py
      tools.py
    orchestrator/
      main.py
      tools.py
  app.py
  requirements.txt
```

---

## End-to-end flow

1. User pastes or uploads resume text in Streamlit.
2. Streamlit calls orchestrator.
3. Orchestrator sends A2A messages to Engineer and HR agents in parallel.
4. Engineer agent runs LangChain ReAct and calls technical tools.
5. HR agent runs Semantic Kernel ReAct loop and calls HR tools.
6. Each agent returns a structured artifact.
7. Orchestrator merges both artifacts into one final verdict.
8. Streamlit displays scores, criticisms, strengths, and raw JSON.

---

## Tech stack

- Python
- FastAPI
- LangChain
- Semantic Kernel
- Azure OpenAI / OpenAI
- Streamlit
- httpx
- Pydantic

---

## Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd AI/A2A/resume-roaster
```

### 2) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\.venv\Scripts\Activate.ps1")
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a .env file in project root:

```env
# Provider selection (default is azure)
LLM_PROVIDER=azure

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=your_default_chat_deployment

# Optional per-agent Azure override
ENGINEER_AGENT_AZURE_DEPLOYMENT=your_engineer_deployment
HR_AGENT_AZURE_DEPLOYMENT=your_hr_deployment

# OpenAI (only needed when provider=openai)
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://your-compatible-endpoint/v1

# Optional per-agent provider/model override
# ENGINEER_AGENT_LLM_PROVIDER=openai
# HR_AGENT_LLM_PROVIDER=openai
# ENGINEER_AGENT_OPENAI_MODEL=gpt-4o-mini
# HR_AGENT_OPENAI_MODEL=gpt-4o-mini
```

---

## Run the project

Use 3 terminals.

### Terminal 1: Engineer agent

```bash
python -m agents.engineer_agent.executor --serve --host 127.0.0.1 --port 8102
```

### Terminal 2: HR agent

```bash
python -m agents.hr_agent.executor --serve --host 127.0.0.1 --port 8103
```

### Terminal 3: Streamlit UI

```bash
streamlit run app.py
```

Open the Streamlit URL shown in terminal (usually <http://localhost:8501>).

---

## Health checks

Before running UI, validate both agent cards:

- <http://127.0.0.1:8102/.well-known/agent-card.json>
- <http://127.0.0.1:8103/.well-known/agent-card.json>

If both return JSON, orchestration should work.

---

## API endpoints

Both agents expose:

- GET /.well-known/agent-card.json
- POST /message:send
- GET /tasks/{task_id}

Example request body for /message:send:

```json
{
  "message": {
    "messageId": "uuid",
    "role": "ROLE_USER",
    "parts": [
      {
        "text": "Paste resume text here",
        "mediaType": "text/plain"
      }
    ]
  }
}
```

---

## What each agent does

### Engineer agent (LangChain ReAct)

Focus:

- stack currency
- measurable impact claims
- project/system depth
- public proof of work

Output:

- score_out_of_10
- technical_criticisms (exactly 3)
- genuine_strength
- tool_signals

### HR agent (Semantic Kernel ReAct)

Focus:

- ATS keyword strength
- bullet language quality
- date/timeline consistency
- resume length/readability band

Output:

- score_out_of_10
- hr_criticisms (exactly 3)
- genuine_strength
- tool_signals

---

## Orchestrator logic

The orchestrator:

- calls both agents concurrently
- extracts artifacts
- computes blended score: (Engineer + HR) / 2
- verdict:
  - interview if blended score >= 6.5
  - reject otherwise

---

## Troubleshooting

### Streamlit says run failed

Check:

- both agent servers are running
- URLs in UI are correct
- environment variables are loaded

### Azure OpenAI errors

Check:

- key, endpoint, api version
- deployment names exist
- model/deployment supports your parameters

### Port already in use

Use different ports and update UI fields accordingly.

---

## LinkedIn-ready project pitch

Use this as your project highlight:

I built a multi-agent resume intelligence system using LangChain, Semantic Kernel, FastAPI, and Azure OpenAI. The system runs Engineer and HR agents in parallel via A2A-style APIs, then merges both perspectives into a single hiring verdict. It includes deterministic tool-based scoring, structured outputs with validation, and a Streamlit UI for real-time evaluation.

### Suggested LinkedIn headline line

Built a Multi-Agent Resume Reviewer with LangChain ReAct, Semantic Kernel ReAct, A2A APIs, and Streamlit.

### Suggested post hashtags

- #AIEngineering
- #MultiAgentSystems
- #LangChain
- #SemanticKernel
- #AzureOpenAI
- #FastAPI
- #Streamlit
- #Python
- #A2A
- #LLMOps

---

## Resume bullet you can use

Designed and implemented a multi-agent resume evaluation platform with parallel A2A orchestration, combining LangChain and Semantic Kernel ReAct agents with deterministic tool signals to produce validated hiring recommendations in real time.

---

## Future improvements

- add full trace viewer for agent tool calls
- store run history in a database
- add authentication and rate limiting
- add PDF report export
- deploy on cloud with CI/CD

---

## License

Add your preferred license (MIT is common for portfolio projects).

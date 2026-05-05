"""Engineer agent main logic.

Flow:
1) Run a ReAct-style LangChain agent that can call 4 technical tools.
2) Synthesize final judgment with structured output.
3) Return A2A-style Artifact payload.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field

from .tools import (
	collect_signals,
	evaluate_impact_claims,
	evaluate_project_depth,
	evaluate_public_work,
	evaluate_stack_currency,
)


load_dotenv()


class EngineerVerdict(BaseModel):
	score_out_of_10: float = Field(ge=0, le=10)
	technical_criticisms: List[str] = Field(min_length=3, max_length=3)
	genuine_strength: str


class EngineerResumeAnalyzer:
	"""Single-responsibility service to evaluate resume technical quality."""

	def __init__(self, model_name: str | None = None, temperature: float = 0.1) -> None:
		provider = (os.getenv("ENGINEER_AGENT_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "azure").strip().lower()

		if provider == "openai":
			openai_api_key = os.getenv("OPENAI_API_KEY")
			openai_model = model_name or os.getenv("ENGINEER_AGENT_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
			openai_base_url = os.getenv("OPENAI_BASE_URL")

			if not openai_api_key:
				raise ValueError("OPENAI_API_KEY is required when ENGINEER_AGENT_LLM_PROVIDER=openai.")
			if not openai_model:
				raise ValueError("Set ENGINEER_AGENT_OPENAI_MODEL or OPENAI_MODEL for engineer agent.")

			effective_temperature = temperature
			if "gpt-5" in openai_model.lower() and temperature != 1:
				effective_temperature = 1

			self._model = ChatOpenAI(
				api_key=openai_api_key,
				model=openai_model,
				base_url=openai_base_url,
				temperature=effective_temperature,
			)
		else:
			azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
			azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
			azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
			azure_deployment = model_name or os.getenv("ENGINEER_AGENT_AZURE_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

			if not azure_api_key:
				raise ValueError("AZURE_OPENAI_API_KEY is required for engineer agent.")
			if not azure_endpoint:
				raise ValueError("AZURE_OPENAI_ENDPOINT is required for engineer agent.")
			if not azure_deployment:
				raise ValueError(
					"Set ENGINEER_AGENT_AZURE_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT for engineer agent."
				)

			# Some GPT-5 Azure deployments only support default temperature (1).
			effective_temperature = temperature
			if "gpt-5" in azure_deployment.lower() and temperature != 1:
				effective_temperature = 1

			self._model = AzureChatOpenAI(
				azure_endpoint=azure_endpoint,
				api_key=azure_api_key,
				api_version=azure_api_version,
				azure_deployment=azure_deployment,
				temperature=effective_temperature,
			)
		self._react_agent = create_agent(
			model=self._model,
			tools=[
				evaluate_stack_currency,
				evaluate_impact_claims,
				evaluate_project_depth,
				evaluate_public_work,
			],
			response_format=EngineerVerdict,
			system_prompt=(
				"You are a skeptical senior software engineer who has run 200+ interview loops. "
				"You dislike vague claims and buzzwords. Before giving a final answer, call each of these tools "
				"once with the same resume_text: evaluate_stack_currency, evaluate_impact_claims, "
				"evaluate_project_depth, evaluate_public_work. Then return the final structured verdict."
			),
		)

	def analyze(self, resume_text: str) -> Dict[str, object]:
		result = self._react_agent.invoke(
			{
				"messages": [
					{
						"role": "user",
						"content": (
							"Evaluate this resume for technical interview-slot worthiness. "
							"Use the available tools first, then return final structured output.\n\n"
							f"resume_text:\n{resume_text}"
						),
					}
				]
			}
		)

		verdict = self._coerce_verdict(result)
		signals = self._extract_signals_from_messages(result.get("messages", []))
		if not signals:
			signals = collect_signals(resume_text).to_dict()
		return self._build_a2a_artifact(verdict, signals)

	@staticmethod
	def _coerce_verdict(result: Dict[str, Any]) -> EngineerVerdict:
		structured = result.get("structured_response")
		if isinstance(structured, EngineerVerdict):
			return structured
		if isinstance(structured, dict):
			return EngineerVerdict.model_validate(structured)

		messages = result.get("messages", [])
		if messages:
			last = messages[-1]
			content = getattr(last, "content", "") if not isinstance(last, dict) else last.get("content", "")
			if isinstance(content, str):
				blob = EngineerResumeAnalyzer._extract_json_blob(content)
				if blob:
					return EngineerVerdict.model_validate(json.loads(blob))

		raise ValueError("Could not parse EngineerVerdict from ReAct agent output.")

	@staticmethod
	def _extract_json_blob(text: str) -> str | None:
		candidate = text.strip()
		if candidate.startswith("```"):
			candidate = re.sub(r"^```[a-zA-Z]*\n", "", candidate)
			candidate = re.sub(r"\n```$", "", candidate)
		try:
			json.loads(candidate)
			return candidate
		except Exception:
			pass

		start = candidate.find("{")
		end = candidate.rfind("}")
		if start >= 0 and end > start:
			maybe = candidate[start : end + 1]
			try:
				json.loads(maybe)
				return maybe
			except Exception:
				return None
		return None

	@staticmethod
	def _extract_signals_from_messages(messages: List[Any]) -> Dict[str, object]:
		signals: Dict[str, object] = {}
		name_map = {
			"evaluate_stack_currency": "stack_signal",
			"evaluate_impact_claims": "impact_signal",
			"evaluate_project_depth": "depth_signal",
			"evaluate_public_work": "public_work_signal",
		}

		for msg in messages:
			msg_type = getattr(msg, "type", None) if not isinstance(msg, dict) else msg.get("type")
			if msg_type != "tool":
				continue

			tool_name = getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
			key = name_map.get(tool_name)
			if not key:
				continue

			content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
			parsed = EngineerResumeAnalyzer._parse_tool_content(content)
			if isinstance(parsed, dict):
				signals[key] = parsed

		return signals

	@staticmethod
	def _parse_tool_content(content: Any) -> Dict[str, Any] | None:
		if isinstance(content, dict):
			return content
		if isinstance(content, list):
			for item in content:
				if isinstance(item, dict) and "text" in item:
					try:
						parsed = json.loads(item["text"])
						if isinstance(parsed, dict):
							return parsed
					except Exception:
						continue
		if isinstance(content, str):
			try:
				parsed = json.loads(content)
				if isinstance(parsed, dict):
					return parsed
			except Exception:
				return None
		return None

	@staticmethod
	def _build_a2a_artifact(verdict: EngineerVerdict, signals: Dict[str, object]) -> Dict[str, object]:
		"""Return a minimal A2A-compatible Artifact object with JSON data part."""
		return {
			"artifactId": str(uuid.uuid4()),
			"name": "engineer_verdict",
			"description": "Technical interview-slot assessment from engineer agent.",
			"parts": [
				{
					"data": {
						"score_out_of_10": verdict.score_out_of_10,
						"technical_criticisms": verdict.technical_criticisms,
						"genuine_strength": verdict.genuine_strength,
						"tool_signals": signals,
					},
					"mediaType": "application/json",
				}
			],
			"metadata": {"source": "engineer_agent"},
		}


def run_engineer_agent(resume_text: str, model_name: str | None = None) -> Dict[str, object]:
	"""Convenience function for orchestrator calls."""
	analyzer = EngineerResumeAnalyzer(model_name=model_name)
	return analyzer.analyze(resume_text)


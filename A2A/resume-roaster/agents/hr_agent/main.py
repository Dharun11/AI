"""HR agent main logic using Semantic Kernel.

Flow:
1) Run a Semantic Kernel ReAct loop over deterministic HR tools.
2) Produce final HR judgment as strict JSON.
3) Return A2A-style Artifact payload.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from .tools import (
	collect_signals,
	evaluate_ats_keywords,
	evaluate_bullet_language_strength,
	evaluate_date_consistency,
	evaluate_resume_length,
)


load_dotenv()


class HRVerdict(BaseModel):
	score_out_of_10: float = Field(ge=0, le=10)
	hr_criticisms: List[str] = Field(min_length=3, max_length=3)
	genuine_strength: str


class HRResumeAnalyzer:
	"""Single-responsibility service to evaluate first-pass recruiter survivability."""

	def __init__(self, model_name: str | None = None) -> None:
		provider = (os.getenv("HR_AGENT_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "azure").strip().lower()

		self._kernel = Kernel()
		self._service_id = "hr_chat"

		if provider == "openai":
			openai_api_key = os.getenv("OPENAI_API_KEY")
			openai_model = model_name or os.getenv("HR_AGENT_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
			openai_base_url = os.getenv("OPENAI_BASE_URL")

			if not openai_api_key:
				raise ValueError("OPENAI_API_KEY is required when HR_AGENT_LLM_PROVIDER=openai.")
			if not openai_model:
				raise ValueError("Set HR_AGENT_OPENAI_MODEL or OPENAI_MODEL for HR agent.")

			self._kernel.add_service(
				OpenAIChatCompletion(
					service_id=self._service_id,
					ai_model_id=openai_model,
					api_key=openai_api_key,
					base_url=openai_base_url,
				)
			)
		else:
			azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
			azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
			azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
			azure_deployment = model_name or os.getenv("HR_AGENT_AZURE_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

			if not azure_api_key:
				raise ValueError("AZURE_OPENAI_API_KEY is required for HR agent.")
			if not azure_endpoint:
				raise ValueError("AZURE_OPENAI_ENDPOINT is required for HR agent.")
			if not azure_deployment:
				raise ValueError(
					"Set HR_AGENT_AZURE_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT for HR agent."
				)

			self._kernel.add_service(
				AzureChatCompletion(
					service_id=self._service_id,
					deployment_name=azure_deployment,
					endpoint=azure_endpoint,
					api_key=azure_api_key,
					api_version=azure_api_version,
				)
			)

		self._react_prompt = (
			"You are a senior HR recruiter following a strict ReAct loop. "
			"You must first call tools, then provide a final verdict.\n\n"
			"Available tools and required usage:\n"
			"- evaluate_ats_keywords(resume_text)\n"
			"- evaluate_bullet_language_strength(resume_text)\n"
			"- evaluate_date_consistency(resume_text)\n"
			"- evaluate_resume_length(resume_text)\n"
			"Call each tool once before finishing.\n\n"
			"Return strict JSON only with this shape:\n"
			"{\n"
			"  \"thought\": \"short reasoning\",\n"
			"  \"action\": \"evaluate_ats_keywords|evaluate_bullet_language_strength|evaluate_date_consistency|evaluate_resume_length|finish\",\n"
			"  \"action_input\": {\"resume_text\": \"...\"},\n"
			"  \"final_verdict\": null OR {\"score_out_of_10\": number, \"hr_criticisms\": [3 strings], \"genuine_strength\": string}\n"
			"}\n\n"
			"If action is not finish, final_verdict must be null.\n"
			"If action is finish, final_verdict must be present and valid.\n"
			"No markdown, no extra keys.\n\n"
			"Resume text:\n{{$resume_text}}\n\n"
			"Tool history so far:\n{{$scratchpad}}\n"
		)

		self._react_function = self._kernel.add_function(
			plugin_name="hr_react",
			function_name="next_step",
			prompt=self._react_prompt,
		)

		self._tool_map: Dict[str, Callable[[str], Dict[str, object]]] = {
			"evaluate_ats_keywords": evaluate_ats_keywords,
			"evaluate_bullet_language_strength": evaluate_bullet_language_strength,
			"evaluate_date_consistency": evaluate_date_consistency,
			"evaluate_resume_length": evaluate_resume_length,
		}

	def analyze(self, resume_text: str) -> Dict[str, object]:
		verdict, signals = self._run_react_loop(resume_text)
		if not signals:
			signals = collect_signals(resume_text).to_dict()
		return self._build_a2a_artifact(verdict, signals)

	def _run_react_loop(self, resume_text: str, max_steps: int = 8) -> Tuple[HRVerdict, Dict[str, object]]:
		history = ChatHistory()
		signals: Dict[str, object] = {}
		called_tools: set[str] = set()
		scratchpad: List[str] = []

		for step in range(1, max_steps + 1):
			arguments = KernelArguments(
				resume_text=resume_text,
				scratchpad="\n".join(scratchpad) if scratchpad else "(none yet)",
			)
			result = self._kernel.invoke(self._react_function, arguments=arguments, chat_history=history)
			raw = str(result).strip()

			step_data = self._parse_react_step(raw)
			action = str(step_data.get("action", "")).strip()
			thought = str(step_data.get("thought", "")).strip()

			if action == "finish":
				final_verdict = step_data.get("final_verdict")
				if isinstance(final_verdict, dict):
					try:
						verdict = HRVerdict.model_validate(final_verdict)
					except ValidationError:
						verdict = self._fallback_verdict(signals or collect_signals(resume_text).to_dict())
				else:
					verdict = self._fallback_verdict(signals or collect_signals(resume_text).to_dict())
				return verdict, signals

			if action not in self._tool_map:
				scratchpad.append(
					f"Step {step}: invalid action '{action}'. Choose a valid tool or finish."
				)
				continue

			tool_output = self._tool_map[action](resume_text)
			called_tools.add(action)
			signal_key = self._tool_to_signal_key(action)
			signals[signal_key] = tool_output

			scratchpad.append(
				f"Step {step} thought: {thought or '(none)'}\n"
				f"Action: {action}\n"
				f"Observation: {json.dumps(tool_output, ensure_ascii=True)}"
			)

			if len(called_tools) == len(self._tool_map):
				scratchpad.append("All required tools have been called. You may now finish.")

		return self._fallback_verdict(signals or collect_signals(resume_text).to_dict()), signals

	@staticmethod
	def _tool_to_signal_key(tool_name: str) -> str:
		mapping = {
			"evaluate_ats_keywords": "ats_signal",
			"evaluate_bullet_language_strength": "language_signal",
			"evaluate_date_consistency": "date_signal",
			"evaluate_resume_length": "length_signal",
		}
		return mapping[tool_name]

	@staticmethod
	def _parse_react_step(raw_text: str) -> Dict[str, Any]:
		text = raw_text.strip()
		if text.startswith("```"):
			text = re.sub(r"^```[a-zA-Z]*\n", "", text)
			text = re.sub(r"\n```$", "", text)

		blob = HRResumeAnalyzer._extract_json_blob(text)
		if not blob:
			return {"thought": "", "action": "finish", "action_input": {}, "final_verdict": None}

		try:
			data = json.loads(blob)
			if isinstance(data, dict):
				return data
		except Exception:
			pass

		return {"thought": "", "action": "finish", "action_input": {}, "final_verdict": None}

	@staticmethod
	def _parse_verdict(raw_text: str, signals: Dict[str, object]) -> HRVerdict:
		text = raw_text.strip()
		if not text:
			return HRResumeAnalyzer._fallback_verdict(signals)

		if text.startswith("```"):
			text = re.sub(r"^```[a-zA-Z]*\n", "", text)
			text = re.sub(r"\n```$", "", text)

		json_blob = HRResumeAnalyzer._extract_json_blob(text)
		if not json_blob:
			return HRResumeAnalyzer._fallback_verdict(signals)

		data = json.loads(json_blob)
		try:
			return HRVerdict.model_validate(data)
		except ValidationError as error:
			return HRResumeAnalyzer._fallback_verdict(signals)

	@staticmethod
	def _extract_json_blob(text: str) -> str | None:
		# First try exact JSON object.
		try:
			json.loads(text)
			return text
		except Exception:
			pass

		# Fallback: extract the first JSON object-like block.
		start = text.find("{")
		end = text.rfind("}")
		if start >= 0 and end > start:
			candidate = text[start : end + 1]
			try:
				json.loads(candidate)
				return candidate
			except Exception:
				return None
		return None

	@staticmethod
	def _fallback_verdict(signals: Dict[str, object]) -> HRVerdict:
		ats = float(signals.get("ats_signal", {}).get("score", 0))
		language = float(signals.get("language_signal", {}).get("score", 0))
		date = float(signals.get("date_signal", {}).get("score", 0))
		length = float(signals.get("length_signal", {}).get("score", 0))

		score = round((ats + language + date + length) / 4.0, 1)
		return HRVerdict(
			score_out_of_10=score,
			hr_criticisms=[
				"ATS keyword coverage is not strong enough for safe first-pass filtering.",
				"Bullet language contains weak phrasing and lacks clear, decisive action statements.",
				"Timeline/length signals suggest the resume may create recruiter friction in a 6-second scan.",
			],
			genuine_strength="The resume contains enough structured content to recover with targeted HR-focused edits.",
		)

	@staticmethod
	def _build_a2a_artifact(verdict: HRVerdict, signals: Dict[str, object]) -> Dict[str, object]:
		return {
			"artifactId": str(uuid.uuid4()),
			"name": "hr_verdict",
			"description": "First-pass HR screening assessment.",
			"parts": [
				{
					"data": {
						"score_out_of_10": verdict.score_out_of_10,
						"hr_criticisms": verdict.hr_criticisms,
						"genuine_strength": verdict.genuine_strength,
						"tool_signals": signals,
					},
					"mediaType": "application/json",
				}
			],
			"metadata": {"source": "hr_agent"},
		}


def run_hr_agent(resume_text: str, model_name: str | None = None) -> Dict[str, object]:
	analyzer = HRResumeAnalyzer(model_name=model_name)
	return analyzer.analyze(resume_text)


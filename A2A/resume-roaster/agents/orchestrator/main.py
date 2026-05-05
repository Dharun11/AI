"""Orchestrator that calls HR and Engineer agents via A2A in parallel."""

from __future__ import annotations

import asyncio
import argparse
import json
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

import httpx

from .tools import build_final_roast


class A2AClientError(Exception):
	"""Raised when an A2A call fails or returns unexpected shape."""


class ResumeRoastOrchestrator:
	"""Coordinates parallel A2A calls and merges their artifacts."""

	def __init__(
		self,
		engineer_base_url: str = "http://127.0.0.1:8102",
		hr_base_url: str = "http://127.0.0.1:8103",
		timeout_seconds: float = 60.0,
	) -> None:
		self.engineer_base_url = engineer_base_url.rstrip("/")
		self.hr_base_url = hr_base_url.rstrip("/")
		self.timeout_seconds = timeout_seconds

	async def _send_message(self, client: httpx.AsyncClient, base_url: str, resume_text: str) -> Dict[str, Any]:
		payload = {
			"message": {
				"messageId": str(uuid4()),
				"role": "ROLE_USER",
				"parts": [{"text": resume_text, "mediaType": "text/plain"}],
			}
		}
		response = await client.post(
			f"{base_url}/message:send",
			json=payload,
			headers={"A2A-Version": "1.0", "Content-Type": "application/json"},
		)
		if response.status_code >= 400:
			raise A2AClientError(f"A2A send failed ({response.status_code}) for {base_url}: {response.text}")
		return response.json()

	async def _fetch_task(self, client: httpx.AsyncClient, base_url: str, task_id: str) -> Dict[str, Any]:
		response = await client.get(f"{base_url}/tasks/{task_id}")
		if response.status_code >= 400:
			raise A2AClientError(f"A2A get task failed ({response.status_code}) for {base_url}: {response.text}")
		return response.json()

	async def _artifact_from_response(
		self,
		client: httpx.AsyncClient,
		base_url: str,
		send_response: Dict[str, Any],
	) -> Dict[str, Any]:
		task = send_response.get("task")
		if not isinstance(task, dict):
			raise A2AClientError(f"Expected task in send response from {base_url}.")

		artifacts = task.get("artifacts")
		if isinstance(artifacts, list) and artifacts:
			return artifacts[0]

		task_id = task.get("id")
		if not task_id:
			raise A2AClientError(f"Task id missing in send response from {base_url}.")

		task_data = await self._fetch_task(client, base_url, task_id)
		fetched_artifacts = task_data.get("artifacts")
		if not isinstance(fetched_artifacts, list) or not fetched_artifacts:
			raise A2AClientError(f"No artifacts found for task {task_id} from {base_url}.")
		return fetched_artifacts[0]

	async def roast_resume_async(self, resume_text: str) -> Dict[str, Any]:
		async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
			engineer_send, hr_send = await asyncio.gather(
				self._send_message(client, self.engineer_base_url, resume_text),
				self._send_message(client, self.hr_base_url, resume_text),
			)

			engineer_artifact, hr_artifact = await asyncio.gather(
				self._artifact_from_response(client, self.engineer_base_url, engineer_send),
				self._artifact_from_response(client, self.hr_base_url, hr_send),
			)

		final_roast = build_final_roast(engineer_artifact, hr_artifact)
		return {
			"engineer_artifact": engineer_artifact,
			"hr_artifact": hr_artifact,
			"final_roast": final_roast,
		}

	def roast_resume(self, resume_text: str) -> Dict[str, Any]:
		return asyncio.run(self.roast_resume_async(resume_text))


def run_orchestrator(
	resume_text: str,
	engineer_base_url: str = "http://127.0.0.1:8102",
	hr_base_url: str = "http://127.0.0.1:8103",
) -> Dict[str, Any]:
	orchestrator = ResumeRoastOrchestrator(
		engineer_base_url=engineer_base_url,
		hr_base_url=hr_base_url,
	)
	return orchestrator.roast_resume(resume_text)


def _read_resume_text(args: argparse.Namespace) -> str:
	if args.resume_file:
		with open(args.resume_file, "r", encoding="utf-8") as file:
			return file.read()

	if not sys.stdin.isatty():
		return sys.stdin.read()

	return args.resume_text or ""


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run orchestrator merge for HR + Engineer agents.")
	parser.add_argument("--resume-file", type=str, help="Path to resume text file.")
	parser.add_argument("--resume-text", type=str, help="Raw resume text.")
	parser.add_argument("--engineer-url", type=str, default="http://127.0.0.1:8102", help="Engineer A2A base URL.")
	parser.add_argument("--hr-url", type=str, default="http://127.0.0.1:8103", help="HR A2A base URL.")
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	resume_text = _read_resume_text(args).strip()
	if not resume_text:
		print("No resume text provided. Use --resume-file, --resume-text, or pipe stdin.", file=sys.stderr)
		return 1

	result = run_orchestrator(
		resume_text=resume_text,
		engineer_base_url=args.engineer_url,
		hr_base_url=args.hr_url,
	)
	print(json.dumps(result, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


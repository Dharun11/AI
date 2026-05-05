"""CLI-style executor for the engineer agent.

Usage examples:
  python -m agents.engineer_agent.executor --resume-file path/to/resume.txt
  echo "resume text" | python -m agents.engineer_agent.executor
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from .main import EngineerResumeAnalyzer
from .a2a_server import create_app


def _read_resume_text(args: argparse.Namespace) -> str:
	if args.resume_file:
		with open(args.resume_file, "r", encoding="utf-8") as file:
			return file.read()

	if not sys.stdin.isatty():
		return sys.stdin.read()

	return args.resume_text or ""


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run engineer resume analysis.")
	parser.add_argument("--resume-file", type=str, help="Path to a raw resume text file.")
	parser.add_argument("--resume-text", type=str, help="Raw resume text as an argument.")
	parser.add_argument("--model", type=str, default=None, help="Override model name.")
	parser.add_argument("--serve", action="store_true", help="Run A2A HTTP server mode.")
	parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
	parser.add_argument("--port", type=int, default=8102, help="Server port.")
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	if args.serve:
		base_url = f"http://{args.host}:{args.port}"
		app = create_app(base_url=base_url)
		uvicorn = importlib.import_module("uvicorn")
		uvicorn.run(app, host=args.host, port=args.port)
		return 0

	resume_text = _read_resume_text(args).strip()
	if not resume_text:
		print("No resume text provided.", file=sys.stderr)
		return 1

	analyzer = EngineerResumeAnalyzer(model_name=args.model)
	artifact = analyzer.analyze(resume_text)
	print(json.dumps(artifact, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


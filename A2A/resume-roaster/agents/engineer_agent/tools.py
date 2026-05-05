"""Technical signal tools for the engineer agent.

These tools are intentionally simple and deterministic. They provide a structured
signal that is later synthesized by one LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from langchain.tools import tool


CURRENT_TECH = {
	"python",
	"fastapi",
	"django",
	"flask",
	"typescript",
	"node",
	"react",
	"next.js",
	"nextjs",
	"kubernetes",
	"docker",
	"aws",
	"gcp",
	"azure",
	"postgres",
	"redis",
	"graphql",
	"langchain",
}

OUTDATED_TECH = {
	"jquery",
	"backbone",
	"angularjs",
	"php 5",
	"bootstrap 3",
}

IMPACT_PATTERNS = [
	r"\b\d+\s*%\b",
	r"\b\d+[kKmM]?\s*(users?|requests?|rps|qps|downloads?)\b",
	r"\b(reduced|improved|increased|decreased|cut)\b",
	r"\b(latency|throughput|availability|uptime|cost|mttr|error rate)\b",
]

DEPTH_PATTERNS = [
	r"\b(scale|scaled|distributed|sharding|partition|multi-tenant)\b",
	r"\b(kafka|queue|pubsub|event[- ]driven|stream)\b",
	r"\b(ci/cd|observability|tracing|monitoring|slo|sla)\b",
	r"\b(load test|benchmark|profil|cache)\b",
	r"\b\d+[kKmM]?\s*(users?|requests?|rps|qps)\b",
]


def _norm(text: str) -> str:
	return text.lower()


def _find_terms(text: str, terms: set[str]) -> List[str]:
	hits: List[str] = []
	low = _norm(text)
	for term in sorted(terms):
		if term in low:
			hits.append(term)
	return hits


@dataclass
class ToolSignals:
	stack_signal: Dict[str, object]
	impact_signal: Dict[str, object]
	depth_signal: Dict[str, object]
	public_work_signal: Dict[str, object]

	def to_dict(self) -> Dict[str, object]:
		return {
			"stack_signal": self.stack_signal,
			"impact_signal": self.impact_signal,
			"depth_signal": self.depth_signal,
			"public_work_signal": self.public_work_signal,
		}


@tool
def evaluate_stack_currency(resume_text: str) -> Dict[str, object]:
	"""Check whether the tech stack looks current versus outdated."""
	modern = _find_terms(resume_text, CURRENT_TECH)
	outdated = _find_terms(resume_text, OUTDATED_TECH)
	score = max(0, min(10, len(modern) - len(outdated) + 4))
	return {
		"score": score,
		"modern_hits": modern,
		"outdated_hits": outdated,
		"summary": "Current stack signal derived from known modern/outdated terms.",
	}


@tool
def evaluate_impact_claims(resume_text: str) -> Dict[str, object]:
	"""Check whether project bullets include measurable outcomes."""
	low = _norm(resume_text)
	match_counts = {pattern: len(re.findall(pattern, low)) for pattern in IMPACT_PATTERNS}
	total_matches = sum(match_counts.values())
	score = max(0, min(10, total_matches + 2))
	return {
		"score": score,
		"total_metric_signals": total_matches,
		"pattern_hits": match_counts,
		"summary": "Impact signal based on numbers and outcome-oriented language.",
	}


@tool
def evaluate_project_depth(resume_text: str) -> Dict[str, object]:
	"""Check whether projects mention architecture, scale, and engineering depth."""
	low = _norm(resume_text)
	match_counts = {pattern: len(re.findall(pattern, low)) for pattern in DEPTH_PATTERNS}
	total_matches = sum(match_counts.values())
	score = max(0, min(10, total_matches + 1))
	return {
		"score": score,
		"total_depth_signals": total_matches,
		"pattern_hits": match_counts,
		"summary": "Depth signal based on scale, systems, and operational engineering hints.",
	}


@tool
def evaluate_public_work(resume_text: str) -> Dict[str, object]:
	"""Check for GitHub, open source, blogs, talks, or public technical footprint."""
	low = _norm(resume_text)
	github_links = re.findall(r"https?://(?:www\.)?github\.com/[\w\-./]+", low)
	open_source_words = len(re.findall(r"\b(open source|oss|maintainer|contributor)\b", low))
	talks_or_writing = len(re.findall(r"\b(blog|talk|speaker|conference|article)\b", low))
	footprint = len(github_links) + open_source_words + talks_or_writing
	score = max(0, min(10, footprint + 1))
	return {
		"score": score,
		"github_links": github_links,
		"open_source_mentions": open_source_words,
		"talks_or_writing_mentions": talks_or_writing,
		"summary": "Public proof-of-work signal from links and external contributions.",
	}


def collect_signals(resume_text: str) -> ToolSignals:
	"""Run all technical tools first and return one structured signal object."""
	return ToolSignals(
		stack_signal=evaluate_stack_currency.invoke({"resume_text": resume_text}),
		impact_signal=evaluate_impact_claims.invoke({"resume_text": resume_text}),
		depth_signal=evaluate_project_depth.invoke({"resume_text": resume_text}),
		public_work_signal=evaluate_public_work.invoke({"resume_text": resume_text}),
	)


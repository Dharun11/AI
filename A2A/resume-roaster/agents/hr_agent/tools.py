"""Deterministic HR screening tools.

These tools produce structured signals for first-pass resume filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


ATS_KEYWORDS = {
	"experience",
	"skills",
	"education",
	"leadership",
	"communication",
	"collaboration",
	"impact",
	"achieved",
	"delivered",
	"managed",
	"projects",
	"certification",
}

VAGUE_FILLER_PATTERNS = [
	r"\bresponsible for\b",
	r"\bworked on\b",
	r"\bhelped with\b",
	r"\bassisted with\b",
	r"\binvolved in\b",
]

STRONG_ACTION_VERBS = {
	"led",
	"built",
	"delivered",
	"owned",
	"drove",
	"improved",
	"increased",
	"reduced",
	"launched",
	"optimized",
	"designed",
	"implemented",
}

DATE_RANGE_PATTERN = re.compile(
	r"(?P<start>(?:\d{1,2}/)?\d{4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\s*[-–]\s*(?P<end>(?:\d{1,2}/)?\d{4}|present|current|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})",
	re.IGNORECASE,
)

CLAIMED_EXPERIENCE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+experience", re.IGNORECASE)


def _lower(text: str) -> str:
	return text.lower()


def _parse_date(token: str) -> datetime:
	token = token.strip().lower()
	if token in {"present", "current"}:
		return datetime.utcnow()

	for fmt in ("%m/%Y", "%b %Y", "%Y"):
		try:
			return datetime.strptime(token, fmt)
		except ValueError:
			continue

	return datetime.utcnow()


def _months_between(start: datetime, end: datetime) -> int:
	return max(0, (end.year - start.year) * 12 + (end.month - start.month))


def evaluate_ats_keywords(resume_text: str) -> Dict[str, object]:
	"""Check ATS keyword presence likely used in first-pass filtering."""
	low = _lower(resume_text)
	hits = sorted([k for k in ATS_KEYWORDS if k in low])
	score = max(0, min(10, int((len(hits) / max(1, len(ATS_KEYWORDS))) * 10)))
	return {
		"score": score,
		"keyword_hits": hits,
		"missing_count": len(ATS_KEYWORDS) - len(hits),
		"summary": "ATS signal from standard recruiter-screen keywords.",
	}


def evaluate_bullet_language_strength(resume_text: str) -> Dict[str, object]:
	"""Measure vague filler usage against strong action language."""
	low = _lower(resume_text)
	filler_hits = {pattern: len(re.findall(pattern, low)) for pattern in VAGUE_FILLER_PATTERNS}
	filler_total = sum(filler_hits.values())

	verb_hits = sorted([verb for verb in STRONG_ACTION_VERBS if re.search(rf"\b{re.escape(verb)}\b", low)])
	strong_total = len(verb_hits)

	raw = 5 + strong_total - filler_total
	score = max(0, min(10, raw))
	return {
		"score": score,
		"filler_phrase_hits": filler_hits,
		"strong_action_verbs_found": verb_hits,
		"summary": "Language signal from weak filler versus clear action verbs.",
	}


def evaluate_date_consistency(resume_text: str) -> Dict[str, object]:
	"""Estimate total timeline and compare with stated years of experience claims."""
	ranges: List[Tuple[str, str]] = []
	months_total = 0
	parse_failures = 0

	for match in DATE_RANGE_PATTERN.finditer(resume_text):
		start_raw = match.group("start")
		end_raw = match.group("end")
		ranges.append((start_raw, end_raw))
		try:
			start_dt = _parse_date(start_raw)
			end_dt = _parse_date(end_raw)
			if end_dt < start_dt:
				start_dt, end_dt = end_dt, start_dt
			months_total += _months_between(start_dt, end_dt)
		except Exception:
			parse_failures += 1

	claimed_years = [float(value) for value in CLAIMED_EXPERIENCE_PATTERN.findall(resume_text)]
	estimated_years = round(months_total / 12.0, 1)
	max_claim = max(claimed_years) if claimed_years else None

	inflated_claim = False
	if max_claim is not None:
		inflated_claim = max_claim > (estimated_years + 1.0)

	score = 8
	if parse_failures > 0:
		score -= 1
	if len(ranges) == 0:
		score -= 3
	if inflated_claim:
		score -= 3
	score = max(0, min(10, score))

	return {
		"score": score,
		"detected_ranges": ranges,
		"estimated_total_years": estimated_years,
		"claimed_years": claimed_years,
		"possible_inflation": inflated_claim,
		"summary": "Date consistency signal from timeline extraction and claim checks.",
	}


def evaluate_resume_length(resume_text: str) -> Dict[str, object]:
	"""Estimate whether resume is in the 1-2 page readability range."""
	words = re.findall(r"\S+", resume_text)
	word_count = len(words)

	# Rough page estimate for plain text resumes.
	estimated_pages = round(word_count / 500.0, 2)

	if word_count < 250:
		score = 5
		band = "too_short"
	elif word_count <= 1100:
		score = 10
		band = "ideal"
	elif word_count <= 1700:
		score = 6
		band = "long"
	else:
		score = 2
		band = "too_long"

	return {
		"score": score,
		"word_count": word_count,
		"estimated_pages": estimated_pages,
		"length_band": band,
		"summary": "Length signal based on concise 1-2 page expectation.",
	}


@dataclass
class HRSignals:
	ats_signal: Dict[str, object]
	language_signal: Dict[str, object]
	date_signal: Dict[str, object]
	length_signal: Dict[str, object]

	def to_dict(self) -> Dict[str, object]:
		return {
			"ats_signal": self.ats_signal,
			"language_signal": self.language_signal,
			"date_signal": self.date_signal,
			"length_signal": self.length_signal,
		}


def collect_signals(resume_text: str) -> HRSignals:
	"""Run all HR tools first and collect structured screening signals."""
	return HRSignals(
		ats_signal=evaluate_ats_keywords(resume_text),
		language_signal=evaluate_bullet_language_strength(resume_text),
		date_signal=evaluate_date_consistency(resume_text),
		length_signal=evaluate_resume_length(resume_text),
	)


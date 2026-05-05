"""Helpers for merging A2A artifacts from HR and Engineer agents."""

from __future__ import annotations

from typing import Any, Dict, List


def _extract_artifact_data(artifact: Dict[str, Any]) -> Dict[str, Any]:
	parts = artifact.get("parts", [])
	if not parts:
		return {}

	first_part = parts[0]
	data = first_part.get("data")
	if isinstance(data, dict):
		return data
	return {}


def build_final_roast(engineer_artifact: Dict[str, Any], hr_artifact: Dict[str, Any]) -> Dict[str, Any]:
	"""Merge both artifacts into one final orchestrator payload."""
	engineer_data = _extract_artifact_data(engineer_artifact)
	hr_data = _extract_artifact_data(hr_artifact)

	engineer_score = float(engineer_data.get("score_out_of_10", 0))
	hr_score = float(hr_data.get("score_out_of_10", 0))
	blended_score = round((engineer_score + hr_score) / 2.0, 1)

	engineer_criticisms: List[str] = list(engineer_data.get("technical_criticisms", []))
	hr_criticisms: List[str] = list(hr_data.get("hr_criticisms", []))

	return {
		"overall_score_out_of_10": blended_score,
		"engineer_score_out_of_10": engineer_score,
		"hr_score_out_of_10": hr_score,
		"top_criticisms": {
			"engineer": engineer_criticisms,
			"hr": hr_criticisms,
		},
		"strengths": {
			"engineer": engineer_data.get("genuine_strength", ""),
			"hr": hr_data.get("genuine_strength", ""),
		},
		"verdict": "interview" if blended_score >= 6.5 else "reject",
	}


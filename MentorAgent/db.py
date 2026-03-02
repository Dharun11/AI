import json
from pathlib import Path
from logger import setup_logger

logger = setup_logger("DB")

DATA_DIR = Path(__file__).parent / "data"


def load_json(filename: str) -> dict | list:
    logger.info(f"Loading data from {filename}")
    filepath = DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filename: str, data: dict | list) -> None:
    logger.info(f"Saving data to {filename}")
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_student(student_id: str) -> dict | None:
    logger.info(f"Fetching student {student_id}")
    students = load_json("students.json")
    for s in students:
        if s["student_id"].upper() == student_id.upper():
            return s
    logger.warning(f"Student {student_id} not found")
    return None


def get_all_students() -> list[dict]:
    return load_json("students.json")


def get_mentors() -> list[dict]:
    return load_json("mentors.json")


def get_placements() -> list[dict]:
    return load_json("placements.json")


def get_conversation(conversation_id: str) -> list[dict]:
    logger.info(f"Retrieving conversation: {conversation_id}")
    conversations = load_json("conversations.json")
    return conversations.get(conversation_id, [])


def save_conversation(conversation_id: str, messages: list[dict]) -> None:
    logger.info(f"Updating conversation: {conversation_id}")
    conversations = load_json("conversations.json")
    conversations[conversation_id] = messages
    save_json("conversations.json", conversations)


def update_student(student_id: str, updates: dict) -> dict | None:
    students = load_json("students.json")
    for i, s in enumerate(students):
        if s["student_id"].upper() == student_id.upper():
            students[i].update(updates)
            save_json("students.json", students)
            return students[i]
    return None

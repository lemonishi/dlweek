"""End-to-end processing of an uploaded document.

Usage:
    python process_upload.py path/to/file.pdf --student student_123 --tag lecture
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# ensure environment variables from .env are loaded
load_dotenv()

from document_analyzer import extract_text_from_doc, classify_document
from skill_extractor import SkillExtractor
from schema_validator import SkillList
from skill_updater import upsert_objective, fetch_student_objectives, upsert_student_profile, merge_skill_lists

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_file(file_path: Path, student_id: str, user_tag: str = None):
    logging.info("Processing file %s", file_path)
    text = extract_text_from_doc(file_path)
    doc_type = classify_document(text, user_tag)
    extractor = SkillExtractor()
    skill_data = extractor.extract(text, doc_type, student_id)
    logging.debug("Raw extracted data: %s", skill_data)

    # validate
    skill_list = SkillList(**skill_data)
    logging.info("Extracted %d learnt, %d to-learn skills", len(skill_list.skills_learnt), len(skill_list.skills_to_learn))

    # upsert objectives individually
    for sk in skill_list.skills_learnt + skill_list.skills_to_learn:
        upsert_objective(sk.model_dump())

    # merge into student profile
    existing = fetch_student_objectives(student_id)
    merged = merge_skill_lists(existing, skill_list.model_dump())
    merged["studentId"] = student_id

    # Update only student_profiles container
    upsert_student_profile(merged)
    logging.info("Student profile updated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an uploaded document into skills")
    parser.add_argument("file", type=Path, help="Path to document file")
    parser.add_argument("--student", required=True, help="Student ID")
    parser.add_argument("--tag", help="Optional user tag (lecture/homework)")
    args = parser.parse_args()
    process_file(args.file, args.student, args.tag)

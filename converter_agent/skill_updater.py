"""Functions for merging extracted skills into Cosmos DB containers."""
import os
import logging
from typing import Dict, Any
from cosmos_conn import container

STUDENTS_CONTAINER = os.getenv("COSMOS_STUDENTS_CONTAINER", "student")


def upsert_objective(skill_obj: Dict[str, Any]):
    """Insert or update an objective document in the `objective` container.

    Before saving, attempt to enrich the record using the canonical entry in
    the `skill_library` container.  This fills in any missing description,
    tags, vector values, difficulty, etc., so downstream consumers always have
    the authoritative metadata.
    """
    enriched = _enrich_from_vocab(skill_obj)
    c = container("objective")
    return c.upsert_item(enriched)


def _enrich_from_vocab(skill_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of **skill_obj** with blanks filled from the skill_library.

    The vocabulary items store `vector` as a list; convert it to the expected
    object form if the incoming object has zeros or is missing.
    """
    c = container("skill_library")
    query = "SELECT * FROM c WHERE c.skillId = @id"
    items = list(c.query_items(
        query=query,
        parameters=[{"name": "@id", "value": skill_obj.get("id")}],
        enable_cross_partition_query=True,
    ))
    if not items:
        return skill_obj
    vocab = items[0]
    # fields we want to copy if absent/empty
    for key in ("description", "tags", "difficulty", "prerequisites", "version"):
        if not skill_obj.get(key) and vocab.get(key) is not None:
            skill_obj[key] = vocab.get(key)
    # handle vector separately
    vocab_vec = vocab.get("vector")
    if isinstance(vocab_vec, list) and len(vocab_vec) == 3:
        # convert to object if necessary
        if (not skill_obj.get("vector") or
            all(v == 0 for v in skill_obj.get("vector", {}).values())):
            skill_obj["vector"] = {
                "mastery": vocab_vec[0],
                "edge_case": vocab_vec[1],
                "implementation": vocab_vec[2],
            }
    return skill_obj


def fetch_student_objectives(student_id: str) -> Dict[str, Any]:
    """Return a summary document for the student if it exists, else None."""
    c = container(STUDENTS_CONTAINER)
    query = "SELECT * FROM c WHERE c.studentId = @sid"
    items = list(c.query_items(
        query=query,
        parameters=[{"name": "@sid", "value": student_id}],
        enable_cross_partition_query=True,
    ))
    return items[0] if items else None


def upsert_student_profile(profile: Dict[str, Any]):
    """Save the entire student profile document (learned/to-learn lists).

    Cosmos requires each document to have a top-level `id` field.  If the
    caller didn't set one we derive it from `studentId`.  We also catch and
    log any errors so calling code can decide how to proceed.
    """
    if "id" not in profile or not profile.get("id"):
        sid = profile.get("studentId") or ""
        # use studentId as id or fall back to uuid
        profile["id"] = sid or str(__import__("uuid").uuid4())
    c = container(STUDENTS_CONTAINER)
    try:
        return c.upsert_item(profile)
    except Exception as e:
        logging.error("Failed to upsert student profile: %s", e)
        raise


def merge_skill_lists(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge learned/to-learn arrays; preserve highest mastery and avoid duplicates."""
    if existing is None:
        return new_data
    result = {"skills_learnt": [], "skills_to_learn": []}
    # index existing by id
    for arr in ("skills_learnt", "skills_to_learn"):
        idx = {s["id"]: s for s in existing.get(arr, [])}
        for s in new_data.get(arr, []):
            if s["id"] in idx:
                # take higher mastery
                if s["vector"]["mastery"] > idx[s["id"]]["vector"]["mastery"]:
                    idx[s["id"]] = s
            else:
                idx[s["id"]] = s
        result[arr] = list(idx.values())
    return result

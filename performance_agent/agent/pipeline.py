from __future__ import annotations

from typing import Any, Dict, List, Literal

from agent.azure_db import CosmosRepo, load_cosmos_config_from_env
from agent.generators import generate_learning_resources
from agent.weakness import compute_weak_skills


def get_weak_skill_names_for_student(
    student_id: str,
    *,
    top_k: int = 5,
    model_path: str = "dkvmn_model.pt",
    skill_map_path: str = "skill2idx.json",
) -> List[str]:
    cfg = load_cosmos_config_from_env()
    repo = CosmosRepo(cfg)

    objective_docs = repo.get_objective_skills(student_id=student_id)
    # Student state can be either: many docs (one per skill) OR one profile doc with arrays
    student_profile = repo.get_student_profile_doc(student_id=student_id)
    student_skill_docs = repo.get_student_skill_states(student_id=student_id)

    # If profile doc has "skills": use that as the student_skill_docs (more accurate)
    if student_profile and isinstance(student_profile.get("skills"), list):
        student_skill_docs = student_profile["skills"]

    weak_names = compute_weak_skills(
        objective_skill_docs=objective_docs,
        student_skill_docs=student_skill_docs,
        student_profile_doc=student_profile,
        model_path=model_path,
        skill_map_path=skill_map_path,
        top_k=top_k,
        use_names=True,
    )
    return weak_names


def run_final_learning_pipeline(
    student_id: str,
    *,
    resource_type: Literal["video", "flashcards", "quiz"],
    top_k: int = 5,
    model_path: str = "dkvmn_model.pt",
    skill_map_path: str = "skill2idx.json",
    generator_model: str = "gpt-5.2",
    quiz_questions: int = 10,
    quiz_difficulty: int = 5,
    flashcard_count: int = 5,
) -> Dict[str, Any]:
    """
    Final workflow:
      1) Read objective + student containers from Cosmos DB
      2) Compute DKVMN-weighted deltas and return weak skill names
      3) Generate selected remediation resources for those weak skills
    """
    weak_skill_names = get_weak_skill_names_for_student(
        student_id,
        top_k=top_k,
        model_path=model_path,
        skill_map_path=skill_map_path,
    )

    resources = generate_learning_resources(
        weak_skill_names,
        resource_type=resource_type,
        quiz_questions=quiz_questions,
        quiz_difficulty=quiz_difficulty,
        flashcard_count=flashcard_count,
        model=generator_model,
    )

    return {
        "studentId": student_id,
        "weakSkills": weak_skill_names,
        "resourceType": resource_type,
        "resources": resources,
    }


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--student_id", required=True)
    p.add_argument("--resource_type", choices=["video", "flashcards", "quiz"], default="video")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--model_path", default="dkvmn_model.pt")
    p.add_argument("--skill_map_path", default="skill2idx.json")
    p.add_argument("--generator_model", default="gpt-5.2")
    p.add_argument("--quiz_questions", type=int, default=10)
    p.add_argument("--quiz_difficulty", type=int, default=5)
    p.add_argument("--flashcard_count", type=int, default=5)
    args = p.parse_args()

    out = run_final_learning_pipeline(
        args.student_id,
        resource_type=args.resource_type,
        top_k=args.top_k,
        model_path=args.model_path,
        skill_map_path=args.skill_map_path,
        generator_model=args.generator_model,
        quiz_questions=args.quiz_questions,
        quiz_difficulty=args.quiz_difficulty,
        flashcard_count=args.flashcard_count,
    )
    print(json.dumps(out, indent=2))

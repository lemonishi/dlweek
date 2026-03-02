from __future__ import annotations

from typing import List, Optional

from agent.azure_db import CosmosRepo, load_cosmos_config_from_env
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


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--student_id", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--model_path", default="dkvmn_model.pt")
    p.add_argument("--skill_map_path", default="skill2idx.json")
    args = p.parse_args()

    out = get_weak_skill_names_for_student(
        args.student_id,
        top_k=args.top_k,
        model_path=args.model_path,
        skill_map_path=args.skill_map_path,
    )
    print(json.dumps(out, indent=2))
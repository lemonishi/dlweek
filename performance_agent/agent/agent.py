from typing import Any, Dict
from agent.parsing import extract_skills, extract_dkvmn_scores, join_skill_metadata_with_scores
from agent.scoring import performance_summary, rank_weak_skills
from agent.generators import generate_recommendations_and_quiz

def agent(
    student_json: Dict[str, Any],
    *,
    weak_threshold: float = 0.6,
    top_k: int = 3,
    make_quiz: bool = False,
    num_questions_per_skill: int = 3,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:

    skills = extract_skills(student_json)
    dkvmn_scores = extract_dkvmn_scores(student_json)
    scored = join_skill_metadata_with_scores(skills, dkvmn_scores)

    summary = performance_summary(scored)
    weak = rank_weak_skills(scored, threshold=weak_threshold, top_k=top_k)

    if not weak:
        return {
            "summary": summary,
            "weak_skills": [],
            "message": "No weak skills under threshold.",
            "recommendations": [],
            "quiz": []
        }

    context = {
        "student_id": student_json.get("student_id") or student_json.get("id"),
        "course": student_json.get("course") or student_json.get("module"),
        "notes": student_json.get("notes"),
    }

    llm_out = generate_recommendations_and_quiz(
        weak_skills=weak,
        student_context=context,
        make_quiz=make_quiz,
        num_questions_per_skill=num_questions_per_skill,
        model=model,
    )

    return {
        "summary": summary,
        "weak_skills": [
            {
                "skillId": s.get("skillId") or s.get("id"),
                "name": s.get("name"),
                "dkvmn_score": score
            }
            for s, score in weak
        ],
        **llm_out
    }
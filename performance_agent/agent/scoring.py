from typing import Any, Dict, List, Tuple

def performance_summary(scored_skills: List[Tuple[dict, float]]) -> Dict[str, Any]:
    if not scored_skills:
        return {"num_skills": 0}
    scores = [s for _, s in scored_skills]
    avg = sum(scores) / len(scores)
    return {
        "num_skills": len(scores),
        "avg_mastery": round(avg, 3),
        "min_mastery": round(min(scores), 3),
        "max_mastery": round(max(scores), 3),
    }

def rank_weak_skills(scored_skills: List[Tuple[dict, float]], threshold: float = 0.6, top_k: int = 3):
    weak = [(skill, score) for skill, score in scored_skills if score < threshold]
    weak.sort(key=lambda x: x[1])
    return weak[:top_k]
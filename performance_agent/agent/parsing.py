from typing import Any, Dict, List, Tuple

def extract_skills(student_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expects a list of skill nodes under 'skills' (preferred),
    but stays flexible.
    """
    skills = student_json.get("skills") or student_json.get("skill_nodes") or student_json.get("topics")
    return skills if isinstance(skills, list) else []

def extract_dkvmn_scores(student_json: Dict[str, Any]) -> Dict[str, float]:
    """
    Preferred: student_json["dkvmn"] = {skillId: score}
    Also supports a list form: [{"skillId": "...", "score": 0.4}, ...]
    """
    scores = {}

    dk = student_json.get("dkvmn")
    if isinstance(dk, dict):
        for k, v in dk.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                scores[k] = float(v)

    if isinstance(dk, list):
        for row in dk:
            if isinstance(row, dict):
                sid = row.get("skillId") or row.get("id")
                v = row.get("score") or row.get("mastery") or row.get("p_correct")
                if isinstance(sid, str) and isinstance(v, (int, float)):
                    scores[sid] = float(v)

    # clamp
    for sid in list(scores.keys()):
        scores[sid] = max(0.0, min(1.0, scores[sid]))

    return scores

def join_skill_metadata_with_scores(
    skills: List[Dict[str, Any]],
    dkvmn_scores: Dict[str, float]
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Returns [(skill_node, score)] for skills that have scores.
    Uses skillId as the key (fallback to id).
    """
    out = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        sid = s.get("skillId") or s.get("id")
        if isinstance(sid, str) and sid in dkvmn_scores:
            out.append((s, dkvmn_scores[sid]))
    return out
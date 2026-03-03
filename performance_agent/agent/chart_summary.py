from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple


def _extract_skill_id(doc: Dict[str, Any]) -> Optional[str]:
    sid = doc.get("skillId") or doc.get("id")
    return sid.strip() if isinstance(sid, str) and sid.strip() else None


def _extract_skill_name(doc: Dict[str, Any], default: str) -> str:
    name = doc.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return default


def _extract_mastery(doc: Dict[str, Any], default: float = 0.0) -> float:
    vector = doc.get("vector")
    if isinstance(vector, dict):
        val = vector.get("mastery")
        if isinstance(val, (int, float)):
            return max(0.0, min(1.0, float(val)))
    if isinstance(vector, list) and len(vector) >= 1 and isinstance(vector[0], (int, float)):
        return max(0.0, min(1.0, float(vector[0])))
    val = doc.get("mastery")
    if isinstance(val, (int, float)):
        return max(0.0, min(1.0, float(val)))
    return default


def _iter_array_skills(doc: Dict[str, Any], keys: Tuple[str, ...]) -> Iterable[Dict[str, Any]]:
    for key in keys:
        value = doc.get(key)
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    yield row


def _aggregate_skills(
    docs: List[Dict[str, Any]],
    *,
    array_keys: Tuple[str, ...],
    default_mastery: float,
) -> Dict[str, Dict[str, Any]]:
    skill_map: Dict[str, Dict[str, Any]] = {}

    def upsert(row: Dict[str, Any]) -> None:
        sid = _extract_skill_id(row)
        if not sid:
            return
        mastery = _extract_mastery(row, default=default_mastery)
        name = _extract_skill_name(row, sid)
        prev = skill_map.get(sid)
        if prev is None or mastery > prev["mastery"]:
            skill_map[sid] = {"id": sid, "name": name, "mastery": mastery}

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        upsert(doc)
        for row in _iter_array_skills(doc, array_keys):
            upsert(row)

    return skill_map


def build_student_skill_bar_chart_payload(
    *,
    student_id: str,
    objective_docs: List[Dict[str, Any]],
    student_docs: List[Dict[str, Any]],
    student_profile_doc: Optional[Dict[str, Any]] = None,
    max_bars: int = 5,
) -> Dict[str, Any]:
    max_bars = max(1, min(5, int(max_bars)))

    student_all_docs = list(student_docs)
    if isinstance(student_profile_doc, dict):
        student_all_docs.append(student_profile_doc)

    objective_skills = _aggregate_skills(
        objective_docs,
        array_keys=("skills_to_learn", "skills", "objectives"),
        default_mastery=1.0,
    )
    student_skills = _aggregate_skills(
        student_all_docs,
        array_keys=("skills_learnt", "skills", "learned_skills"),
        default_mastery=0.0,
    )

    all_skill_ids = sorted(set(objective_skills.keys()) | set(student_skills.keys()))
    bars: List[Dict[str, Any]] = []

    for sid in all_skill_ids:
        obj = objective_skills.get(sid)
        stu = student_skills.get(sid)

        to_learn = obj["mastery"] if obj else 0.0
        learnt = stu["mastery"] if stu else 0.0
        gap = round(max(0.0, to_learn - learnt), 4)

        if to_learn > 0.0 and learnt == 0.0:
            status = "to_learn"
        elif to_learn > 0.0 and learnt < to_learn:
            status = "in_progress"
        else:
            status = "learnt"

        bars.append(
            {
                "skillId": sid,
                "skillName": (obj or stu or {}).get("name", sid),
                "learnt": round(learnt, 4),
                "toLearn": round(to_learn, 4),
                "gap": gap,
                "status": status,
            }
        )

    bars.sort(key=lambda x: (x["gap"], x["toLearn"], -x["learnt"]), reverse=True)
    selected = bars[:max_bars]

    return {
        "studentId": student_id,
        "graph": {
            "type": "bar",
            "title": "Skill Summary (Learnt vs To Learn)",
            "xKey": "skillName",
            "series": [
                {"key": "learnt", "label": "Learnt"},
                {"key": "toLearn", "label": "To Learn"},
            ],
            "maxBars": max_bars,
            "bars": selected,
        },
        "summary": {
            "totalObjectiveSkills": len(objective_skills),
            "totalLearntSkills": len(student_skills),
            "selectedBars": len(selected),
        },
    }

# agent/weakness.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import torch

from dkvmn.infer import load_model, load_skill_map  # existing loader :contentReference[oaicite:3]{index=3}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _as_vec3(v: Any) -> Optional[Tuple[float, float, float]]:
    """
    Accepts either:
      - list[3] as in your synthetic skills.json :contentReference[oaicite:4]{index=4}
      - dict with keys mastery/edge_case/implementation (your new schema)
    """
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1]), float(v[2])

    if isinstance(v, dict):
        a = v.get("mastery")
        b = v.get("edge_case")
        c = v.get("implementation")
        if all(isinstance(x, (int, float)) for x in (a, b, c)):
            return float(a), float(b), float(c)

    return None


def _vec_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    # simple L2 (bounded in [0, sqrt(3)] if vectors are in [0,1])
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _extract_skill_id(doc: Dict[str, Any]) -> Optional[str]:
    sid = doc.get("skillId") or doc.get("id")
    return sid if isinstance(sid, str) else None


def _extract_skill_name(doc: Dict[str, Any]) -> str:
    nm = doc.get("name")
    return nm if isinstance(nm, str) and nm.strip() else (_extract_skill_id(doc) or "unknown")


def _extract_history_events(student_profile_or_rows: Any) -> List[Dict[str, Any]]:
    """
    Looks for history in common places:
      - doc["history"]
      - doc["interactions"]
      - doc["attempts"]
      - OR if student container is "one row per event": those rows themselves
    Expected event shape: {"skillId": "...", "correct": 0|1}
    """
    if isinstance(student_profile_or_rows, dict):
        for k in ("history", "interactions", "attempts", "events"):
            v = student_profile_or_rows.get(k)
            if isinstance(v, list):
                return [e for e in v if isinstance(e, dict)]
        return []

    if isinstance(student_profile_or_rows, list):
        # If rows are events, keep those that look like events
        events = []
        for r in student_profile_or_rows:
            if not isinstance(r, dict):
                continue
            if ("correct" in r) and (_extract_skill_id(r) is not None):
                events.append(r)
        return events

    return []


def compute_weak_skills(
    *,
    objective_skill_docs: List[Dict[str, Any]],
    student_skill_docs: List[Dict[str, Any]],
    student_profile_doc: Optional[Dict[str, Any]],
    model_path: str = "dkvmn_model.pt",
    skill_map_path: str = "skill2idx.json",
    top_k: int = 5,
    use_names: bool = True,
) -> List[str]:
    """
    Returns ["skill name", ...] sorted from weakest to strongest.

    Delta model:
      delta = w_mastery*(1 - dkvmn_p_correct) + w_gap*(vector_gap) + w_diff*(difficulty)
    where:
      vector_gap = distance(objective_vector, student_vector)

    If DKVMN history is missing, we fall back to just vector_gap + difficulty.
    """
    # Index objective + student docs by skillId
    obj_by_id: Dict[str, Dict[str, Any]] = {}
    for d in objective_skill_docs:
        sid = _extract_skill_id(d)
        if sid:
            obj_by_id[sid] = d

    stu_by_id: Dict[str, Dict[str, Any]] = {}
    for d in student_skill_docs:
        sid = _extract_skill_id(d)
        if sid:
            stu_by_id[sid] = d

    # DKVMN mastery (optional)
    history_events = _extract_history_events(student_profile_doc) if student_profile_doc else _extract_history_events(student_skill_docs)

    skill2idx = load_skill_map(skill_map_path)  # :contentReference[oaicite:5]{index=5}
    dkvmn_scores: Dict[str, float] = {}

    if history_events:
        # Build tensors (same rules as your infer helpers) :contentReference[oaicite:6]{index=6}
        skills = []
        corrects = []
        for ev in history_events:
            sid = _extract_skill_id(ev)
            c = ev.get("correct")
            if isinstance(sid, str) and sid in skill2idx and c in (0, 1):
                skills.append(skill2idx[sid])
                corrects.append(int(c))

        if skills:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(model_path, n_skills=len(skill2idx), device=device)  # :contentReference[oaicite:7]{index=7}

            hs = torch.tensor(skills, dtype=torch.long, device=device)
            hc = torch.tensor(corrects, dtype=torch.long, device=device)

            mastery_vec = model.infer_mastery(hs, hc)  # model method :contentReference[oaicite:8]{index=8}
            # invert
            idx2skill = {v: k for k, v in skill2idx.items()}
            for i in range(mastery_vec.shape[0]):
                dkvmn_scores[idx2skill[i]] = float(mastery_vec[i].item())

    # Compute deltas over the union of skills we know about
    all_skill_ids = set(obj_by_id.keys()) | set(stu_by_id.keys())
    if not all_skill_ids:
        return []

    w_mastery = 0.70
    w_gap = 0.25
    w_diff = 0.05

    scored: List[Tuple[str, float]] = []
    for sid in all_skill_ids:
        obj = obj_by_id.get(sid, {})
        stu = stu_by_id.get(sid, {})

        obj_vec = _as_vec3(obj.get("vector"))
        stu_vec = _as_vec3(stu.get("vector"))

        gap = 0.0
        if obj_vec and stu_vec:
            gap = _vec_distance(obj_vec, stu_vec) / math.sqrt(3.0)  # normalize to ~[0,1]
            gap = _clamp01(gap)

        difficulty = obj.get("difficulty")
        difficulty = float(difficulty) if isinstance(difficulty, (int, float)) else 0.5
        difficulty = _clamp01(difficulty)

        p_correct = dkvmn_scores.get(sid)  # None if not available
        if isinstance(p_correct, (int, float)):
            p_correct = _clamp01(float(p_correct))
            delta = w_mastery * (1.0 - p_correct) + w_gap * gap + w_diff * difficulty
        else:
            # No DKVMN: just compare vectors + difficulty
            delta = (0.85 * gap) + (0.15 * difficulty)

        name_or_id = _extract_skill_name(obj) if use_names else sid
        scored.append((name_or_id, float(delta)))

    scored.sort(key=lambda x: x[1], reverse=True)  # higher delta = weaker
    return [name for name, _ in scored[:top_k]]
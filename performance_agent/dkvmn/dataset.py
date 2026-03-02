# dkvmn/dataset.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset


# ----------------------------
# Skills
# ----------------------------
def load_skills(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_skill2idx(skills: List[dict]) -> Dict[str, int]:
    """
    Accepts skill docs with either:
      - {"skillId": "..."} (existing)
      - {"id": "..."}      (your new schema)
    """
    skill_ids: List[str] = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        sid = s.get("skillId") or s.get("id")
        if isinstance(sid, str) and sid.strip():
            skill_ids.append(sid.strip())

    # stable ordering
    skill_ids = sorted(set(skill_ids))
    return {sid: i for i, sid in enumerate(skill_ids)}


# ----------------------------
# Sequences
# ----------------------------
def _parse_ts(val: Any) -> Optional[float]:
    """
    Tries to convert timestamps into a sortable float.
    Supports:
      - int/float epochs
      - numeric strings
      - ISO strings => None (we'll keep input order if only ISO strings exist)
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except Exception:
            return None
    return None


def _extract_event_fields(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[float]]:
    sid = ev.get("skillId") or ev.get("id")
    c = ev.get("correct")

    # timestamp field names you might use
    ts_raw = (
        ev.get("timestamp")
        or ev.get("ts")
        or ev.get("time")
        or ev.get("createdAt")
        or ev.get("created_at")
    )
    ts = _parse_ts(ts_raw)

    if not isinstance(sid, str):
        sid = None
    else:
        sid = sid.strip()

    if c in (0, 1):
        corr = int(c)
    else:
        corr = None

    return sid, corr, ts


def normalize_sequences(
    rows: List[dict],
    *,
    sort_by_time: bool = True,
) -> List[dict]:
    """
    Normalizes input JSONL rows into:
      [{"student_id": "...", "seq": [{"skillId": "...", "correct": 0/1, "ts": <optional>}, ...]}, ...]

    Supports TWO input styles:

    A) per-student rows:
       {"student_id" or "studentId": "...", "seq" or "history" or "interactions": [...]}

    B) event log rows:
       {"student_id"/"studentId": "...", "skillId": "...", "correct": 0/1, "timestamp": ...}
    """
    if not rows:
        return []

    # detect A) if at least one row has a seq-like field
    has_seq_style = any(
        isinstance(r, dict) and isinstance(r.get("seq") or r.get("history") or r.get("interactions"), list)
        for r in rows
    )

    out: List[dict] = []

    if has_seq_style:
        # A) already grouped
        for r in rows:
            if not isinstance(r, dict):
                continue
            student_id = r.get("student_id") or r.get("studentId") or r.get("studentID")
            if not isinstance(student_id, str) or not student_id.strip():
                continue
            student_id = student_id.strip()

            seq = r.get("seq") or r.get("history") or r.get("interactions") or []
            if not isinstance(seq, list):
                continue

            norm_seq = []
            for ev in seq:
                if not isinstance(ev, dict):
                    continue
                sid, corr, ts = _extract_event_fields(ev)
                if sid is None or corr is None:
                    continue
                item = {"skillId": sid, "correct": corr}
                if ts is not None:
                    item["ts"] = ts
                norm_seq.append(item)

            if sort_by_time and norm_seq and all(("ts" in e) for e in norm_seq):
                norm_seq.sort(key=lambda e: e["ts"])

            out.append({"student_id": student_id, "seq": norm_seq})

        return out

    # B) event log: group by student
    grouped: Dict[str, List[dict]] = defaultdict(list)

    for r in rows:
        if not isinstance(r, dict):
            continue
        student_id = r.get("student_id") or r.get("studentId") or r.get("studentID")
        if not isinstance(student_id, str) or not student_id.strip():
            continue
        student_id = student_id.strip()

        sid, corr, ts = _extract_event_fields(r)
        if sid is None or corr is None:
            continue

        item = {"skillId": sid, "correct": corr}
        if ts is not None:
            item["ts"] = ts
        grouped[student_id].append(item)

    for student_id, seq in grouped.items():
        if sort_by_time and seq and all(("ts" in e) for e in seq):
            seq.sort(key=lambda e: e["ts"])
        out.append({"student_id": student_id, "seq": seq})

    return out


def load_sequences(path: str, *, sort_by_time: bool = True) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return normalize_sequences(rows, sort_by_time=sort_by_time)


# ----------------------------
# Dataset
# ----------------------------
class DKVMNDataset(Dataset):
    """
    Produces (skill_seq, correct_seq) per student, truncated to max_len.
    """
    def __init__(
        self,
        sequences: List[dict],
        skill2idx: Dict[str, int],
        *,
        max_len: int = 200,
        min_len: int = 2,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.skill2idx = skill2idx
        self.max_len = int(max_len)
        self.min_len = int(min_len)

        for row in sequences:
            if not isinstance(row, dict):
                continue
            seq = row.get("seq", [])
            if not isinstance(seq, list):
                continue

            skills: List[int] = []
            corrects: List[int] = []

            for ev in seq[: self.max_len]:
                if not isinstance(ev, dict):
                    continue
                sid = ev.get("skillId") or ev.get("id")
                c = ev.get("correct")
                if isinstance(sid, str) and sid in skill2idx and c in (0, 1):
                    skills.append(skill2idx[sid])
                    corrects.append(int(c))

            if len(skills) >= self.min_len:
                self.samples.append(
                    (torch.tensor(skills, dtype=torch.long), torch.tensor(corrects, dtype=torch.long))
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pads sequences to max length in batch.
    Returns:
      skill_seq (B,T), correct_seq (B,T), mask (B,T)
    """
    B = len(batch)
    maxT = max(x[0].shape[0] for x in batch)

    skill = torch.zeros(B, maxT, dtype=torch.long)
    corr = torch.zeros(B, maxT, dtype=torch.long)
    mask = torch.zeros(B, maxT, dtype=torch.float32)

    for i, (s, c) in enumerate(batch):
        T = s.shape[0]
        skill[i, :T] = s
        corr[i, :T] = c
        mask[i, :T] = 1.0

    return skill, corr, mask
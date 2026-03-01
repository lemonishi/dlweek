import json
import torch
from typing import Dict, List

from dkvmn.model import DKVMN


def load_skill_map(path: str) -> Dict[str, int]:
    with open(path, "r") as f:
        return json.load(f)


def load_model(model_path: str, n_skills: int, device: torch.device) -> DKVMN:
    ckpt = torch.load(model_path, map_location=device)
    model = DKVMN(n_skills=n_skills)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def interactions_to_tensors(interactions: List[dict], skill2idx: Dict[str, int]):
    skills = []
    corrects = []
    for ev in interactions:
        sid = ev.get("skillId") or ev.get("id")
        c = ev.get("correct")
        if isinstance(sid, str) and sid in skill2idx and c in (0, 1):
            skills.append(skill2idx[sid])
            corrects.append(int(c))
    if not skills:
        return None, None
    return torch.tensor(skills, dtype=torch.long), torch.tensor(corrects, dtype=torch.long)


def mastery_scores_from_history(
    interactions: List[dict],
    skill2idx: Dict[str, int],
    model_path: str = "dkvmn_model.pt",
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, n_skills=len(skill2idx), device=device)

    s, c = interactions_to_tensors(interactions, skill2idx)
    if s is None:
        return {}

    s = s.to(device)
    c = c.to(device)

    mastery = model.infer_mastery(s, c)  # (n_skills,)
    # invert mapping
    idx2skill = {v: k for k, v in skill2idx.items()}

    out = {idx2skill[i]: float(mastery[i].item()) for i in range(mastery.shape[0])}
    return out
"""Validate skill objects against expected schema using pydantic."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class VectorModel(BaseModel):
    mastery: float
    edge_case: float
    implementation: float

    @validator('mastery', 'edge_case', 'implementation', pre=True)
    def float_default(cls, v):
        return float(v) if v not in (None, "") else 0.0


class SkillModel(BaseModel):
    id: str
    studentId: Optional[str] = Field(default="")
    name: str
    description: str
    tags: List[str]
    vector: VectorModel
    prerequisites: List[Any]
    difficulty: float
    version: int
    timestamp: Optional[str] = None

    # coerce empties
    @validator('difficulty', pre=True)
    def diff_default(cls, v):
        # try numeric first
        try:
            return float(v)
        except Exception:
            # handle common descriptive terms
            if isinstance(v, str):
                text = v.strip().lower()
                if text in ('beginner', 'easy'):
                    return 1.0
                if text in ('intermediate', 'medium'):
                    return 2.0
                if text in ('advanced', 'hard'):
                    return 3.0
            # fallback to zero if conversion fails
            return 0.0

    @validator('version', pre=True)
    def version_default(cls, v):
        return int(v) if v not in (None, "") else 1

    @validator('prerequisites', pre=True)
    def prereq_default(cls, v):
        return v if isinstance(v, list) else []


class SkillList(BaseModel):
    skills_learnt: List[SkillModel]
    skills_to_learn: List[SkillModel]

    @validator("skills_learnt", "skills_to_learn", pre=True, each_item=False)
    def default_empty(cls, v):
        return v or []

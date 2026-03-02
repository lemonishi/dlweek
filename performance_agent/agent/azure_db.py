from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from azure.cosmos import CosmosClient


@dataclass(frozen=True)
class CosmosConfig:
    endpoint: str
    key: str
    database_name: str
    objectives_container: str
    students_container: str


def load_cosmos_config_from_env() -> CosmosConfig:
    endpoint = os.environ.get("COSMOS_ENDPOINT")
    key = os.environ.get("COSMOS_KEY")
    db = os.environ.get("COSMOS_DB_NAME")
    obj = os.environ.get("COSMOS_OBJECTIVES_CONTAINER")
    stu = os.environ.get("COSMOS_STUDENTS_CONTAINER")

    missing = [k for k, v in {
        "COSMOS_ENDPOINT": endpoint,
        "COSMOS_KEY": key,
        "COSMOS_DB_NAME": db,
        "COSMOS_OBJECTIVES_CONTAINER": obj,
        "COSMOS_STUDENTS_CONTAINER": stu,
    }.items() if not v]

    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    return CosmosConfig(
        endpoint=endpoint,
        key=key,
        database_name=db,
        objectives_container=obj,
        students_container=stu,
    )


class CosmosRepo:
    """
    Minimal Cosmos repo (SQL API).
    Assumes both containers are queryable by studentId (partition key ideally),
    but includes safe fallbacks if your schema differs.
    """
    def __init__(self, cfg: CosmosConfig):
        self.client = CosmosClient(cfg.endpoint, credential=cfg.key)
        self.db = self.client.get_database_client(cfg.database_name)
        self.objectives = self.db.get_container_client(cfg.objectives_container)
        self.students = self.db.get_container_client(cfg.students_container)

    def _q(self, container, query: str, params: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        params = params or []
        items = container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,  # safe default
        )
        return list(items)

    # -------------------------
    # Objectives (target state)
    # -------------------------
    def get_objective_skills(self, *, student_id: str) -> List[Dict[str, Any]]:
        # Most common: documents have a studentId field
        res = self._q(
            self.objectives,
            "SELECT * FROM c WHERE c.studentId = @sid",
            [{"name": "@sid", "value": student_id}],
        )
        if res:
            return res

        # Fallback: maybe objectives are global skills (no per-student partition)
        res = self._q(self.objectives, "SELECT * FROM c")
        return res

    # -------------------------
    # Student (current state)
    # -------------------------
    def get_student_skill_states(self, *, student_id: str) -> List[Dict[str, Any]]:
        # If student container stores one doc per skill-state:
        res = self._q(
            self.students,
            "SELECT * FROM c WHERE c.studentId = @sid",
            [{"name": "@sid", "value": student_id}],
        )
        return res

    def get_student_profile_doc(self, *, student_id: str) -> Optional[Dict[str, Any]]:
        """
        If your students container instead has ONE doc per student with arrays like:
          { studentId: "...", skills: [...], history: [...] }
        then this will find it.
        """
        res = self._q(
            self.students,
            "SELECT * FROM c WHERE c.studentId = @sid AND (IS_DEFINED(c.skills) OR IS_DEFINED(c.history) OR IS_DEFINED(c.interactions))",
            [{"name": "@sid", "value": student_id}],
        )
        return res[0] if res else None
"""Transform Cosmos DB responses into unified skill schema and analyze learning progress."""

import json
from typing import List, Dict, Any
from fetch_db import get_student_interactions, get_skills


def transform_interaction_to_skill_schema(
    interaction: Dict[str, Any],
    skill_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Transform a student interaction into the unified skill schema."""
    
    signal_value = interaction.get("signal", {}).get("value", 0)
    
    # Extract vector from interaction or skill_info
    vector_raw = interaction.get("vector")
    if isinstance(vector_raw, dict):
        vector = vector_raw
    elif isinstance(vector_raw, list) and len(vector_raw) >= 3:
        vector = {
            "mastery": vector_raw[0],
            "edge_case": vector_raw[1],
            "implementation": vector_raw[2]
        }
    else:
        vector = {"mastery": signal_value, "edge_case": 0.5, "implementation": 0.5}
    
    return {
        "id": interaction.get("skillId", ""),
        "studentId": interaction.get("studentId", ""),
        "name": skill_info.get("name", "") if skill_info else "",
        "description": skill_info.get("description", "") if skill_info else "",
        "tags": skill_info.get("tags", []) if skill_info else [],
        "vector": vector,
        "prerequisites": skill_info.get("prerequisites", []) if skill_info else [],
        "difficulty": skill_info.get("difficulty", 0) if skill_info else 0,
        "version": skill_info.get("version", 1) if skill_info else 1,
        "timestamp": interaction.get("timestamp", ""),
    }


def analyze_student_skills(student_id: str, mastery_threshold: float = 0.5) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze a student's learning progress.
    
    Returns:
        Dict with two lists:
        - "skills_learnt": Skills with mastery >= threshold
        - "skills_to_learn": Skills not yet attempted or mastery < threshold
    """
    
    # Fetch all data
    interactions = get_student_interactions(student_id)
    
    # Get all domains from interactions
    domains = set(i.get("domain") for i in interactions if i.get("domain"))
    
    # Map skill IDs to their full info
    skill_map = {}
    for domain in domains:
        skills = get_skills(domain)
        for skill in skills:
            skill_map[skill.get("skillId") or skill.get("id")] = skill
    
    # Transform interactions to schema
    learned_skills = []
    attempted_skill_ids = set()
    
    for interaction in interactions:
        skill_id = interaction.get("skillId", "")
        attempted_skill_ids.add(skill_id)
        
        skill_info = skill_map.get(skill_id)
        transformed = transform_interaction_to_skill_schema(interaction, skill_info)
        
        mastery_level = transformed["vector"].get("mastery", 0)
        if mastery_level >= mastery_threshold:
            learned_skills.append(transformed)
    
    # Find skills to learn (in same domains but not attempted)
    to_learn_skills = []
    for domain in domains:
        all_skills = get_skills(domain)
        for skill in all_skills:
            skill_id = skill.get("skillId") or skill.get("id")
            if skill_id not in attempted_skill_ids:
                to_learn_skills.append({
                    "id": skill_id,
                    "studentId": student_id,
                    "name": skill.get("name", ""),
                    "description": skill.get("description", ""),
                    "tags": skill.get("tags", []),
                    "vector": {
                        "mastery": 0.0,
                        "edge_case": 0.0,
                        "implementation": 0.0
                    },
                    "prerequisites": skill.get("prerequisites", []),
                    "difficulty": skill.get("difficulty", 0),
                    "version": skill.get("version", 1),
                    "timestamp": None,
                })
    
    return {
        "skills_learnt": learned_skills,
        "skills_to_learn": to_learn_skills
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze student learning progress")
    parser.add_argument("student_id", help="Student ID to analyze")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mastery threshold (0-1)")
    args = parser.parse_args()
    
    result = analyze_student_skills(args.student_id, args.threshold)
    
    print(json.dumps(result, indent=2))
    print(f"\n📊 Summary:")
    print(f"   Skills Learnt: {len(result['skills_learnt'])}")
    print(f"   Skills to Learn: {len(result['skills_to_learn'])}")

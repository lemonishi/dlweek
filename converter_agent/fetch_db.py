"""Standalone script demonstrating queries against the Cosmos DB instance.

Usage examples:

    # fetch interactions for a student
    python fetch_db.py --student student_123

    # fetch skills for a domain
    python fetch_db.py --domain math

    # both
    python fetch_db.py --student student_123 --domain math
"""

import json
from typing import List, Dict, Any
from cosmos_conn import container


def get_student_interactions(student_id: str) -> List[Dict[str, Any]]:
    c = container("student_interactions")
    query = "SELECT * FROM c WHERE c.studentId = @sid ORDER BY c.timestamp DESC"
    items = list(c.query_items(
        query=query,
        parameters=[{"name": "@sid", "value": student_id}],
        enable_cross_partition_query=True,
    ))
    return items


def get_skills(domain: str) -> List[Dict[str, Any]]:
    c = container("skill_library")
    query = "SELECT * FROM c WHERE c.domain = @d"
    items = list(c.query_items(
        query=query,
        parameters=[{"name": "@d", "value": domain}],
        enable_cross_partition_query=True,
    ))
    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch data from Cosmos DB containers")
    parser.add_argument("--student", help="student id to query interactions")
    parser.add_argument("--domain", help="domain to query skills")
    args = parser.parse_args()

    if args.student:
        interactions = get_student_interactions(args.student)
        print(f"Found {len(interactions)} interaction(s) for student '{args.student}'")
        print(json.dumps(interactions[:3], indent=2))

    if args.domain:
        skills = get_skills(args.domain)
        print(f"Found {len(skills)} skill(s) for domain '{args.domain}'")
        print(json.dumps(skills, indent=2))

    if not args.student and not args.domain:
        parser.print_help()

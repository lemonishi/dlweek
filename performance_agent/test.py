import json
from agent.weakness import compute_weak_skills

# Load demo data
objective = json.load(open("data/demo_objectives.json"))
student_profile = json.load(open("data/demo_students.json"))

weak = compute_weak_skills(
    objective_skill_docs=objective,
    student_skill_docs=student_profile["skills"],
    student_profile_doc=student_profile,
    model_path="dkvmn_model.pt",
    skill_map_path="skill2idx.json",
    top_k=4
)

print("\nWeak skills detected:\n")

for s in weak:
    print("-", s)
import subprocess
import sys

STEPS = [
    "generate_data.py",
    "data_pipeline.py",
    "validate.py",
    "run_queries.py",
    "analysis.py",
    "part_b.py",
    "lapse_model.py",
    "triage.py",
    "rag.py",
    "agent.py",
]

for script in STEPS:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, script], check=True)

print("\nAll parts finished.")
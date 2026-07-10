"""
Automated evaluation harness for the Healthcare RAG Agent.
Runs the Golden Set of 20 queries through the classification node
and verifies that intent extraction matches expectations.
"""
from __future__ import annotations

import os
import sys
import yaml

# Add the project root to sys.path so we can import 'agent'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agent.classify import test_classify_standalone

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def load_golden_set(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def run_evals():
    golden_set_path = os.path.join(os.path.dirname(__file__), "golden_set.yaml")
    queries = load_golden_set(golden_set_path)

    print(f"Running evaluation on {len(queries)} queries...\n")

    passed = 0
    failed = 0

    for i, test in enumerate(queries, 1):
        query = test["query"]
        expected_type = test["expected_type"]
        expected_states = set(test["expected_states"] or [])
        expected_ftype = test["expected_facility_type"]

        print(f"[{i:02d}] QUERY: {query}")
        
        try:
            # We only evaluate the classification node here because it's deterministic 
            # and is the highest-leverage point of failure.
            result = test_classify_standalone(query)
            
            actual_type = result.query_type
            actual_states = set(result.states or [])
            actual_ftype = result.facility_type

            errors = []
            if actual_type != expected_type:
                errors.append(f"Type mismatch: expected '{expected_type}', got '{actual_type}'")
            if actual_states != expected_states:
                errors.append(f"States mismatch: expected {expected_states}, got {actual_states}")
            if actual_ftype != expected_ftype:
                errors.append(f"Facility mismatch: expected '{expected_ftype}', got '{actual_ftype}'")

            if not errors:
                print(GREEN + "  [PASS]" + RESET)
                passed += 1
            else:
                print(RED + "  [FAIL]" + RESET)
                for err in errors:
                    print(RED + f"    - {err}" + RESET)
                print(RED + f"    - Got: {result.model_dump_json(indent=2)}" + RESET)
                failed += 1

        except Exception as e:
            print(RED + f"  [ERROR] {e}" + RESET)
            failed += 1

    print("\n" + "=" * 40)
    print(f"EVALUATION COMPLETE")
    print(f"Passed: {passed}/{len(queries)}")
    print(f"Failed: {failed}/{len(queries)}")
    print("=" * 40)
    
    if failed > 0:
        exit(1)


if __name__ == "__main__":
    run_evals()

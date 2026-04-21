"""
Record live HuggingFace Space responses into the fixtures cache.

Use this to refresh fixtures with REAL HF Space outputs instead of synthetic.

Usage:
    SPACES_MODE=record HF_TOKEN=hf_xxx python scripts/record_fixtures.py [--task TASK_ID]

Without --task, records for ALL tasks in fixtures/tasks.json.

WARNING: this hits live HF Spaces and will:
  - Be slow (2-30s per Space call)
  - Consume HF Inference API quota
  - Require an HF token if your Spaces are gated

Recommended workflow:
  1. Use synthetic mock data (default) for development + training.
  2. Run this script periodically (or before demo) to refresh with real outputs.
  3. Commit the responses/ directory updates.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force record mode
os.environ["SPACES_MODE"] = "record"

from server.space_caller import SpaceCaller  # noqa: E402
from server.space_catalog import SpaceCatalog  # noqa: E402

FIXTURES_DIR = ROOT / "fixtures"
TASKS_FILE = FIXTURES_DIR / "tasks.json"


def resolve_inputs(step_inputs, task_input):
    """Resolve <input.X> placeholders in a gold pipeline step's inputs."""
    resolved = {}
    for k, v in step_inputs.items():
        if isinstance(v, str) and v.startswith("<input."):
            field = v.replace("<input.", "").replace(">", "")
            resolved[k] = task_input.get(field, "")
        elif isinstance(v, str) and v.startswith("<step"):
            # Cannot resolve cross-step refs without running pipeline; skip
            resolved[k] = "PLACEHOLDER_STEP_REF"
        else:
            resolved[k] = v
    return resolved


def record_task(task, catalog, caller):
    """Record live responses for one task's gold pipeline."""
    task_id = task["task_id"]
    print(f"\n=== Task: {task_id} ===")
    gold = task.get("gold_pipeline", [])
    if not gold:
        print("  (no gold pipeline; skipping)")
        return 0

    success = 0
    for step in gold:
        space_id = step["space_id"]
        # Verify Space exists in catalog (or is reachable)
        card = catalog.read_card(space_id)
        if card is None:
            print(f"  ✗ {space_id}: card unavailable")
            continue

        inputs = resolve_inputs(step.get("inputs", {}), task.get("input", {}))

        print(f"  → calling {space_id}...")
        response = caller.call(task_id, space_id, inputs)
        if response.get("success"):
            print(f"  ✓ recorded ({len(str(response.get('output')))} chars)")
            success += 1
        else:
            print(f"  ✗ failed: {response.get('error', '')[:100]}")

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Record only this task ID")
    parser.add_argument("--list", action="store_true", help="List task IDs and exit")
    args = parser.parse_args()

    tasks = json.loads(TASKS_FILE.read_text())

    if args.list:
        print("Available tasks:")
        for t in tasks:
            print(f"  {t['task_id']:<35} {t.get('domain', ''):<20} {t.get('difficulty', '')}")
        return

    catalog = SpaceCatalog(mode="record")
    caller = SpaceCaller(mode="record")

    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
        if not tasks:
            print(f"Task '{args.task}' not found")
            sys.exit(1)

    total_success = 0
    for task in tasks:
        total_success += record_task(task, catalog, caller)

    print(f"\nTotal successful recordings: {total_success}")


if __name__ == "__main__":
    main()

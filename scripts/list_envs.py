#!/usr/bin/env python3
"""List registered xTerra mjlab tasks."""

import xterra_mjlab.tasks  # noqa: F401

from mjlab.tasks.registry import list_tasks

if __name__ == "__main__":
    tasks = sorted(t for t in list_tasks() if t.startswith("xTerra-"))
    print("\nRegistered xTerra mjlab tasks:")
    for task_id in tasks:
        print(f"  {task_id}")

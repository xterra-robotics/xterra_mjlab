#!/usr/bin/env python3
"""Train an xTerra M2Metal mjlab policy (vanilla PPO).

Example:
    python scripts/train.py xTerra-Mjlab-Velocity-Flat-M2Metal --env.scene.num-envs 4096
"""

import xterra_mjlab.tasks  # noqa: F401  (registers the xTerra tasks)

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()

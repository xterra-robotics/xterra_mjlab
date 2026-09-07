#!/usr/bin/env python3
"""Train an xTerra SvanM2 mjlab policy (vanilla PPO).

Example:
    python scripts/train.py xTerra-Mjlab-Velocity-Flat-SvanM2 --env.scene.num-envs 4096
"""

import xterra_mjlab.tasks  # noqa: F401  (registers the xTerra tasks)

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()

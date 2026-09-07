#!/usr/bin/env python3
"""Play / visualize a trained xTerra SvanM2 mjlab policy.

Example:
    python scripts/play.py xTerra-Mjlab-Velocity-Flat-SvanM2 --checkpoint-file <run>/model_3000.pt
    python scripts/play.py xTerra-Mjlab-Velocity-Flat-SvanM2 --agent zero   # no policy
"""

import xterra_mjlab.tasks  # noqa: F401  (registers the xTerra tasks)

from mjlab.scripts.play import main

if __name__ == "__main__":
    main()

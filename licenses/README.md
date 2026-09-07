# Third-party licenses

`xterra_mjlab` is licensed under Apache-2.0 (see the top-level `LICENSE` and
`NOTICE`). It depends on the following third-party components at runtime; their
licenses are collected here for convenience, with canonical text upstream.

| Component | License | File | Redistributed here |
|-----------|---------|------|--------------------|
| mjlab (`mjlab`) | Apache-2.0 | [mjlab-Apache-2.0.txt](mjlab-Apache-2.0.txt) | No (pip install) |
| rsl_rl (`rsl-rl-lib`) | BSD-3-Clause | [rsl_rl-BSD-3-Clause.txt](rsl_rl-BSD-3-Clause.txt) | No (pip install) |
| MuJoCo / MuJoCo-Warp / Warp | Apache-2.0 | [MuJoCo-and-Warp-Apache-2.0.txt](MuJoCo-and-Warp-Apache-2.0.txt) | No (pip install) |
| PyTorch, Gymnasium, NumPy | BSD / MIT | (see each package's distribution) | No (pip install) |

None of these are vendored into this repository. The only vendored artifacts are
the SvanM2 robot description (MJCF + STL meshes) under
`xterra_mjlab/assets/svanm2/`, which is xTerra's own and released under this
repository's Apache-2.0 license (see `NOTICE`).

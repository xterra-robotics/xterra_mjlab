"""xTerra SvanM2 robot constants and configuration for mjlab.

SvanM2 is a quadruped (FL/FR/RL/RR, uniform ``*_hip/thigh/calf_joint``). Uses
the vendored ``xml/svanm2_mjlab.xml`` (adds ``imu_ang_vel`` / ``imu_lin_vel``
body-frame sensors so the observation layout matches the rest of the pipeline).

Actuation uses the SvanM2 parallel-mechanism actuator (a single group over all
12 leg joints) so simulation reproduces the thigh->calf belt coupling and the
joint<->motor gain transform of the real drivetrain. Gains kp=20, kd=0.7;
joint-side effort 12 N.m (hip/thigh) / 24 N.m (calf).
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from xterra_mjlab.actuators import SvanM2ActuatorCfg

SVANM2_XML: Path = Path(__file__).parent / "xml" / "svanm2_mjlab.xml"
assert SVANM2_XML.exists(), f"SvanM2 XML not found at {SVANM2_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SVANM2_XML))


STIFFNESS = 20.0
DAMPING = 0.7
EFFORT_HIP = 12.0
EFFORT_KNEE = 24.0

# Single actuator group over all 12 leg joints — required for the per-leg 3x3
# belt-coupling block to live in one Jacobian. effort_limit sizes the MJCF
# <motor> forcerange (max); the per-joint clamp comes from effort_hip/knee.
SVANM2_ACTUATOR_CFG = SvanM2ActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_KNEE,
    effort_hip=EFFORT_HIP,
    effort_knee=EFFORT_KNEE,
    armature=0.01,
    gear_ratio=(8.0, 8.0, 16.0),
    hfe_kfe_trans=0.5,
)
SVANM2_ACTUATOR_DELAYED_CFG = SvanM2ActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_KNEE,
    effort_hip=EFFORT_HIP,
    effort_knee=EFFORT_KNEE,
    armature=0.01,
    gear_ratio=(8.0, 8.0, 16.0),
    hfe_kfe_trans=0.5,
    delay_min_lag=0,
    delay_max_lag=2,
    delay_update_period=0,
)

# Nominal ~0.32 m stance.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.34),
    joint_pos={
        ".*_hip_joint": 0.0,
        "F.*_thigh_joint": 0.5806,
        "R.*_thigh_joint": 0.7167,
        "F.*_calf_joint": -1.1716,
        "R.*_calf_joint": -1.1342,
    },
    joint_vel={".*": 0.0},
)

_foot_regex = "^(FL|FR|RL|RR)_foot$"
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    condim={_foot_regex: 3},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
    contype=1,
    conaffinity=0,
    disable_other_geoms=False,
)

SVANM2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(SVANM2_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)
SVANM2_ARTICULATION_DELAYED = EntityArticulationInfoCfg(
    actuators=(SVANM2_ACTUATOR_DELAYED_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_svanm2_robot_cfg(delayed: bool = False) -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=SVANM2_ARTICULATION_DELAYED if delayed else SVANM2_ARTICULATION,
    )


# Action scale: mjlab convention 0.25 * effort / kp per joint (effective joint
# stiffness is kp by the parallel-mechanism gain transform).
SVANM2_ACTION_SCALE: dict[str, float] = {
    ".*_hip_joint": 0.25 * EFFORT_HIP / STIFFNESS,
    ".*_thigh_joint": 0.25 * EFFORT_HIP / STIFFNESS,
    ".*_calf_joint": 0.25 * EFFORT_KNEE / STIFFNESS,
}

"""Register the xTerra M2Metal flat velocity task (vanilla PPO).

Task: ``xTerra-Mjlab-Velocity-Flat-M2Metal`` — flat-terrain velocity tracking,
standard MLP actor-critic PPO via mjlab's ``VelocityOnPolicyRunner``.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import m2_metal_flat_env_cfg
from .rl_cfg import m2_metal_ppo_runner_cfg

FLAT_TASK = "xTerra-Mjlab-Velocity-Flat-M2Metal"

register_mjlab_task(
    task_id=FLAT_TASK,
    env_cfg=m2_metal_flat_env_cfg(),
    play_env_cfg=m2_metal_flat_env_cfg(play=True),
    rl_cfg=m2_metal_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

__all__ = ["FLAT_TASK"]

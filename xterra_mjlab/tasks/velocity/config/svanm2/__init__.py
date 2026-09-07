"""Register the xTerra SvanM2 flat velocity task (vanilla PPO).

Task: ``xTerra-Mjlab-Velocity-Flat-SvanM2`` — flat-terrain velocity tracking,
standard MLP actor-critic PPO via mjlab's ``VelocityOnPolicyRunner``.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import svanm2_flat_env_cfg
from .rl_cfg import svanm2_ppo_runner_cfg

FLAT_TASK = "xTerra-Mjlab-Velocity-Flat-SvanM2"

register_mjlab_task(
    task_id=FLAT_TASK,
    env_cfg=svanm2_flat_env_cfg(),
    play_env_cfg=svanm2_flat_env_cfg(play=True),
    rl_cfg=svanm2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

__all__ = ["FLAT_TASK"]

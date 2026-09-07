"""RL configuration for the xTerra SvanM2 velocity task (vanilla PPO)."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

_PPO_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)

_CRITIC_CFG = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
)

_ACTOR_CFG = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    },
)


def svanm2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Standard MLP actor-critic PPO runner."""
    return RslRlOnPolicyRunnerCfg(
        actor=_ACTOR_CFG,
        critic=_CRITIC_CFG,
        algorithm=_PPO_CFG,
        experiment_name="svanm2_velocity_flat",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=3000,
    )

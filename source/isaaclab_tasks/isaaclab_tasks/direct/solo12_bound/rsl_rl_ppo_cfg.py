"""RSL-RL configuration for the isolated Solo12 bound expert."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_tasks.direct.solo12.agents.rsl_rl_ppo_cfg import Solo12PPORunnerCfg


@configclass
class Solo12BoundPPORunnerCfg(Solo12PPORunnerCfg):
    """Conservative PPO settings for the new gait objective."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "solo12_rsl_rl_bound_runs"
    run_name = "solo12_bound_1p5_v1"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )

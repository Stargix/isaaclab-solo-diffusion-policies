"""Conservative PPO defaults for a bounded residual policy."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ResidualDiffusionPPOCfg(RslRlOnPolicyRunnerCfg):
    clip_actions = 1.0
    num_steps_per_env = 32
    max_iterations = 12_000
    # Preserve checkpoints for fixed evaluation; training return alone is not
    # a valid route-success criterion.
    save_interval = 50
    experiment_name = "solo12_residual_diffusion_rl"
    run_name = "phase_b1_local_residual"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.001,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HierarchicalPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 20_000
    save_interval = 500
    experiment_name = "solo12_hierarchical_diffuseloco"
    run_name = "continuous-command-route-pose-time"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

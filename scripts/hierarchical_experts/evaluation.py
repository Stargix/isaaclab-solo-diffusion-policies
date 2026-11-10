"""Bridge to the existing route evaluator without duplicating its metrics."""

import torch
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.dppo_diffusion_rl.checkpointing import sha256_file, validate_resume_task_config
from .policy import Experts, Hierarchy


class EvaluationAdapter:
    def __init__(self, checkpoint, reference, walk, crouch, device, horizon):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("algorithm") != "hierarchical_expert_ppo":
            raise ValueError("Expected a hierarchy checkpoint.")
        validate_resume_task_config(state, reference["dppo_task_config"])
        self.policy = Hierarchy(GaussianPPOConfig(**state["ppo_config"])).to(device).eval()
        actor = dict(state["actor_state_dict"])
        if actor["command_limits"].ndim == 1:
            actor["command_limits"] = actor["command_limits"][None].expand(2, -1).clone()
        self.policy.load_state_dict(actor)
        self.experts = Experts(walk, crouch).to(device).eval()
        actual = {"walk": sha256_file(walk), "crouch": sha256_file(crouch)}
        if actual != state["expert_sha256"]:
            raise ValueError("Expert hashes differ from the training checkpoint.")
        self.horizon = self.policy.ppo_cfg.exec_horizon
        if horizon != self.horizon:
            raise ValueError("Evaluation must preserve the trained decision interval.")

    @torch.no_grad()
    def predict_action_denormalized(self, proprio, actions, goals, guidance_scale=1.0):
        if guidance_scale != 1.0:
            raise ValueError("Classifier-free guidance does not apply to the hierarchy.")
        normal, categorical = self.policy.distribution(proprio, actions, goals)
        latent = torch.cat((normal.mean, categorical.logits.argmax(-1)[:, None].float()), -1)
        return latent[:, None].expand(-1, self.horizon, -1)

    def executable_chunk(self, tokens, horizon):
        if horizon != self.horizon:
            raise ValueError("Wrong execution horizon.")
        return tokens

    @torch.no_grad()
    def low_level_action(self, env, token):
        skill, command = self.policy.command(token)
        return self.experts.action(env, skill, command)

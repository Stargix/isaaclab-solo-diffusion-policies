"""Bridge to the existing route evaluator without duplicating its metrics."""

import torch
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.dppo_diffusion_rl.checkpointing import sha256_file, validate_resume_task_config
from .policy import Experts, Hierarchy
from .policy_v2 import HierarchyV2


class EvaluationAdapter:
    def __init__(self, checkpoint, reference, walk, crouch, device, horizon,
                 capture_diagnostics=False):
        self.capture_diagnostics = bool(capture_diagnostics)
        self.last_diagnostics = None
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("algorithm") != "hierarchical_expert_ppo":
            if state.get("algorithm") != "hierarchical_expert_ppo_v2":
                raise ValueError("Expected a recognized hierarchy V1/V2 checkpoint.")
            self._init_v2(state, reference, walk, crouch, device, horizon)
            return
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

    def _init_v2(self, state, reference, walk, crouch, device, horizon):
        if state.get("format_version") != 3 or state.get("architecture") != HierarchyV2.architecture:
            raise ValueError("Unrecognized hierarchy V2 checkpoint schema/architecture.")
        validate_resume_task_config(state, reference["dppo_task_config"])
        settings = state.get("run_config", {})
        self.policy = HierarchyV2(execution_steps=settings["execution_steps"],
            initial_std=settings["initial_std"], min_std=settings["min_std"],
            max_std=settings["max_std"], command_range_mode=settings["command_range_mode"]).to(device).eval()
        self.policy.load_state_dict(state["actor_state_dict"])
        self.experts = Experts(walk, crouch).to(device).eval()
        actual = {"walk": sha256_file(walk), "crouch": sha256_file(crouch)}
        if actual != state["expert_sha256"]:
            raise ValueError("Expert hashes differ from the training checkpoint.")
        self.horizon = int(settings["execution_steps"])
        if horizon != self.horizon:
            raise ValueError("Evaluation must preserve the trained decision interval.")
        self.is_v2 = True

    @torch.no_grad()
    def predict_action_denormalized(self, proprio, actions, goals, guidance_scale=1.0):
        if guidance_scale != 1.0:
            raise ValueError("Classifier-free guidance does not apply to the hierarchy.")
        if getattr(self, "is_v2", False):
            if self.capture_diagnostics:
                skill, command_latent, self.last_diagnostics = (
                    self.policy.deterministic_with_diagnostics(proprio, actions, goals)
                )
            else:
                skill, command_latent = self.policy.deterministic(proprio, actions, goals)
                self.last_diagnostics = None
            latent = torch.cat((skill[:, None].float(), command_latent), -1)
        else:
            normal, categorical = self.policy.distribution(proprio, actions, goals)
            latent = torch.cat((normal.mean, categorical.logits.argmax(-1)[:, None].float()), -1)
        return latent[:, None].expand(-1, self.horizon, -1)

    def executable_chunk(self, tokens, horizon):
        if horizon != self.horizon:
            raise ValueError("Wrong execution horizon.")
        return tokens

    @torch.no_grad()
    def low_level_action(self, env, token):
        if getattr(self, "is_v2", False):
            skill = token[:, 0].long()
            command = self.policy.command(skill, token[:, 1:4])
            if self.capture_diagnostics and self.last_diagnostics is not None:
                self.last_diagnostics["skill"] = skill
            return self.experts.action(env, skill, command,
                                       capture_observation=self.capture_diagnostics)
        skill, command = self.policy.command(token)
        return self.experts.action(env, skill, command)

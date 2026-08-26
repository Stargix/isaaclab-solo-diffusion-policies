# Solo12 straight bound expert v2

> Experimental result: `bound_v2_1.pt` remained in curriculum stage 0 and
> converged to an asymmetric, under-speed gait. It is retained as a negative
> result; the corrected design is documented in `BOUND_EXPERT_V3.md`.

## Decision and scope

`solo12-bound-v2` corrects the gait objective without changing
`solo12-bound-v0` or any walk, crouch, diffusion, DPPO, or hierarchical task.
It keeps the same 50-D observation and 12-D action contracts as bound v1, so
the stable v1 policy can initialize v2 while PPO and its optimizer start fresh.

This expert has one capability: a visibly distinct symmetric bound in a straight
line at 0.6--1.5 m/s. Its command distribution fixes `vy=wz=0`. Turns and lateral
motion belong to the walk expert; asking this fast expert to learn them weakened
the left/right symmetry and made drift part of the accepted training distribution.

## Why v1 was not accepted

The v1 checkpoint was stable and tracked forward speed, but a 1.25 m/s rollout
measured about 0.085 m/s body-frame lateral velocity and 0.40 m world lateral
displacement over six seconds. Exact phase/contact agreement was only about
26%, and left/right stance Jaccard was about 0.55--0.60.

The old reward multiplied velocity reward by
`exp(-0.5 * auxiliary_cost)`. Contact schedule was only one weak item in that
aggregate, so PPO could obtain high return with a fast, stable non-bound gait.
Moreover, the old `front_pair_sync` metric counted two simultaneously airborne
feet as synchronized, and history-max contact forces blurred the intended phase.

## V2 reward

The positive bounded reward is

```text
command_score = score_vx * score_vy * score_wz * score_heading
gait_gate      = exp(-0.80 * gait_cost)
stability_gate = exp(-0.50 * stability_cost)
r              = 3.5 * command_score * gait_gate * stability_gate * dt
```

The four command scores have separate tolerances: 0.30 m/s forward, 0.08 m/s
lateral, 0.12 rad/s yaw rate, and 8 degrees absolute heading. The heading term
is needed because body-frame `vy=0,wz=0` alone does not penalize an accumulated
world-frame heading offset.

`gait_cost` contains current-force phase agreement, probabilistic left/right XOR
for each pair, diagonal-contact probability, pair foot-height symmetry, and pair
vertical-velocity symmetry. It is a primary gate rather than a cosmetic penalty.
The schedule creates front/rear alternation; XOR rejects split pairs; the diagonal
term explicitly rejects a trot. `stability_cost` retains slip, unwanted swing
force, roll, a permissive pitch corridor, soft height, smoothness, torque, thigh
contact, and weak vertical-speed terms. Falling still ends future positive reward;
there is no large discontinuous fall penalty.

This structure follows the phase/offset gait representation used by
[Walk These Ways](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf),
while preserving the conservative joint-position/PD and dynamics-randomization
choices supported by [Solo12 locomotion](https://www.nature.com/articles/s41598-023-38259-7).

## Training

Initialize compatible network and normalizer tensors from v1, but deliberately
discard its optimizer and iteration state:

```bash
./isaaclab.sh -p source/scripts/rsl_rl/train.py \
  --task solo12-bound-v2 \
  --checkpoint checkpoints_iri/checkpoints_bound/bound_v1.pt \
  --reuse-mlp \
  --num_envs 4096 \
  --max_iterations 3000 \
  --seed 42 \
  --run-name solo12_bound_straight_v2_seed42 \
  --symmetry-mode none \
  --headless \
  --device cuda:0
```

`--reuse-mlp` is important: a normal resume would restore PPO's v1 optimizer,
whose value function and momentum correspond to the old reward. A from-scratch
v2 run remains a useful later ablation, but is not the lowest-risk first run.

## Acceptance criteria

Select the final checkpoint only after stage 2 and evaluate 0.8, 1.0, 1.25, and
1.5 m/s. Mean reward is now aligned with the objective, but report the components:

- base-contact terminations below 1%;
- mean absolute lateral speed below 0.03 m/s;
- mean absolute heading error below 5 degrees;
- exact desired contact fraction above 0.70;
- bound-pair pattern fraction above 0.70;
- diagonal pattern fraction below 0.05;
- front and rear stance Jaccard above 0.85;
- forward-speed RMSE below 0.12 m/s.

These are initial acceptance thresholds, not facts guaranteed by code. Convergence
and gait quality must be established by the cluster training and rollout traces.

## Interactive playback and collection

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task solo12-bound-v2 --checkpoint checkpoints_iri/checkpoints_bound/bound_v2.pt --num_envs 1 --real-time --command_ui --command 1.25 0.0 0.0 --device cuda:0
```

The v2 UI exposes only `vx`; `vy` and `wz` are shown as fixed zero values. Both
data collectors now sample bound demonstrations only in `vx=[1.0,1.5]` with
`vy=wz=0`. Do not collect v1 or v2 until the acceptance metrics above pass.

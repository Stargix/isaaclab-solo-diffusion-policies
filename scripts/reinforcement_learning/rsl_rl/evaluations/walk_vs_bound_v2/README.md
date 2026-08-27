# Walk versus bound-v2 gait decision

## Controlled comparison

Both deterministic actors were evaluated simultaneously under the same
`solo12-bound-v2` physics, on a plane, at a command of 0.8 m/s. The first 16
environments used `walk_final.pt` with the common 48-D observation; the other
16 used `bound_v2.pt` with the same observation plus its 2-D gait clock. The
metrics cover eight seconds after a two-second warm-up.

| Metric | walk_final | bound_v2 |
|---|---:|---:|
| Survival | 100% | 100% |
| Mean forward speed | 0.768 m/s | 0.803 m/s |
| Forward-speed RMSE | 0.047 m/s | 0.016 m/s |
| Mean absolute lateral speed | 0.0067 m/s | 0.0217 m/s |
| Mean absolute yaw rate | 0.031 rad/s | 0.078 rad/s |
| Trot-pattern fraction | 0.658 | 0.117 |
| Bound-pattern fraction | 0.000 | 0.283 |
| Flight fraction | 0.000 | 0.207 |
| Pitch RMS | 0.068 rad | 0.102 rad |

`walk_final` is a clean diagonal trot: FL/RR contact correlation is 0.82 and
FR/RL correlation is 0.72. `bound_v2` is visibly and quantitatively different,
but it is not a clean bound. Its front and rear left/right correlations are
only about 0.36, and the requested bound topology occurs in only 28% of valid
samples. It is best described as a mixed aerial gait. It also has roughly three
times the lateral drift and 2.5 times the yaw-rate error of walk.

The checkpoint is useful as a negative result or ablation, but it should not be
used as the fast expert for the main project.

## Recommended fast skill: dynamic diagonal trot

The lowest-risk replacement is a high-speed running/flying trot, not pace,
pronk, gallop, or another bound attempt:

- warm-start the complete 48-D `walk_final.pt` actor;
- keep its diagonal coordination and observation contract unchanged;
- fine-tune on a forward-speed curriculum from 0.8 to 1.5 m/s;
- use velocity tracking as the dominant objective;
- add only soft diagonal-pair consistency and touchdown air-time terms;
- keep roll, lateral velocity, yaw drift, torque and action-rate costs;
- do not reward flight directly at every step and do not multiply the entire
  task reward by a gait gate.

The behavioral target is a diagonal trot with a lower duty factor and short
aerial intervals. It is distinct from walk through dynamics rather than an
arbitrary leg topology: `walk_final` has zero flight, whereas the fast skill
should reach 8--25% flight while preserving diagonal support and straightness.

Suggested acceptance criteria at 1.0, 1.25, and 1.5 m/s:

- survival at least 95%;
- speed RMSE below 0.12 m/s;
- mean absolute lateral speed below 0.03 m/s;
- trot-pattern fraction above 0.60;
- flight fraction between 0.08 and 0.25;
- pitch RMS below 0.10 rad.

Pace sacrifices roll stability, pronk adds impacts without useful path
manoeuvrability, crawl is not a fast expert, and gallop introduces more
asymmetry and transition complexity than bound. A dynamic trot retains the
stable learned manifold while creating the speed-versus-manoeuvrability tradeoff
needed by the later path-conditioned diffusion policy.

## Reproduce

```powershell
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py --walk_checkpoint checkpoints/walk_final.pt --bound_checkpoint checkpoints_iri/checkpoints_bound/bound_v2.pt --speed 0.8 --duration_s 8 --warmup_s 2 --num_envs 16 --output_dir scripts/reinforcement_learning/rsl_rl/evaluations/walk_vs_bound_v2 --headless --device cuda:0
```

Outputs:

- `gait_comparison.png`: contact rasters, contact correlations, velocity and
  gait fingerprint;
- `gait_comparison.json`: scalar metrics and contact matrices;
- `gait_timeseries.npz`: raw masked rollout traces.

## Literature basis

- Aractingi et al., [Controlling the Solo12 quadruped robot with deep
  reinforcement learning](https://doi.org/10.1038/s41598-023-38259-7):
  joint-position impedance actions, velocity tracking and physically motivated
  regularization on Solo12.
- Margolis and Agrawal, [Walk These Ways](https://arxiv.org/abs/2212.03238):
  structured gait variation through auxiliary locomotion parameters.
- Shafiee et al., [Viability leads to the emergence of gait
  transitions](https://doi.org/10.1038/s41467-024-47443-w): gait transitions
  can emerge from viability and efficiency objectives rather than requiring an
  arbitrary target topology.
- Tan et al., [Sim-to-Real: Learning Agile Locomotion for Quadruped
  Robots](https://www.roboticsproceedings.org/rss14/p10.pdf): robust trotting
  and galloping can be learned from simple task rewards, with a reference used
  only when a specific gait must be imposed.

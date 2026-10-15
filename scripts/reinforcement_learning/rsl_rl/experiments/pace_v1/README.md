# Pace expert pilot

The optional `solo12-pace-v0` task explores ipsilateral contact coordination
as a possible source of slow demonstrations. It is not part of the reported
WC/WCT training recipe, and reliable acquisition was not established. A high
requested speed alone is not evidence of a pace gait.

The task warm-starts the walk actor while initializing the critic and optimizer
fresh. Its training command range is broader than any subsequently selected
demonstration range. Contact patterns, realized velocity and survival must be
checked together before using this expert to generate data.

With an independently supplied pilot checkpoint, inspect the contact and
velocity traces using:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate_gait_comparison.py --comparison walk_pace --walk_checkpoint checkpoints/walk.pt --pace_checkpoint checkpoints/pace.pt --speed 0.4 --num_envs 16 --duration_s 8 --output_dir output/pace_check --headless
```

Keep the outputs outside versioned source. The main project does not require
this pilot; start with the [running guide](../../../../../docs/project/running.md).

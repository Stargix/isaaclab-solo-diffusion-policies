# Exploratory pace expert

[Main project](../../../../../README.md) · [Scope](../../../../../docs/project/method.md#why-these-choices)

The `solo12-pace-v0` task attempts ipsilateral coordination (FL–RL and FR–RR) from the 48D walk actor. Critic and optimizer initialize fresh; the expert uses existing quadruped augmentation.

Pace is intended as a slow demonstration source, reusing walk for fast motion. Expert training keeps vx in [-1,1] m/s; the eventual data range must come from measured slow-speed performance.

The first seed completed training but did not pass the local acquisition gate for stability and coordination. Pace is exploratory and is not a validated component of the reported path controller.

The task and [comparison evaluator](../../evaluate_gait_comparison.py) remain available. Scheduler scripts, working notes and raw diagnostics are local. Pace demonstrations have not been added to the evaluated imitation prior.

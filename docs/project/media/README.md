# Simulator demo

[Project overview](../../../README.md) · [Running guide](../running.md)

These assets are captured from live closed-loop WC-DPPO playback in Isaac Lab, not an animation generated from the reference path. The robot's motion is produced by the frozen seed-42 actor used in the result illustrations.

![Curved route with two posture changes](route_posture_demo.gif)

## Conditions and interpretation

- A predefined 4 m S-shaped route, anchored to the robot's reset pose.
- Required base height: 0.2932 m, then 0.1705 m between route distances 1.6 and 2.8 m, then 0.2932 m again.
- Requested route-average speed: 0.55 m/s; instantaneous speed is not commanded to stay constant.
- Physics at 200 Hz, control at 50 Hz, four actions executed per diffusion plan.
- Red: reference XY. Blue: measured base XY. Yellow: a display-only guide 0.55 m ahead along the reference, not the complete policy preview.

The GIF is a continuous fixed-camera traverse, resampled to 12.5 frames/s and compressed for GitHub. It ends near the final waypoint; later playback is retained locally. Its playback speed follows simulated time. Recording with rendering enabled is not evidence of wall-clock real-time inference.

The two close-up photos come from a separate seeded playback of the same actor and reference. The on-image measured heights come from the simulator state. Presentation cameras and non-colliding visual overlays do not change policy inputs, rewards or physics.

This is an illustrative episode, not a benchmark success rate, proof of optimal temporal allocation, or real-robot validation. Aggregate evidence and limits are in [results](../results.md).

## Replay the reference

The [route array](demo_route.npy) contains `[x, y, required_height]` points in the robot frame. Supply your local checkpoint; reported weights are not bundled.

```bash
./isaaclab.sh -p scripts/diffusion_policy/play_policy.py \
  --checkpoint checkpoints/policy.pt --num_envs 1 \
  --path_file docs/project/media/demo_route.npy --path_file_frame robot \
  --desired_speed 0.55 --exec_horizon 4 \
  --visualize_path --visualize_preview --visualize_goal \
  --visualize_actual_path --visualize_height --device cuda:0
```

The interactive viewer uses its existing camera and debug overlays, which differ from the fixed presentation camera used here. [Provenance](provenance.json) records the actor SHA-256, reference, screenshot states, asset hashes and trim rule. Raw frames and capture tooling remain local.

# Simulator demo

[Project overview](../../../README.md) · [Running guide](../running.md)

These assets are captured from live closed-loop WC-DPPO playback in Isaac Lab, not an animation generated from the reference path. The robot's motion is produced by the frozen seed-42 actor used in the result illustrations.

![Curved route with two posture changes](route_posture_demo.gif)

## Conditions and interpretation

- A predefined 4 m S-shaped route, anchored to the robot's reset pose.
- Required base height: 0.2932 m, then 0.1705 m between route distances 1.6 and 2.8 m, then 0.2932 m again.
- Requested route-average speed: 0.55 m/s; instantaneous speed is not commanded to stay constant.
- Physics at 200 Hz, control at 50 Hz, four actions executed per diffusion plan.
- Red: the reference XYZ, including required height changes. Ground-to-path guides are hidden for clarity.
- Orange: the upcoming reference segment. Blue: the measured base trajectory in 3D.
- Yellow: the native future target, centred on its reference XYZ point; the short golden arrow shows its yaw.

The GIF is a continuous fixed-camera traverse, resampled to 12.5 frames/s and compressed for GitHub. It ends near the final waypoint; later playback is retained locally. Its playback speed follows simulated time. Recording with rendering enabled is not evidence of wall-clock real-time inference.

The two close-up photos come from the same seeded rollout. The on-image measured heights come from the simulator state. No physics steps are taken while capturing a close-up; the wide camera is restored before recording the next animation frame.

The route, preview, measured trace, sphere and heading arrow use `PathDebugVisualizer` with its optional `clean` preset: shaded, non-colliding lines and no dense vertical guides. Orange replaces red along the upcoming segment instead of drawing over it, avoiding red/orange depth conflicts. Coordinates, preview horizon and target selection are unchanged. The sphere marks a future waypoint, so its height can differ from the robot's current required height; it is not an auxiliary guide added for the screenshots. Policy inputs, rewards and physics are unchanged. The original `classic` preset remains the default.

This is an illustrative episode, not a benchmark success rate, proof of optimal temporal allocation, or real-robot validation. Aggregate evidence and limits are in [results](../results.md).

## Replay the reference

The [route array](demo_route.npy) contains `[x, y, required_height]` points in the robot frame. Supply your local checkpoint; reported weights are not bundled.

```bash
./isaaclab.sh -p scripts/diffusion_policy/play_policy.py \
  --checkpoint checkpoints/policy.pt --num_envs 1 \
  --path_file docs/project/media/demo_route.npy --path_file_frame robot \
  --desired_speed 0.55 --exec_horizon 4 \
  --visualize_path --visualize_preview --visualize_goal \
  --visualize_actual_path --visualization_style clean --device cuda:0
```

The interactive viewer uses its usual robot-following camera; the clean preset is the same as that captured here. Add `--visualize_height` if you want sparse ground-to-path guides. [Provenance](provenance.json) records the actor and rendering sources' SHA-256, reference, screenshot states, asset hashes and trim rule. Raw frames and capture tooling remain local.

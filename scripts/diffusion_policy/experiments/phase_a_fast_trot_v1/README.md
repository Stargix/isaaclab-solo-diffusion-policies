# Phase A fast-trot data gate

Status: **pilot data validated structurally but rejected by the temporal-fidelity gate; training has not been configured or started.**

The post-collection analysis found that the frozen successful A0 artifact was
collected with a 3 s command interval and 10% stop probability, whereas the
fast-trot pilot used the stale 5 s / 8% recipe documented for an earlier
dataset. Do not train from the current merged HDF5. See
[`DATA_REVIEW.md`](DATA_REVIEW.md) for the evidence and corrected recollection
decision.

This iteration extends the frozen, successful walk/crouch Phase-A dataset with one independently collected fast-trot expert. It deliberately does not regenerate walk/crouch, add transition demonstrations, or change diffusion training code. The aim is to preserve the successful A0 experiment and isolate the effect of adding a genuinely faster action manifold.

## Frozen baseline

- Baseline dataset: `scripts/diffusion_policy/data/datasets/walk_crouch_phase_a_waypoint_v3.hdf5`
- SHA-256: `00AEA50B71D7CE3FB69C05996F51AE43F6A56623AC8A3314F7B684AE903E3A3F`
- Contents: 3,345 walk demos + 3,345 crouch demos; 1,000,155 frames per skill.
- Goal reconstructed by the loader: `hindsight_geom_avg12`, 2.0 s / 100 control steps.
- The authoritative A0 model and dataset provenance remains in `../phase_a_a0/manifest.json`.

## Fast-trot collection decision

The collection follows the successful Phase-A protocol: one frozen expert per file, 6 s episodes, 25 startup frames retained, commands held for 5 s, 8% explicit 2 s stops, aligned pre-action observations, and 1,000,000 requested frames. Only the expert capability envelope changes.

- Task: `solo12-flying-trot-v0`
- Expert: `checkpoints_iri/checkpoints_bound/flying_trot.pt`
- Expert SHA-256: `4CE1CC3452044959A35923B24A7A2DF3A1381851F11B35B866EA6863441AF9F9`
- Dataset label: `sprint` (the high-speed locomotion slot; the source gait is fast/flying trot)
- Forward support: 0.75--1.50 m/s
- Arc support: up to 1.20 m/s
- Lateral support: +/-0.12 m/s, with 0.75--1.10 m/s forward motion
- Yaw support: +/-0.30 rad/s
- Reverse: disabled because it is outside this expert's intended capability
- Desired base height: 0.28 m
- Seed: 44
- Physics DR: off, matching how this expert was trained; no new domain shift is introduced at this gate

The executable values are stored in `../../data/phase_a_collection_presets.json` under `sprint_fast_trot`.

## Artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `sprint_fast_trot_phase_a_v1.hdf5` | 348,931,617 | `D369A7E560C04010794AE0D514CA89AA9F4F7B3D37E7A366AB2B12287F0B5622` |
| `walk_crouch_sprint_phase_a_v1.hdf5` | 1,021,092,973 | `CC6BEBE9904B7AF4E961A4025C53FC118D9C94DD42A2E24D4DB49F46CAD133B2` |

The HDF5 files are intentionally ignored by Git. `manifest.json`, the audit JSON files, and plots are the versioned provenance.

## Data audit result

All structural gates pass:

- Fast-trot: 3,345 saved demos, 1,000,155 frames, 665,655 valid 2 s windows.
- Collection survival: 99.940% (3,345 saved / 3,347 attempted; one simulator discard and one guardrail discard).
- Merge: exactly 1,000,155 frames per skill; balance ratio 1.0; skill table `[walk, crouch, sprint]`.
- No NaN/Inf, length mismatch, mixed-skill demo, malformed termination, or schema error was detected.
- Duplicate hindsight waypoints: 0.02%; exact-stop windows below 0.02 m/s: 1.842%.

Key hindsight statistics at the exact conditioning horizon:

| Skill | speed p50 | speed p95 | terminal height p50 | action delta L2 p50 |
|---|---:|---:|---:|---:|
| crouch | 0.240 m/s | 0.411 m/s | 0.158 m | 0.488 |
| walk | 0.346 m/s | 0.621 m/s | 0.299 m | 0.519 |
| fast-trot / sprint | 0.976 m/s | 1.372 m/s | 0.285 m | 1.381 |

Interpretation:

1. The added data genuinely extends the achieved path-speed range instead of merely relabeling walk. Command-to-achieved tracking is close to identity in the moving regime.
2. Crouch remains cleanly identifiable by height. Walk and fast-trot have nearby body heights, but they are separated mainly by achieved speed and by their action/state manifolds.
3. Fast-trot action deltas are about 2.7 times the walk median. This is expected for its airborne, higher-frequency gait and is also the main optimization risk.
4. There is useful overlap near the walk/fast boundary rather than a hard disconnected condition space. Because policy observations include state history, equal geometric goals are not conditionally identical: the current gait phase is observable.
5. No explicit walk-to-fast-trot transitions are present. This intentionally mirrors the successful A0 single-expert protocol. The first training trial should test whether diffusion composes them; transition data should only be added if evaluation shows switching hysteresis or instability.

Plots:

- [`raw_multiskill_support.png`](../../data/audits/fast_trot_v1/merged_final/raw_multiskill_support.png)
- [`hindsight_multiskill_support.png`](../../data/audits/fast_trot_v1/merged_final/hindsight_multiskill_support.png)
- [`hindsight_geom_avg12_coverage.png`](../../data/coverage/walk_crouch_sprint_fast_trot_v1/hindsight_geom_avg12_coverage.png)

## Reproduction commands

Run from the repository root in the IsaacLab conda environment. Commands are kept as single lines so they work in PowerShell.

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/data/collect_data.py --mode single --task solo12-flying-trot-v0 --checkpoint checkpoints_iri/checkpoints_bound/flying_trot.pt --skill_name sprint --desired_base_height 0.28 --route_profile phase_a --include_warmup_frames --startup_hold_steps 25 --command_resample_time_s 5.0 --phase_a_forward_speed_min 0.75 --phase_a_forward_speed_max 1.50 --phase_a_arc_forward_speed_max 1.20 --phase_a_reverse_speed_abs_max 0.0 --phase_a_lateral_speed_abs_max 0.12 --phase_a_lateral_forward_speed_min 0.75 --phase_a_lateral_forward_speed_max 1.10 --phase_a_yaw_rate_abs_max 0.30 --phase_a_stop_probability 0.08 --phase_a_stop_hold_s 2.0 --num_envs 1024 --num_steps 1000000 --physics_dr_mode off --seed 44 --output_name sprint_fast_trot_phase_a_v1.hdf5 --headless
```

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/merge_datasets.py --inputs scripts/diffusion_policy/data/datasets/walk_crouch_phase_a_waypoint_v3.hdf5 scripts/diffusion_policy/data/datasets/sprint_fast_trot_phase_a_v1.hdf5 --output scripts/diffusion_policy/data/datasets/walk_crouch_sprint_phase_a_v1.hdf5 --shuffle_seed 46
```

```powershell
conda run --no-capture-output -n env_isaaclab python scripts/diffusion_policy/data/audit_multiskill_hindsight.py --dataset scripts/diffusion_policy/data/datasets/walk_crouch_sprint_phase_a_v1.hdf5 --output_dir scripts/diffusion_policy/data/audits/fast_trot_v1/merged_final --expected_skills walk crouch sprint --max_frames_per_skill 30000 --max_windows_per_skill 20000 --seed 46
```

## Stop gate

No diffusion model, dataloader, optimizer, or training configuration has been modified in this iteration. Before training, decide whether the observed action-frequency gap is acceptable for a direct A1 trial or whether an explicit transition dataset is required. The least confounded next experiment is one direct A1 trial first; transition demonstrations remain a targeted fallback, not an automatic addition.

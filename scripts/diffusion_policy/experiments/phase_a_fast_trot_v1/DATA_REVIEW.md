# Fast-trot dataset review before diffusion training

Date: 2026-08-27

Decision: **do not train the final diffusion policy from the current pilot merge.**

The fast-trot demonstrations have a coherent and distinct action manifold.
The rejection is not caused by the PCA shape or by corrupted actions. It is
caused by a temporal-protocol mismatch discovered when the generated HDF5 was
compared with the actual frozen A0 artifacts.

## Sources of truth

The frozen successful input is:

- `walk_crouch_phase_a_waypoint_v3.hdf5`
- SHA-256: `00AEA50B71D7CE3FB69C05996F51AE43F6A56623AC8A3314F7B684AE903E3A3F`

Its source files contain the authoritative collection metadata:

| Source | Seed | Command interval | Stop probability |
|---|---:|---:|---:|
| `walk_phase_a_waypoint_v3.hdf5` | 44 | 3.0 s | 0.10 |
| `crouch_phase_a_waypoint_v3.hdf5` | 45 | 3.0 s | 0.10 |

The older `balanced_slow_v2` recipe documented 5.0 s, 0.08 and seeds 42/43.
Those files were not the sources merged into the successful frozen A0 dataset.
The HDF5 metadata and frozen hash take precedence over stale prose or presets.

The current fast-trot pilot used 5.0 s, 0.08 and seed 44. It therefore verifies
the expert and command envelope but is not a faithful third-skill extension of
A0.

## Why terminal XY is not circular

Solo12 is mechanically symmetric, but the desired fast skill is not
omnidirectional. It is intentionally forward-dominant with limited lateral and
yaw authority. Mechanical symmetry requires approximately equal left/right
support, not a circular distribution of endpoints.

Across all 665,655 valid fast-trot hindsight windows:

- terminal-y mean: -0.0025 m;
- terminal-y p05: -0.642 m;
- terminal-y p95: +0.641 m;
- fraction with terminal x below zero: 0.755%;
- fraction with absolute terminal y above 1 m: 0.0126%.

The bilateral statistics are effectively symmetric. The forward wedge and
left/right lobes result from straight, arc and limited-lateral command
families over a 2 s local horizon. Arbitrary global routes are composed from
these local receding-horizon segments; the fast expert is not expected to
cover fast backwards or sideways motion.

## Exact outlier audit

Only 84 of 665,655 fast-trot windows have `|terminal_y| > 1 m`, distributed
across six demonstrations. Four are compatible with commanded high-curvature
arcs. Two demonstrations are physical excursions rather than useful expert
behavior:

- `demo_2365`: maximum terminal-y magnitude 1.822 m, near-straight yaw command
  -0.044 rad/s, actual yaw change -1.075 rad over 2 s, minimum height 0.208 m,
  maximum projected-gravity XY norm 0.625 and action delta 4.73.
- `demo_3331`: maximum terminal-y magnitude 1.289 m, maximum
  projected-gravity XY norm 0.713 and action delta 4.84.

`||projected_gravity_xy|| > 0.5` corresponds to body tilt above approximately
30 degrees and selects exactly these two demos. The recollection should use
this physical guardrail rather than manually deleting named examples.

## Why the PCA has two hollow wings

The PCA is computed from raw 12-dimensional expert actions using one global
mean for all skills. Its first two components explain 69.9% of total action
variance. Approximately 51.5% of the total variance is attributable to the
different per-skill mean postures, which is why crouch appears as a separate
island.

Per-skill cumulative variance explained by two PCs:

| Skill | First two PCs |
|---|---:|
| walk | 54.6% |
| crouch | 46.8% |
| fast-trot | 75.6% |

Fast-trot is especially low-dimensional. Its actions are governed mainly by
gait phase and commanded speed, with smaller steering and stabilization
corrections. The two diagonal support phases trace two mirror-related curved
surfaces in action space. Higher speeds expand the curves; lower speeds remain
on their inner edges.

The empty interiors are expected. They contain interpolations between gait
phases that the expert never executes and which need not correspond to
dynamically valid joint coordination. Filling the triangles artificially
would reduce demonstration quality. Representing a non-convex, multimodal
action support without averaging through its empty interior is one motivation
for using a diffusion policy.

Diagnostic plot:

- [`sprint_pca_regimes.png`](../../data/audits/fast_trot_v1/sprint_final/sprint_pca_regimes.png)

## What accumulates at zero

There is no accumulation of zero actions:

- exact zero-action fraction for walk, crouch and fast-trot: 0%;
- fast-trot fraction with any action component above absolute 5: 0.034%;
- no clipping-boundary mass was observed.

The visible histogram spike is the raw expert command `[vx, vy, wz] = 0`, not
the action:

| Skill | All-zero command frames |
|---|---:|
| walk A0 | 16.1% |
| crouch A0 | 15.3% |
| fast-trot pilot | 12.3% |

Every episode begins with 26 recorded zero-command frames so starts from rest
are represented. Explicit held stops create the remaining mass. Because a
probability atom at exactly zero falls into one narrow density-histogram bin,
its plotted height is much larger than its probability mass.

`command_speed` is stored for provenance and expert analysis but is not part
of the diffusion proprioceptive input. The model receives joint position,
joint velocity, base angular velocity, projected gravity, action history and
the geometric hindsight goal. Consequently it cannot exploit an exact raw
command zero as a direct skill shortcut.

Only 1.842% of sampled training goals have hindsight average speed below
0.02 m/s. The training condition distribution is therefore not dominated by
stationary goals.

## Zero-command dynamics

Fast-trot frames separate cleanly into startup, later stop/deceleration and
movement:

| Regime | Frames | Planar speed p50 | Action delta p50 | Fraction below 0.05 m/s |
|---|---:|---:|---:|---:|
| startup zero | 86,970 | 0.012 m/s | 0.574 | 98.3% |
| later zero | 35,875 | 0.041 m/s | 0.307 | 58.5% |
| moving | 877,310 | 1.000 m/s | 1.448 | 0.25% |

An action vector need not be zero for the body to stand still: the legs must
hold posture and settle after reset. Later zero-command frames also include
the physically important deceleration from the airborne gait. These regimes
occupy the inner PCA bands while the fast moving limit cycle forms the outer
wings.

## Training-data verdict

Positive findings:

- the expert tracks its moving command range;
- fast-trot is visibly and quantitatively distinct from walk;
- its action support is structured rather than noisy;
- left/right geometric support is symmetric;
- there is no action-zero or clipping pathology;
- walk p95 hindsight speed (0.621 m/s) and fast-trot p05 (0.613 m/s) provide a
  small but real boundary overlap.

Blocking findings:

- the pilot temporal distribution does not match frozen A0;
- it contains fewer within-episode command changes and decelerations;
- its stop mass differs systematically from walk/crouch;
- seed 44 duplicates the walk collection seed;
- two physically unstable episodes passed the old >53-degree fall guardrail.

Thus the pilot is useful evidence but not the final training dataset.

## Corrected recollection decision

Keep the validated fast-trot capability envelope and change only the values
required for fidelity and quality:

- command interval: 3.0 s;
- stop probability: 0.10;
- stop hold: 2.0 s;
- startup hold: 25 simulator steps, producing 26 recorded zero frames under
  the aligned convention;
- seed: 46;
- body-tilt guardrail: 30 degrees, implemented as
  `--fall_gravity_z -0.866`;
- forward, lateral, yaw, height and physics-DR settings: unchanged from the
  pilot;
- requested saved frames: 1,000,000.

After recollection, rerun the exact terminal audit, compare zero-command mass
and PCA regimes, merge with the frozen A0 HDF5 without regenerating walk or
crouch, and stop again before changing any diffusion trainer configuration.

PowerShell recollection command (single line):

```powershell
.\isaaclab.bat -p scripts/diffusion_policy/data/collect_data.py --mode single --task solo12-flying-trot-v0 --checkpoint checkpoints_iri/checkpoints_bound/flying_trot.pt --skill_name sprint --desired_base_height 0.28 --route_profile phase_a --include_warmup_frames --startup_hold_steps 25 --command_resample_time_s 3.0 --phase_a_forward_speed_min 0.75 --phase_a_forward_speed_max 1.50 --phase_a_arc_forward_speed_max 1.20 --phase_a_reverse_speed_abs_max 0.0 --phase_a_lateral_speed_abs_max 0.12 --phase_a_lateral_forward_speed_min 0.75 --phase_a_lateral_forward_speed_max 1.10 --phase_a_yaw_rate_abs_max 0.30 --phase_a_stop_probability 0.10 --phase_a_stop_hold_s 2.0 --fall_gravity_z -0.866 --num_envs 1024 --num_steps 1000000 --physics_dr_mode off --seed 46 --output_name sprint_fast_trot_phase_a_waypoint_v2.hdf5 --headless
```

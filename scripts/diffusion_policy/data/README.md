# Expert Data Collection for Diffusion Policy (Solo12)

This directory collects expert demonstrations from trained RSL-RL policies to train
Imitation Learning policies (Diffusion Policy). It supports:

1. **Single Mode (Dataset A):** one expert policy per run (walk, crouch, jump, ...).
2. **Chained Mode (Dataset B):** several experts alternated in real time to capture
   physically-realistic skill transitions (walk <-> crouch <-> jump).
3. **Merge utility:** combine several raw HDF5 files into one training dataset, with
   correct `skill_idx` remapping onto a unified `skill_names` table.

---

## 0. Collection convention (IMPORTANT for the DataLoader)

`collect_data.py` records **aligned** `(obs_t, a_t)` pairs: `obs[t]` is the
proprioceptive state of the robot *before* executing `actions[t]`, and
`last_action[t] = actions[t-1]` (with `last_action[0] = 0`). This makes the
proposed `LocomotionHindsightDataset` reconstruction (`last_act[1:] = acts[:-1]`)
correct by construction. The convention is also stored as an HDF5 attribute
(`data.attrs["convention"]`).

Other collection defaults (per the approved master plan
`context/project_decisions_and_roadmap2.md`):

| Setting | Value | Rationale |
|:---|:---|:---|
| Domain Randomization | **Light ON by default** (masses, frictions, inertias, CoM with narrowed ranges) | Keeps robustness-oriented variability without killing most fixed-checkpoint expert rollouts |
| Observation corruption / actuation delay / base pushes | OFF | Clean obs, no unanticipatable perturbations |
| `transition_blend_steps` | **0 (abrupt)** | Lets the simulator resolve the transition physically; survival filtering keeps only successful episodes. Blending is still available as an option. |
| `command_resample_time_s` | **2.0 s** | More velocity-transition diversity inside each episode |
| `warmup_steps` | **25 (0.5 s)** | Discards the settling transient after every reset |
| Crouch command ranges | vx ±0.6, vy ±0.2, wz ±0.6 | Wider coverage of the crouch policy's stable envelope |

### HDF5 schema (per demo, under `data/demo_<k>`)

```
data/demo_<k>/
    obs/
        joint_pos          (T, 12) float32
        joint_vel          (T, 12) float32
        base_ang_vel       (T, 3)  float32
        projected_gravity  (T, 3)  float32
        last_action        (T, 12) float32   -- actions[t-1], zeros at t0
        root_pos_w         (T, 3)  float32   -- viz / hindsight only
        root_quat_w        (T, 4)  float32   -- viz / hindsight only (WXYZ)
        command_speed      (T, 3)  float32   -- commanded (vx, vy, wz)
    actions                (T, 12) float32   -- expert action executed at step t
    dones                  (T,)    bool      -- True only on the last step
    skill_idx              (T,)    int8      -- index into data.attrs["skill_names"]
    attrs: num_samples, skills_sequence
data.attrs: skill_names, convention, control_rate_hz
```

> **Note on existing datasets:** the HDF5 files in `datasets/` collected with the
> previous version of the code use the *post-step* convention, have no
> `last_action` / `skill_idx`, and were collected with DR off. **Regenerate them**
> with the commands below before training the Diffusion Policy.

---

## 1. Data Collection Commands

Run from the `isaaclab-solo` root directory with the Isaac Lab environment active.
`--num_steps` is the number of timesteps **saved** (after warmup and survival
filtering), so the simulator will run somewhat longer to reach the target.

### Single Mode (Dataset A)
```bash
# Walk Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" \
    --checkpoint checkpoints/walk_safe.pt --num_envs 128 --num_steps 1500000 \
    --output_name walk_raw.hdf5 --headless

# Crouch Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" \
    --checkpoint checkpoints/crouch_exponential.pt --num_envs 128 --num_steps 1500000 \
    --output_name crouch_raw.hdf5 --headless

# Jump Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" \
    --checkpoint checkpoints/jumpy_safe.pt --num_envs 128 --num_steps 1500000 \
    --output_name jump_raw.hdf5 --headless
```

### Chained Mode (Dataset B)
```bash
# A. Abrupt Transitions (No Blending) -- RECOMMENDED (physically real, per master plan)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" \
    --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt \
    --num_envs 512 --num_steps 1500000 --output_name chained_abrupt_raw.hdf5 \
    --transition_blend_steps 0 --headless

# B. Cosine Blending (optional, for ablation/experimentation only)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" \
    --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt \
    --num_envs 512 --num_steps 1500000 --output_name chained_blend_raw.hdf5 \
    --transition_blend_steps 8 --transition_blend_type cosine --headless

# C. Linear Blending (optional)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" \
    --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt \
    --num_envs 512 --num_steps 1500000 --output_name chained_linear_raw.hdf5 \
    --transition_blend_steps 8 --transition_blend_type linear --headless
```

### Useful optional flags
```text
--warmup_steps 25               # skip first N steps of every episode (settling transient)
--command_resample_time_s 2.0   # in-episode speed resampling period
--min_demo_len 100              # discard episodes shorter than this
--fall_gravity_z -0.6           # active-fall guardrail: base tilt threshold
--fall_height 0.11              # active-fall guardrail: base height threshold
--physics_dr_mode light         # off | light | full (default: light)
--disable_physics_dr            # alias/diagnostic shortcut for --physics_dr_mode off
```

---

## 2. Merging Datasets

`merge_datasets.py` concatenates several raw HDF5 files into one training dataset,
re-indexing demos (`demo_0`, `demo_1`, ...) and **remapping `skill_idx`** onto a
unified `skill_names` table (without this, `skill_idx=0` would mean "walk" in one
file and "crouch" in another after merging).

```bash
# Dataset A: walk + crouch combined (baseline)
python scripts/diffusion_policy/data/merge_datasets.py \
    --inputs scripts/diffusion_policy/data/datasets/walk_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/crouch_raw.hdf5 \
    --output scripts/diffusion_policy/data/datasets/combined_A.hdf5

# Dataset A + B: pure skills + abrupt transitions (final model)
python scripts/diffusion_policy/data/merge_datasets.py \
    --inputs scripts/diffusion_policy/data/datasets/walk_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/crouch_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/chained_abrupt_raw.hdf5 \
    --output scripts/diffusion_policy/data/datasets/combined_AB.hdf5
```

---

## 3. Dataset Quality & Transition Analysis

`compare_datasets.py` evaluates transition smoothness, control continuity, and
physical stress across datasets, saving a 3x2 grid plot to
`scripts/diffusion_policy/data/plots/dataset_transition_comparison.png`.

```bash
python scripts/diffusion_policy/data/compare_datasets.py \
  --datasets scripts/diffusion_policy/data/datasets/chained_abrupt_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/chained_blend_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/chained_linear_raw.hdf5 \
  --labels "Abrupto" "Cosine 8" "Linear 8"
```

### Historical metrics (raw transition-step metrics, pre-DR datasets)

The table below was measured on the previous-generation datasets (DR off) and
reflects *raw* transition-step metrics only. It is kept for reference.

| Transition Metric | Abrupt (No Blend) | Linear (8 steps) | Cosine (8 steps) |
| :--- | :---: | :---: | :---: |
| **Mean Action Jump** ($\|a_t - a_{t-1}\|_2$) | 1.7044 | 0.7149 | **0.6847** |
| **Max Action Jump** ($\|a_t - a_{t-1}\|_2$) | 6.0764 | 4.2535 | **4.0723** |
| **Mean Joint Velocity Jump** ($\|v_t - v_{t-1}\|_2$) | 9.1021 rad/s | 5.4622 rad/s | **5.2133 rad/s** |
| **Mean Base Tilt** ($\theta$) | 2.27° | 2.35° | 2.37° |
| **Max Base Tilt** ($\theta$) | 15.02° | 14.45° | **11.94°** |

### Design decision: abrupt is the approved default

Although blending yields smoother *raw* action/joint-velocity jumps at the
transition boundary, the approved master plan (`context/project_decisions_and_roadmap2.md`
§3.3 and `context/transitions_and_conditioning_analysis.md` §1) chooses **abrupt
transitions** because:

1. Blending mixes two policies' actions **outside** the physics loop, producing
   commands the robot never physically executed. A Diffusion Policy trained on
   blended data would model non-physical action sequences.
2. Abrupt switching lets the simulator resolve the transition under real physics;
   the **survival filtering** in `collect_data.py` then keeps only episodes that
   stayed stable through the switch, yielding a dataset of physically-realistic
   transitions for the DP to imitate.
3. Combined with the **`v_mean_required` hindsight conditioning**, this is what
   enables the emergent walk <-> crouch transition that is the central goal of the TFG.

Blending is preserved as an optional flag for ablation studies.

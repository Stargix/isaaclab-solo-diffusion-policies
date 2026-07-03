# Skill Chaining & Data Blending for Locomotion Policies

This directory contains scripts for expert trajectory data collection and analysis to train Imitation Learning policies (e.g., Diffusion Policy).

It supports:
1. **Single Mode (Dataset A):** Data collected from a single active expert policy.
2. **Chained Mode (Dataset B):** Data collected by dynamically switching (chaining) between multiple expert policies (walk, crouch, jump) in real time.

---

## 1. Data Collection Commands

Run these commands from the `isaaclab-solo` root directory with the environment active:

### Single Mode (Dataset A)
```bash
# Walk Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" --checkpoint "checkpoints/walk_safe.pt" --num_envs 128 --num_steps 1500000 --output_name walk_raw.hdf5 --headless

# Crouch Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" --checkpoint "checkpoints/crouch_exponential.pt" --num_envs 128 --num_steps 1500000 --output_name crouch_raw.hdf5 --headless

# Jump Policy
python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-v0" --checkpoint "checkpoints/jumpy_safe.pt" --num_envs 128 --num_steps 1500000 --output_name jump_raw.hdf5 --headless
```

### Chained Mode (Dataset B)
These commands collect transitions between walking, crouching, and jumping. They use different blending methods during skill transitions:

```bash
# A. Abrupt Transitions (No Blending)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt --num_envs 512 --num_steps 1500000 --output_name chained_abrupt_raw.hdf5 --transition_blend_steps 0 --headless

# B. Smooth Cosine Blending (8 Steps - Recommended)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt --num_envs 512 --num_steps 1500000 --output_name chained_blend_raw.hdf5 --transition_blend_steps 8 --transition_blend_type cosine --headless

# C. Linear Blending (8 Steps)
python scripts/diffusion_policy/data/collect_data.py --mode chained --task="solo12-v0" --checkpoints checkpoints/walk_safe.pt checkpoints/crouch_exponential.pt checkpoints/jumpy_safe.pt --num_envs 512 --num_steps 1500000 --output_name chained_linear_raw.hdf5 --transition_blend_steps 8 --transition_blend_type linear --headless
```

---

## 2. Dataset Quality & Transition Analysis

To evaluate the transition smoothness, control continuity, and physical stress on the robot chasis across datasets, run the comparison tool:

```bash
python scripts/diffusion_policy/data/compare_datasets.py \
  --datasets scripts/diffusion_policy/data/datasets/chained_abrupt_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/chained_blend_raw.hdf5 \
             scripts/diffusion_policy/data/datasets/chained_linear_raw.hdf5 \
  --labels "Abrupto" "Cosine 8" "Linear 8"
```
This tool extracts metrics around all transition events and generates a 3x2 grid plot saved to `scripts/diffusion_policy/data/plots/dataset_transition_comparison.png`.

### Quantitative Metrics Comparison

The following table summarizes the physical and control metrics evaluated exactly at the step of policy transition (based on 1.5M step datasets):

| Transition Metric | Abrupt (No Blend) | Linear Blending (8 steps) | Cosine Blending (8 steps) |
| :--- | :---: | :---: | :---: |
| **Mean Action Jump ($\|a_t - a_{t-1}\|_2$)** | 1.7044 | 0.7149 | **0.6847** *(~60% smoother)* |
| **Max Action Jump ($\|a_t - a_{t-1}\|_2$)** | 6.0764 | 4.2535 | **4.0723** *(~33% lower peak)* |
| **Mean Joint Velocity Jump ($\|v_t - v_{t-1}\|_2$)** | 9.1021 rad/s | 5.4622 rad/s | **5.2133 rad/s** *(~43% lower shock)* |
| **Mean Base Tilt ($\theta$)** | 2.27° | 2.35° | 2.37° |
| **Max Base Tilt ($\theta$)** | 15.02° | 14.45° | **11.94°** *(~20% more stable)* |

*Note: Global dataset-wide means are almost identical (~11.4k rad/s³ mean joint jerk) because transitions represent less than 5% of the total dataset length. Boundary transition analysis is the standard method to evaluate alignment quality.*

### Key Insights:
1. **Cosine vs. Linear:** The cosine curve has zero derivative boundaries ($\dot{\alpha}(0) = \dot{\alpha}(T) = 0$). This results in smoother initial transitions, reducing the chasis stumble (Max Base Tilt drops to **11.94°** vs **14.45°** on Linear).
2. **Physical Feasibility:** Naive abrupt chaining spikes joint velocities to **9.1 rad/s** in a single control frame ($20\text{ms}$), inducing violent simulated motor torque spikes. Cosine blending reduces this shock by **43%**, preventing hardware-damaging control signals.
3. **Training Benefit:** Using Cosine Blended datasets prevents Imitation Learning policies (like Diffusion Policy) from modeling unphysical discontinuities, which avoids jittery locomotion or failures due to out-of-distribution states at policy transition boundaries.

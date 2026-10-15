# Physical transfer to MuJoCo

The frozen WC-v3 actor controls a MuJoCo model converted from the actual Isaac
USD asset. Closed-loop physics creates each next observation; replaying saved
observations is a separate runtime check. No policy retraining is performed.

The model gate checks named joint frames, static FK, masses and inertia
eigenvalues against native PhysX. Explicit MuJoCo saturated PD and PhysX
implicit actuation are not physically identical. Engine-specific contacts and
diffusion noise batches also bound comparison claims.

## Reproduction order

1. Use `audit_usd` in IsaacLab to resolve body/joint properties of
   `source/isaaclab_assets/data/Robots/Solo12/SoloFlat.usd`.
2. Convert with `convert`, capture native FK with `capture_isaac_fk`, then
   check it with `verify_model`. Do not substitute an unvalidated robot model.
3. Generate a predeclared development bank with `scenarios`.
4. Run the actor with `run`, then compare native/MuJoCo outcomes with `compare`.
5. Use `deployment_suite`, `deployment_validation` and `deployment_analysis`
   for bounded parameter/impulse sensitivity experiments.

Each module supports `--help`. Supply the checkpoint separately and install
MuJoCo 3.3.7 in the environment or an isolated dependency directory. The model
and its matching validation JSON must be supplied together. Outputs are kept
outside versioned source.

The nominal reference uses policy/physics periods 20/5 ms, action scale 0.25,
Kp 9, Kd 0.2, torque saturation 2.65 Nm and the named FL/FR/RL/RR joint order.
These are simulator parameters, not hardware-qualified limits.

First-arrival task success, cross-track adherence and complete-horizon survival
are distinct outcomes. Failures remain in rate denominators. Tracking averages
describe valid prefixes, so failure truncation can lower an apparent mean error.
Successful simulated transfer does not establish target-computer timing or
real-robot safety. See the [validation protocol](../../../docs/project/validation.md)
for the final route evaluation and runtime scope.

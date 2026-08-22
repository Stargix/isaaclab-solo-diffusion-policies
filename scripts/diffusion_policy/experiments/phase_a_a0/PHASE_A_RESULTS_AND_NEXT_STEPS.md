# Phase A result and next decision

## Frozen base result

The canonical `core_envelope` run evaluated checkpoint SHA-256
`7975FE63FDFB5526A8E4613951C6337742CD95A043E1272A2DF98F4CC7CAB553`
from clean evaluator commit `03bcd2597511ebfd671fd395a1300bd86f03316f`.
It covered 135 scenarios: five paths, three requested mean speeds, three
constant heights and three stochastic diffusion samples.

The result is `base_envelope_not_ready`:

- overall survival: 80.7%;
- worst route-speed survival: 33.3% (circle, 0.6 m/s);
- route success: 3.0%;
- mean absolute speed-ratio error: 0.884;
- mean terminal position error: 2.33 m;
- mean cross-track RMSE: 0.106 m;
- endpoint-height causal direction: 45/45 paired groups;
- crouch survival at 0.1705 m: 97.8%;
- intermediate-height survival at 0.23185 m: 93.3%, but height MAE 0.064 m;
- walk survival at 0.2932 m: 51.1%.

The direct tangent-velocity measurements agree with the route-progress metric,
so the speed failure is not a projection artefact. On straight paths, a
0.2 m/s request produces approximately 0.40 m/s in crouch and 0.64-0.67 m/s
at the intermediate/walk heights. Geometry is therefore substantially better
than temporal calibration: the policy follows the route locally while moving
at a skill-dependent nominal speed.

The dataset audit also prevents a simplistic “no slow examples” explanation.
Nearest training goals exist near 0.2, 0.4 and 0.6 m/s for both endpoint
heights. The more plausible hypotheses are weak use of waypoint spacing,
height/skill shortcuts, insufficient density or balance within each joint
condition, and closed-loop covariate shift. These must be distinguished before
choosing an online algorithm.

The random-height challenge is intentionally not executed. It cannot change a
failed constant-height foundation and would violate the protocol's early-stop
rule.

## Next pass: command-controllability diagnosis (no training)

The next computation should be a small intervention test, not RL:

1. Measure achieved tangent speed for requested waypoint speeds below the
   current grid (0.05, 0.10, 0.15, 0.20, 0.40 and 0.60 m/s) at crouch and walk
   heights, first on straight and then on a circle.
2. Report monotonicity, local slope, attainable minimum speed, survival and
   cross-track error separately for each height.
3. Audit dataset density, not only nearest neighbours, over the joint bins of
   achieved speed, height and curvature. Verify whether low-speed walk and
   low-speed turning windows have enough mass to identify the conditioning.
4. Add an explicit zero-progress/stop probe only after the finite-route metric
   supports a zero requested speed without division by zero.

This pass distinguishes three cases:

- **Monotonic and attainable:** keep diffusion frozen and learn or fit a
  low-dimensional command adapter that inverts the speed response and reduces
  speed before curvature. This is the preferred online route.
- **Monotonic but the requested minimum is unattainable:** a frozen high-level
  cannot create a missing slow gait. Rebalance or lightly fine-tune the base on
  existing slow windows before online control.
- **Non-monotonic or pose-mode collapse:** repair conditioning/data balance.
  Joint residual PPO and DPPO are premature because the base does not expose a
  controllable command manifold.

## Online phase, only after the diagnostic gate

If the command manifold is controllable, the first online policy should act in
command space rather than directly on 12 joints. Its observation should contain
route preview in the body frame, scheduled-progress error, tangent-speed error,
cross-track error, height error, body state and the previous correction. Its
bounded action should adjust progress rate / waypoint spacing and optionally a
small yaw-rate or height offset. The diffusion policy remains frozen.

The objective must be evaluated as separate quantities before being combined:

- safety termination and fall rate;
- potential-based path progress relative to the time schedule;
- cross-track error;
- average tangent-speed error over the fixed horizon;
- final position error at the scheduled horizon;
- height tracking;
- correction magnitude and temporal variation.

There must be no positive alive reward. Survival is a constraint/gate, not a
source of return that can make standing still optimal. Progress reward should
be the change in scheduled path potential, while speed and endpoint terms use
finite-horizon errors. Success, timeout and fall statistics remain independent
from episodic return.

Joint residual PPO becomes justified only if the command adapter reaches the
correct temporal/geometric targets but still exhibits localized physical
failures that require foot-level corrections. DPPO is a later alternative when
the action manifold itself must change broadly; it is more expensive and risks
forgetting the useful geometric prior.

## Command-controllability result

The diagnostic was executed from clean commit
`2d29cf2cdc146c74b5a4ec686cead04b8b752881` with 24 straight-path scenarios
(four low proxy speeds, two endpoint heights and three diffusion samples). A
final nine-scenario walk-only floor probe closed the only unresolved boundary.
All 33 scenarios survived the eight-second horizon.

| Height mode | Proxy command [m/s] | Achieved tangent speed [m/s] |
|---|---:|---:|
| crouch | 0.05 | 0.133 +/- 0.044 |
| crouch | 0.10 | 0.315 +/- 0.027 |
| crouch | 0.15 | 0.318 +/- 0.014 |
| crouch | 0.20 | 0.398 +/- 0.029 |
| walk | 0.01 | 0.276 +/- 0.055 |
| walk | 0.025 | 0.232 +/- 0.005 |
| walk | 0.05 | 0.333 +/- 0.008 |
| walk | 0.10 | 0.481 +/- 0.016 |
| walk | 0.15 | 0.506 +/- 0.011 |
| walk | 0.20 | 0.610 +/- 0.009 |

The frozen policy therefore has useful command authority, but its interface is
strongly height-dependent, high-gain and locally non-monotonic near zero. The
0.2 m/s task speed is empirically attainable for both endpoint skills without
changing diffusion weights: crouch crosses it between proxy commands 0.05 and
0.10, while walk reaches 0.232 m/s at proxy command 0.025.

This changes the next decision from base fine-tuning to a command-space online
adapter candidate. The adapter must learn the internal proxy command required
for the external desired speed and reduce it before curvature. It must not be a
joint residual. A height-conditioned lookup/interpolation baseline is required
to show whether RL adds curvature, feedback and transition robustness rather
than merely learning a static inverse calibration.

No further open-loop capability grid is justified before implementing that
adapter. The next evaluation should compare the frozen direct command, the
static calibration baseline and the learned high-level under identical held-out
routes.

## Research interpretation

This result does not reduce the project to a Solo reproduction of DiffuseLoco.
It exposes a concrete research boundary: sparse random demonstrations can yield
surprisingly good geometric and endpoint-pose causal behaviour while failing
to produce a uniformly calibrated, safe temporal command manifold. The central
question becomes whether a small online command-level adapter can correct that
weak conditioning without destroying the generative prior. More complex skills
can be added only after this interface is demonstrated on walk/crouch.

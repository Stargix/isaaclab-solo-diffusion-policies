# Post-selection WC-v3 evaluation

The model and source snapshot were frozen before generating new routes. No
model, reward or route-generator adjustment followed the outcomes.

56 independent geometries (eight per family), three speeds and four profiles
give 672 conditions. The primary bank uses the 32 ID geometries at 0.25 and
0.50 m/s, with constant walk/crouch and both single-transition directions.

| Primary outcome | Result |
|---|---|
| Task and CTE adherence | 252/256 (98.4%) |
| Full-24-s survival | 250/256 (97.7%) |
| Mean active CTE RMSE | 1.97 cm |
| Mean active height MAE | 0.97 cm |
| Arrival-conditional speed MAE | 0.44 cm/s |
| Route-stratified bootstrap interval for adherence | 96.5--100.0% |

At 0.85 m/s, ID upright walk succeeds in 31/32, crouch in 2/32,
walk-to-crouch in 25/32 and crouch-to-walk in 29/32. Geometry-stress results
are separately reported. Constant fast crouch is a demanding condition,
not pooled into the primary supported-task conclusion.

`results.json` contains the verified aggregates; `summary.csv` contains six
speed/bank rows; `routes.npz` contains the immutable numeric bank.
`protocol.json` records input and original execution-source hashes without
machine-specific paths. `episode_source_identity.json` inventories the Python
files checked against the exact frozen execution archive. It distinguishes
unchanged public files from exploratory expert code retained only in the
archive and the removal of its optional task registrations. It is not a claim
that the cleaned public tree is the exact historical execution snapshot.
Later runtime work and a signed-error reporting correction do not
change episodes or task predicates. The original source ZIP and raw outputs
remain local archive assets; weights and datasets are supplied separately.

One selected actor and one inference seed do not replicate training. Reusing a
geometry across speeds/profiles does not create independent routes. Height
scoring retains the original transition exemption, and active averages use
valid prefixes. These results do not establish arbitrary-path generalization,
target-computer latency or real-robot safety.

Reproduction is described in the [validation protocol](../../docs/project/validation.md).

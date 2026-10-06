# Geometry and posture diagnostics

[Running guide](../../../../docs/project/running.md#evaluation) · [Results](../../../../docs/project/results.md)

The stored `route_banks/wct_geometry_sweep_v2.npz` contains 200 independent references: 50 each for S-turn, hard-turn, rounded-turn and compound-turn families. These evaluation references are never used for training.

The sweep pairs geometries across means and native heights. Posture diagnostics vary transition direction and boundary location. Descriptors and route-aligned analysis are in [diagnostic_v2.py](../../../diffusion_policy/evaluation/diagnostic_v2.py).

`prepare_route_banks.py` refuses to overwrite frozen files without an explicit flag. A new bank is a new protocol artifact: retain its seed and checksum.

Scalar-only temporal interventions retain normal preview and change only the final scalar. Older `pace_budget_mode` interventions can also affect preview and are not scalar-only evidence. See [results](../../../../docs/project/results.md#posture-and-temporal-behaviour).

Generated galleries, traces and cluster launchers are local outputs.

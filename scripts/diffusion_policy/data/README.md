# Locomotion demonstrations and hindsight labels

[Method](../../../docs/project/method.md#hindsight-supervision) · [Running guide](../../../docs/project/running.md#offline-imitation)

The evaluated pipeline collects trajectories from velocity-controlled SOLO12 experts and derives path conditions from achieved future motion. Desired-height commands provide posture labels; achieved route length over the relabelling interval provides average-speed labels.

Tools support collection, merging, inspection, support audits and alternative teacher protocols. Alternative teachers are experimental options, not interchangeable with the reported achieved-hindsight recipe.

The recipe is pinned in [the profile16 config](../train/configs/walk_crouch_hindsight_geom_profile16_a0_faithful_k10.json). HDF5 datasets, generated coverage plots and raw audits are local outputs.

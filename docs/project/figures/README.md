# Figure provenance and reading guide

Seven PNGs are copied without changing their data from the manuscript's reviewed figure set of 2026-10-04. They are compact presentation assets, not raw evaluation archives.

| File | Purpose | Interpretation |
|---|---|---|
| `compact_overview.png` | Method diagram | Behaviour labels are data sources, not runtime skill IDs |
| `geometry_pace_maps.png` | Spatial speed allocation | WC seed 42 at mean request 0.8 m/s; normalized route-tangent progress speed |
| `temporal_placement.png` | Temporal adaptation | One route/mean, with crouch boundaries at 0.8/2.0/3.2 m |
| `fig_posture_transitions.png` | Native-height changes | Route-aligned medians and descriptive 10th–90th percentile bands |
| `routes.png` | Ordinary examples | Same references reused across targets and postures |
| `visible_stress_examples.png` | Geometry limits | Paired alternating-turn, sustained-arc and hard-corner examples |
| `compact_support_families.png` | Input/data support | Route draws and demonstration-speed labels, not policy performance |

Result illustrations use the frozen WC seed-42 DPPO actor. Population tables use their stated online seeds. Examples follow declared geometric selection rules without outcome filtering; they are not evidence of universal generalization.

The local source archive retains complete selection records, trace identities and plot-building scripts. The public [manifest](../../../research_artifacts/paper_snapshot/figure_manifest.json) provides image checksums and selected provenance. [Results](../results.md) define success, height exemptions and uncertainty.

These figures belong to this research project. Framework images remain attributed in [README_ISAACLAB.md](../../../README_ISAACLAB.md).

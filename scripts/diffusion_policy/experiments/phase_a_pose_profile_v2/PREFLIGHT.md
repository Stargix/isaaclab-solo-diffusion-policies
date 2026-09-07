# Preflight result

The new loader was run over the complete frozen walk/crouch HDF5 and 50,000
goal windows were sampled with seed 42.

Results:

- 6,690 demonstrations, exactly 3,345 per skill;
- 1,338,000 indexed training windows;
- terminal distance p50/p95: 0.536/1.094 m;
- average achieved speed p50/p95: 0.283/0.555 m/s;
- duplicated-preview fraction: 0.00026;
- non-finite values: none;
- within-window height transitions: 0%;
- future height different from current height: 0%.
- full quadruped-augmented windows: 5,352,000;
- fitted 16-D normalizer: finite, minimum feature std 0.0554;
- all five task-height fields share mean/std 0.2341/0.0614 m, as expected
  for the balanced disjoint experts.

The last two values are expected and scientifically important: Dataset A-WC
contains disjoint constant-height experts. Therefore this run is the strict
zero-shot composition ablation. It can establish whether the diffusion prior
generalizes to unseen mixtures, but it cannot identify the role of each height
slot from supervised variation alone. If constant routes pass and mixed routes
fail, the next and only justified data change is the predeclared size-matched
transition dataset; it is not a reward-tuning problem.

Generated local audit artifacts (ignored by Git) are under:

`scripts/diffusion_policy/data/coverage/walk_crouch_profile16_a_wc_v1/`

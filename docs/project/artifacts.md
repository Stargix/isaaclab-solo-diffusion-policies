# Reproducing the reported controller

The [identity manifest](../../research_artifacts/reproduction/manifest.json)
records expected SHA256 checksums for external weights, datasets and evaluation
files. These large files are not committed to Git. An artifact bundle has been
prepared locally; an external download location has not yet been published.
The manifest is an inventory, not a promise that every file is available.

## Verify before loading

After obtaining and extracting an artifact bundle to `artifacts/`, run:

```bash
python scripts/tools/verify_artifacts.py --root artifacts --profile final_evaluation
```

Verification uses only the Python standard library. It does not load a
checkpoint, import PyTorch, start Isaac Sim or download anything. Exit code 0
means every file in the selected profile matches its pinned identity; 1 means
missing or mismatched files; 2 means invalid input. Obtain the manifest through
a trusted source. A checksum establishes identity, not the safety of an
untrusted pickle checkpoint.

| Profile | Contents | Local preparation status |
|---|---|---|
| `demo` | WC-v3 seed-42 weights and exported configuration/normalizer | Verified |
| `final_evaluation` | Demo, original execution source, frozen routes, protocols and four suites of results | Verified |
| `training_inputs` | Both merged HDF5 datasets and recovered metadata | Verified inputs, not a complete run archive |
| `archived_tables` | Twelve actors/priors, 132 suite CSVs, evaluation metadata and ID/OOD route banks | Verified |
| `comparison_weights` | Nine online actors and three offline priors | Five of twelve weights locally verified |

Profiles are independent selections, not a hierarchy. The final-evaluation ZIP
does not contain the large training datasets or every comparison weight. The
archive of tabular evidence reproduces the recorded measurements, not simulator
rollouts or plots requiring raw time series.

## Run the selected actor

The reported WC-v3 actor is `weights/wc_seed42.pt` in the artifact bundle.
It already contains its own normalizer and configuration. The JSON sidecar is
provided for inspection; it must not replace or refit the embedded normalizer.
After verifying `demo`, use the [visual-demo command](running.md#visual-demo)
with `--checkpoint artifacts/weights/wc_seed42.pt` and four-action execution.

## Reproduce the frozen evaluation

Use `final/original_execution_source.zip` for the historical source, extracted
into a separate directory. Its identity is pinned independently of the current
cleaned public repository. The original protocol and four CSV/JSON outputs are
also included. The [validation procedure](validation.md#new-route-bank)
describes freeze, run and verify. A new output directory is required; never
overwrite the historical result folders.

The historical final run used Python 3.11, Isaac Sim 5.1.0.0, PyTorch
2.7.0+cu128, NumPy 1.26.0, SciPy 1.15.3, Diffusers 0.39.0 and RSL-RL 3.1.2.
These describe that run, not a universal requirement or proof of equivalent
physics in other simulator versions. Earlier cluster experiments used their
own environments and provenance. Runtime measurements are not measurements on
the robot's control computer.

## Reproduce training and comparisons

`datasets/walk_crouch_phase_a_waypoint_v3.hdf5` is the exact WC input.
`datasets/walk_crouch_sprint_phase_a_waypoint_v4.hdf5` is the exact WCT input.
Do not rebuild either merged dataset from a similarly named subset and call it
the same input. Offline checkpoints retain the episode split, normalizer and
training configuration; metadata sidecars expose these without loading weights.
The WCT configuration recovered from its refined actor is inherited metadata,
not a substitute for the missing original WCT imitation checkpoint and split.

The [training recipes](running.md#offline-imitation) explain offline imitation,
DPPO and the matched Gaussian PPO control. Select the deterministic backend
explicitly with `--policy_backend deterministic_chunk`. The three online seeds
share one offline prior per method; they are not three independent repetitions
of the entire offline-to-online pipeline. GPU retraining is not promised to
produce byte-identical weights.

The comparison archive retains exact CSV bytes and exported evaluation
metadata. Absolute machine paths are reduced to basenames in the JSON exports,
nonfinite JSON numbers become null, and each export records the original
summary's checksum. Original checkpoint files and CSV bytes are unchanged.
The evaluation commits recorded there identify evaluation code, not necessarily
the exact training source. The current public tree is a cleaned derivative,
not a byte-identical historical checkout.

## Archive gaps and release boundary

The original WC, WCT and Gaussian seed-43/44 checkpoints and the exact WCT
offline prior still need retrieval. Their expected identities are pinned from
the archived evaluations. A similarly named WCT checkpoint is not interchangeable.
Training metrics and run-time configuration/provenance also need consolidation
before describing the full training archive as complete. Existing evaluation
CSVs alone cannot establish that provenance.

Missing originals must remain marked missing. Do not repair this by exporting
new weights, renormalizing data, changing a selection rule or retraining under
the old run name. Hosting, licensing and a final review of historical archives
for machine-specific information precede external distribution. No private
cluster launcher, credentials, generated report or binary artifact belongs in
the public source repository.

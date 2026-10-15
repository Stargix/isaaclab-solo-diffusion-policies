# Walk/crouch versus walk/crouch/fast-trot demonstrations

WC uses walk and crouch demonstrations. WCT adds fast-trot demonstrations.
Both offline models use profile16 hindsight conditioning, the same architecture,
50 epochs and seed 42. The additional WCT data increase optimizer updates, so
this is a dataset-recipe comparison rather than an isolated causal gait test.

The completed online comparison uses `supported_hybrid_v3` for both priors,
three online seeds (42, 43, 44) and the same nominal interaction budget. It
does not compare WC-v3 against a differently trained WCT-v6. No skill index,
gait reward or contact-pattern objective is added to the task policy.

## Frozen inputs

| Input | SHA256 |
|---|---|
| WC dataset | `00aea50b71d7ce3fb69c05996f51ae43f6a56623ac8a3314f7b684ae903e3a3f` |
| WCT dataset | `4e323bd426ccb6f503e70ee255b397899e0da2f847eb60fa6a21f147f7a6b730` |
| WC offline prior | `b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd` |
| WCT offline prior | `2f28fd4beb963c74293948507eecd5e7061f34a1436d8014626bd60f2d73da54` |
| Selected WC-v3 actor | `2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868` |

Datasets and model weights are supplied separately. The shared offline config
uses history 8, horizon 16, execution offset 8, goal horizon 100 ticks,
transformer 128/4/4, K10 sampling and quadruped data augmentation.

## Online protocol

The shared v3 protocol uses 4096 environments, 150 iterations, 32 rollout
chunks, route-speed maximum 1.0 m/s, remaining-speed cap 1.5 m/s, transition
positions 0.8--3.2 m, path weight 2.0, profile-height weight 0.75, height
tolerance 0.04 m and reference KL coefficient 0. The geometric preview uses
the requested mean, with remaining speed supplied as the final scalar.

See the [running guide](../../../../docs/project/running.md) for direct
training and evaluation commands. Machine-specific SLURM scripts are private.
Existing versions and alternative experiments retain their own configurations.

## Interpretation

WC matches or exceeds WCT on the evaluated fast tasks. WCT is better on some
repeated transitions starting crouched. Contact audits do not establish
reliable reproduction of the fast-trot expert. These results support task
capability acquisition, not demonstrated implicit selection of three gaits.
The [Gaussian comparison](../../../gaussian_chunk_rl/README.md) addresses a
separate question about the complete imitation-and-refinement pipeline.

The [final validation protocol](../../../../docs/project/validation.md) freezes the selected
WC-v3 actor, executable source snapshot and a new post-selection route bank.

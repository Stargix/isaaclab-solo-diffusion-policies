#!/usr/bin/env bash
set -euo pipefail
# Run inside an allocated GPU node with the Isaac Lab environment active.
mode="${1:?Usage: bash scripts/hierarchical_experts/run_cluster.sh smoke|probe|pilot|train|eval [seed] [checkpoint]}"
seed="${2:-42}"
artifact_root="${ARTIFACT_ROOT:-artifacts}"
run_root="${HIERARCHY_RUN_ROOT:-scratch/hierarchy_runs}"
common=(--walk_checkpoint "$artifact_root/experts/walk_final.pt"
        --crouch_checkpoint "$artifact_root/experts/crouch_final.pt")
export PYTHONPATH="$PWD:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
case "$mode" in
  probe)
    ./isaaclab.sh -p scripts/hierarchical_experts/probe.py "${common[@]}" --seed "$seed" \
      --output_dir "$run_root/probe_seed$seed" --steps 1000 --headless --device cuda:0
    ;;
  smoke|pilot|train)
    count=4096; iterations=150; chunks=32; extra=()
    if [[ "$mode" == smoke ]]; then count=4; iterations=1; chunks=310; extra=(--smoke); fi
    if [[ "$mode" == pilot ]]; then count=256; iterations=150; fi
    ./isaaclab.sh -p scripts/hierarchical_experts/train.py "${common[@]}" \
      --reference_checkpoint "$artifact_root/weights/wc_seed42.pt" \
      --output_dir "$run_root/${mode}_seed$seed" --seed "$seed" \
      --num_envs "$count" --iterations "$iterations" --rollout_chunks "$chunks" \
      --command_range_mode configured "${extra[@]}" --headless --device cuda:0
    ;;
  eval)
    actor="${3:?Provide the hierarchy checkpoint as the third argument}"
    ./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py "${common[@]}" \
      --checkpoint "$artifact_root/weights/wc_seed42.pt" --hierarchy_checkpoint "$actor" \
      --path_shapes straight --path_heights 0.1705 0.2932 --speeds 0.25 0.50 \
      --repeats 1 --duration_s 24 --exec_horizon 4 --seed "$seed" \
      --output_dir "$run_root/eval_seed$seed" --require_empty_output_dir \
      --headless --device cuda:0
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

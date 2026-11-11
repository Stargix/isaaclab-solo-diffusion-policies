#!/usr/bin/env bash
set -euo pipefail
# Run inside an allocated GPU node with the Isaac Lab environment active.
mode="${1:?Usage: bash scripts/hierarchical_experts/run_cluster.sh smoke|probe|pilot|train|eval|v2smoke|v2train|v2resume|v2eval [seed] [checkpoint]}"
seed="${2:-42}"
artifact_root="${ARTIFACT_ROOT:-artifacts}"
run_root="${HIERARCHY_RUN_ROOT:-scratch/hierarchy_runs}"
range_mode="${HIERARCHY_COMMAND_RANGE:-configured}"
entropy_coef="${HIERARCHY_CATEGORICAL_ENTROPY:-0.0}"
execution_steps="${HIERARCHY_EXECUTION_STEPS:-4}"
common=(--walk_checkpoint "$artifact_root/experts/walk_final.pt"
        --crouch_checkpoint "$artifact_root/experts/crouch_final.pt")
export PYTHONPATH="$PWD:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
case "$mode" in
  v2smoke|v2train|v2resume)
    count=4096; chunks=16; budget=80000000; epochs=5; actor_batch=8192; critic_batch=4096
    output="$run_root/v2_seed$seed"; extra=()
    if [[ "$mode" == v2smoke ]]; then
      count=4; chunks=130; budget=5200; epochs=1; actor_batch=520; critic_batch=520
      output="$run_root/v2_smoke_seed$seed"; extra=(--smoke --save_interval_updates 1)
    elif [[ "$mode" == v2resume ]]; then
      checkpoint="${3:?Provide a post-update checkpoint as the third argument}"
      output="$(dirname "$checkpoint")"; extra=(--resume_from "$checkpoint")
    fi
    ./isaaclab.sh -p scripts/hierarchical_experts/train_v2.py "${common[@]}" \
      --reference_checkpoint "$artifact_root/weights/wc_seed42.pt" \
      --output_dir "$output" --seed "$seed" --num_envs "$count" \
      --interaction_budget "$budget" --rollout_chunks "$chunks" --execution_steps 10 \
      --update_epochs "$epochs" --minibatch_size "$actor_batch" \
      --critic_minibatch_size "$critic_batch" --command_range_mode shared "${extra[@]}" \
      --headless --device cuda:0
    ;;
  v2eval)
    actor="${3:?Provide the hierarchy V2 checkpoint as the third argument}"
    ./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py "${common[@]}" \
      --checkpoint "$artifact_root/weights/wc_seed42.pt" --hierarchy_checkpoint "$actor" \
      --path_shapes straight --path_heights 0.1705 0.2932 --speeds 0.25 0.50 \
      --repeats 1 --duration_s 24 --route_length_m 4.0 --exec_horizon 10 --seed "$seed" \
      --output_dir "$run_root/v2_eval_seed$seed" --require_empty_output_dir \
      --headless --device cuda:0
    ;;
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
      --command_range_mode "$range_mode" --categorical_entropy_coef "$entropy_coef" --execution_steps "$execution_steps" "${extra[@]}" --headless --device cuda:0
    ;;
  eval)
    actor="${3:?Provide the hierarchy checkpoint as the third argument}"
    ./isaaclab.sh -p scripts/diffusion_policy/evaluate_policy.py "${common[@]}" \
      --checkpoint "$artifact_root/weights/wc_seed42.pt" --hierarchy_checkpoint "$actor" \
      --path_shapes straight --path_heights 0.1705 0.2932 --speeds 0.25 0.50 \
      --repeats 1 --duration_s 24 --exec_horizon "$execution_steps" --seed "$seed" \
      --output_dir "$run_root/eval_seed$seed" --require_empty_output_dir \
      --headless --device cuda:0
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

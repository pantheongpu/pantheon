#!/bin/bash
# Acceptance test of one node: run it after the node is installed, after a
# card is replaced, or before the node goes back into the queue.
#
#   sbatch --nodelist=<node> --gres=gpu:<number of GPUs of the node> burnin.sbatch
#
# The job fails when the verdict is not HEALTHY, so "sacct -j <id>" shows the
# result, and the reports of every card are kept beside the output file.
#
#SBATCH --job-name=pantheon-burnin
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=pantheon-%N-%j.out

# Load what gives you nvcc or hipcc and the pantheon command, for example:
#   module load CUDA PantheonGPU
#   module load cuda && source ~/pantheon-venv/bin/activate

NODE_CHECK=${PANTHEON_NODE_CHECK:-pantheon_node_check.py}
DURATION=${PANTHEON_DURATION:-60}
WORKLOADS=${PANTHEON_WORKLOADS:-galpat march_test memory_read tensor_virus}

tests=()
for workload in $WORKLOADS; do
    tests+=(--test "$workload")
done

# galpat and march_test come first: they load a card the least, so a weak
# power supply shows up before the heavy workloads start.
# PANTHEON_EXTRA_ARGS holds several arguments, so it is split on purpose.
# shellcheck disable=SC2086
exec "$NODE_CHECK" --context job --duration "$DURATION" "${tests[@]}" \
    --report-dir "pantheon-${SLURMD_NODENAME:-node}-${SLURM_JOB_ID:-0}" $PANTHEON_EXTRA_ARGS

#!/bin/bash
# Slurm epilog: test the GPUs of the job that has just ended, and drain the
# node when a card has a fault.
#
#   slurm.conf:   Epilog=/etc/slurm/epilog.d/*
#   install as:   /etc/slurm/epilog.d/50-pantheon.sh
#
# The job that ran last is the best stress test a card gets, so the moment
# after it is a good time to ask if the memory still holds a pattern. The node
# stays in the state "completing" while this runs, about 20 seconds with the
# defaults below once the workloads are compiled.
#
# slurmd runs the epilog as root with a short PATH and an environment of its
# own, so the places of things are named here. To change one, set it in
# /etc/pantheon/epilog.conf, which is read as a shell file when it exists.

CONFIG=${PANTHEON_EPILOG_CONFIG:-/etc/pantheon/epilog.conf}
# shellcheck source=/dev/null
[ -r "$CONFIG" ] && . "$CONFIG"

NODE_CHECK=${PANTHEON_NODE_CHECK:-/usr/local/sbin/pantheon_node_check.py}
WORKLOADS=${PANTHEON_WORKLOADS:-march_test}
DURATION=${PANTHEON_DURATION:-10}
LOG=${PANTHEON_LOG:-/var/log/pantheon-epilog.log}
# nvcc or hipcc, and the pantheon command itself
export PATH=${PANTHEON_PATH:-/usr/local/cuda/bin:/opt/rocm/bin:/usr/local/bin}:$PATH
# compiled workloads are kept here between runs
export PANTHEON_BUILD_CACHE_DIR=${PANTHEON_BUILD_CACHE_DIR:-/var/cache/pantheon}

# A job without a GPU has nothing to test.
[ -n "$SLURM_JOB_GPUS" ] || exit 0

tests=()
for workload in $WORKLOADS; do
    tests+=(--test "$workload")
done

# PANTHEON_EXTRA_ARGS holds several arguments, so it is split on purpose.
# shellcheck disable=SC2086
output=$("$NODE_CHECK" --context epilog --duration "$DURATION" "${tests[@]}" $PANTHEON_EXTRA_ARGS 2>&1)
code=$?
verdict=$(printf '%s\n' "$output" | head -n 1)

{
    printf '%s job=%s gpus=%s exit=%s %s\n' "$(date -Is)" "$SLURM_JOB_ID" "$SLURM_JOB_GPUS" "$code" "$verdict"
    [ "$code" -eq 0 ] || printf '%s\n' "$output" | sed -e '1d' -e 's/^/    /'
} >> "$LOG" 2>/dev/null

if [ "$code" -eq 2 ]; then
    scontrol update NodeName="$SLURMD_NODENAME" State=DRAIN Reason="$(printf '%s' "$verdict" | cut -c1-120)"
fi

# Slurm drains a node when its epilog fails. Only a fault is a reason for
# that, and the lines above have drained the node and said why. A warning, or
# a check that could not run, goes to the log and leaves the node in service.
exit 0

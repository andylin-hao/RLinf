#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    echo "Please provide a config name as the first argument."
    exit 1
else
    CONFIG_NAME=$1
fi

# CI runs these configs back to back on one self-hosted runner. Without an explicit
# teardown, a previous run's Ray cluster and its sglang scheduler/detokenizer
# subprocesses can outlive the step and keep holding ports, while the next run starts a
# fresh PortLockManager that has no record of them -- which shows up as a random
# EADDRINUSE partway through engine startup.
cleanup() {
    status=$?
    ray stop --force >/dev/null 2>&1 || true
    exit $status
}
ray stop --force >/dev/null 2>&1 || true
trap cleanup EXIT

python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path $REPO_PATH/tests/e2e_tests/reasoning/  --config-name $CONFIG_NAME

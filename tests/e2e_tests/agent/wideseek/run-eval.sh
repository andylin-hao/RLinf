#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
# These have segfaulted during interpreter teardown, after the results were
# written, which exits 139 with no diagnostic. Print the stack instead.
export PYTHONFAULTHANDLER=1

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/agent/wideseek_r1/eval.py --config-path ${REPO_PATH}/tests/e2e_tests/agent/wideseek --config-name qwen3-eval

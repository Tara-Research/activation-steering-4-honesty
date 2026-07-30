#!/bin/bash

if [[ "$SLURM_PROCID" -eq "0" ]]; then
        echo "Ray head node already started..."
        sleep 10

else
        export VLLM_HOST_IP=`hostname --ip-address`
        ray start --address "${HEAD_NODE}:${RAY_PORT}" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=4 --block
        sleep 5
        echo "ray worker started!"
fi

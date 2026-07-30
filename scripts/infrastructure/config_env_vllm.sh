#!/bin/bash

module load gcc python/3.11 arrow/24 opencv/4.13 cuda

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

export CUDA_HOME=$EBROOTCUDA
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/scratch/n/niklas/tiktoken_encodings}"

pip install --upgrade pip --no-index
pip install pydantic --no-index
pip install ray tqdm --no-index
pip install accelerate --no-index
pip install sentence_transformers --no-index
pip install sacrebleu --no-index
pip install rouge-score nltk --no-index
pip install lm_eval==0.4.11 --no-index --no-deps
python -m pip install --no-index \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  --no-deps datasets==4.4.1
pip install --no-index transformers==5.5.4
pip install multiprocess xxhash --no-index
pip install vllm==0.20.0 --no-index
pip install triton==3.6.0 --no-index
pip install fire --no-index
pip install deep_gemm==2.1.1 --no-index

echo "Environment setup complete."

echo "=== TORCH/VLLM DIAGNOSTICS ==="
echo "--- module list ---"
module list 2>&1
echo "--- pip versions ---"
pip show torch  | grep -E "^(Name|Version|Location)"
pip show vllm   | grep -E "^(Name|Version|Location)"
pip show triton | grep -E "^(Name|Version|Location)"
python -c "import torch; print('torch:', torch.__version__)" || true
nm -D $(python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib/libc10.so')" 2>/dev/null) 2>/dev/null | grep MessageLogger | head -5
echo "==============================="


deactivate

#!/bin/bash

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --upgrade pip --no-index
pip install pydantic --no-index
pip install ray tqdm --no-index
pip install accelerate --no-index
pip install sentence_transformers --no-index
pip install matplotlib --no-index
pip install sacrebleu --no-index
pip install rouge-score nltk --no-index
pip install lm_eval==0.4.11 --no-index --no-deps
python -m pip install --no-index \
  --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
  --no-deps datasets==4.4.1
pip install multiprocess xxhash --no-index
pip install --no-index transformers==5.5.4
pip install triton==3.6.0 --no-index
pip install fire --no-index
pip install aiohttp --no-index
pip install python-dotenv --no-index
pip install deep_gemm==2.1.1 --no-index
pip install peft==0.19.1 --no-index

echo "Environment setup complete."

deactivate

#!/bin/bash
# Installs vllm + persona_playground pip dependencies into the currently
# active virtualenv.
#
# This is the install-only counterpart of config_env_vllm.sh:
# - config_env_vllm.sh: loads modules, creates the venv, installs deps
#   (used by most pipelines that don't need extra layered deps)
# - install_vllm_deps.sh (this file): assumes modules are loaded and a
#   venv is already active, only runs the pip installs
#
# Used by config_env_auditing.sh, which needs to install auditing-agents
# deps BEFORE the heavy vllm/torch wheels so package resolution doesn't
# end up downgrading vllm-critical packages.

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

echo "vllm dependency install complete."

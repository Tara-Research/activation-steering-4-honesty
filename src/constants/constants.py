import os
import socket
from pathlib import Path

ROOT_FOLDER = Path(__file__).resolve().parent.parent.parent

CONFIG_HONESTY = os.path.join(ROOT_FOLDER, "configs", "config_honesty.yaml")
CONFIG_COMPASSION = os.path.join(ROOT_FOLDER, "configs", "config_compassion.yaml")
CONFIG_LLM_JUDGE = os.path.join(ROOT_FOLDER, "configs", "config_llm_judge.yaml")

DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
RESULTS_FOLDER = os.path.join(ROOT_FOLDER, "results")
PLOTS_FOLDER = os.path.join(ROOT_FOLDER, "plots")

_TAMIA_PERSONA_VECTORS = "/project/aip-irina/shared/steering/persona_vectors"


def _detect_persona_vectors_folder() -> str:
    """Pick the persona-vectors directory for the current cluster.

    Order: explicit env var > Mila > Alliance/tamia > tamia path as last resort.
    """
    override = os.environ.get("PERSONA_VECTORS_FOLDER")
    if override:
        return override

    hostname = socket.gethostname().lower()
    scratch = os.environ.get("SCRATCH", "")

    on_mila = (
        "mila" in hostname
        or scratch.startswith("/network/scratch")
        or os.path.isdir("/network/scratch")
    )
    if on_mila:
        if scratch:
            return os.path.join(scratch, "persona_vectors")
        user = os.environ.get("USER") or os.environ.get("LOGNAME")
        if user:
            return f"/network/scratch/{user[0]}/{user}/persona_vectors"
        raise RuntimeError(
            "Detected Mila cluster but neither $SCRATCH nor $USER is set; "
            "cannot resolve PERSONA_VECTORS_FOLDER. Set PERSONA_VECTORS_FOLDER explicitly to override."
        )

    on_alliance = (
        bool(os.environ.get("CC_CLUSTER"))
        or hostname.startswith("tamia")
        or "alliancecan" in hostname
        or os.path.isdir("/project/aip-irina")
    )
    if on_alliance:
        return _TAMIA_PERSONA_VECTORS

    # Off-cluster (e.g. a local checkout): keep artifacts inside the repo.
    return os.path.join(ROOT_FOLDER, "persona_vectors")


PERSONA_VECTORS_FOLDER = _detect_persona_vectors_folder()

# All steering experiments use logistic-regression vectors; the constants keep
# the historical "logistic_regression" segment in results paths and the "logreg"
# label in result filenames.
VECTOR_TYPE = "logistic_regression"
VECTOR_TYPE_LABEL = "logreg"


def get_config_for_trait(trait: str) -> str:
    _CONFIGS = {
        "honesty": CONFIG_HONESTY,
        "compassion": CONFIG_COMPASSION,
    }
    if trait not in _CONFIGS:
        raise ValueError(
            f"No config for trait={trait!r}. Valid traits: {list(_CONFIGS)}"
        )
    return str(_CONFIGS[trait])


CONFIG_MT_BENCH = os.path.join(ROOT_FOLDER, "configs", "config_mt_bench.yaml")
MT_BENCH_DATA_DIR = os.path.join(DATA_FOLDER, "mt_bench")
MT_BENCH_RESULTS_DIR = os.path.join(RESULTS_FOLDER, "mt_bench")
MT_BENCH_PLOTS_DIR = os.path.join(PLOTS_FOLDER, "mt_bench")

ALPACA_EVAL_RESULTS_DIR = os.path.join(RESULTS_FOLDER, "alpaca_eval")
ALPACA_EVAL_PLOTS_DIR = os.path.join(PLOTS_FOLDER, "alpaca_eval")

SEED = 42

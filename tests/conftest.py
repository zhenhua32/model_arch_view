"""Shared fixtures and helpers for the model-architecture estimation test-suite.

The suite validates ``serve_model_arch.py`` against every model under
``model_configs/``. It has four layers:

  * test_unit_estimation.py   -- white-box math on estimate_llm_metrics/parse_llm_dims/scalar_int
  * test_build_all_models.py  -- every model directory builds a payload without error
  * test_llm_ground_truth.py  -- LLM param counts vs safetensors ``total_size`` ground truth
  * test_golden_metrics.py    -- frozen golden values for a curated set of anchor models

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# Make serve_model_arch importable regardless of the pytest invocation cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_ROOT = PROJECT_ROOT / "model_configs"

# Directory-name tokens that indicate a quantized / non-bf16 checkpoint. For such
# checkpoints the "bytes = 2 * params" proxy does not hold, so they are excluded
# from the strict ground-truth byte comparison.
_QUANT_NAME_RE = re.compile(
    r"(?:^|[-_./])(fp8|fp4|int8|int4|w8a8|w4a16|awq|gptq|gguf|nf4|mxfp4|bnb)(?:$|[-_./])",
    re.IGNORECASE,
)


def all_model_dirs() -> list[Path]:
    """Every immediate sub-directory of model_configs/ (one per model)."""
    if not MODEL_ROOT.exists():
        return []
    return sorted(d for d in MODEL_ROOT.iterdir() if d.is_dir())


def model_ids() -> list[str]:
    return [d.name for d in all_model_dirs()]


def read_total_size(model_dir: Path) -> int | None:
    """Ground-truth parameter byte-count from the safetensors index, if present."""
    import serve_model_arch as s

    idx = model_dir / "model.safetensors.index.json"
    if not idx.exists():
        return None
    data = s.read_json_file(idx) or {}
    meta = data.get("metadata") or {}
    ts = meta.get("total_size")
    return int(ts) if ts else None


def is_quantized(model_dir: Path) -> bool:
    """True when the checkpoint is quantized (config flag OR name token)."""
    import serve_model_arch as s

    cfg = s.primary_config(model_dir)
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict) and qc:
        return True
    dtype = str(cfg.get("torch_dtype") or cfg.get("dtype") or "")
    if re.search(r"fp8|float8|int8|int4|fp4", dtype, re.IGNORECASE):
        return True
    return bool(_QUANT_NAME_RE.search(model_dir.name))


@pytest.fixture(scope="session")
def serve():
    import serve_model_arch as s

    return s


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT

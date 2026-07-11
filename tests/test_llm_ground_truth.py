"""LLM parameter-count accuracy vs the safetensors ``total_size`` ground truth.

For a non-quantized (bf16/fp16) checkpoint the on-disk byte count equals
``2 * parameter_count``. We therefore compare our estimated ``total_params * 2``
against ``metadata.total_size`` from ``model.safetensors.index.json``.

Quantized checkpoints (fp8/int8/fp4/awq/...) are skipped for this byte check
because their bytes-per-parameter ratio is not 2 -- they are still exercised by
the build smoke test and the golden-metric test.
"""
from __future__ import annotations

import pytest

from conftest import all_model_dirs, is_quantized, read_total_size

# Default tolerance for the bf16 byte proxy.
DEFAULT_TOL = 0.05

# Documented, architecture-driven exceptions (still bounded, just looser). The
# reason explains why the checkpoint bytes legitimately differ from 2*params.
KNOWN_EXCEPTIONS = {
    # Embedding model: the generative lm_head is stripped from the checkpoint,
    # so on-disk bytes are smaller than the parameter estimate.
    "Qwen__Qwen3-Embedding-8B": (0.12, "embedding model, lm_head removed from checkpoint"),
    # Checkpoint ships extra MTP / multi-token-prediction layers that are not part
    # of the headline num_hidden_layers count.
    "openpangu__openPangu-2.0-Flash": (0.12, "checkpoint includes extra MTP layers"),
}


def _candidates():
    out = []
    for d in all_model_dirs():
        import serve_model_arch as s

        if s.classify_model_dir(d) != "llm":
            continue
        if is_quantized(d):
            continue
        if read_total_size(d) is None:
            continue
        out.append(d)
    return out


CANDIDATES = _candidates()
CANDIDATE_IDS = [d.name for d in CANDIDATES]


def test_have_ground_truth_models():
    assert CANDIDATES, "expected at least some non-quantized LLMs with total_size"


@pytest.mark.parametrize("model_dir", CANDIDATES, ids=CANDIDATE_IDS)
def test_param_count_matches_checkpoint(serve, model_dir):
    cfg = serve.primary_config(model_dir)
    params_cfg = serve.read_json_file(model_dir / "params.json") or {}
    metrics = serve.estimate_llm_metrics(serve.parse_llm_dims(cfg, params_cfg))
    assert metrics is not None, f"{model_dir.name}: no estimate produced"

    total_size = read_total_size(model_dir)
    est_bytes = metrics["total_params"] * 2  # bf16 proxy
    deviation = abs(est_bytes - total_size) / total_size

    tol, reason = KNOWN_EXCEPTIONS.get(model_dir.name, (DEFAULT_TOL, ""))
    msg = (
        f"{model_dir.name}: est {metrics['total_params']/1e9:.2f}B "
        f"(~{est_bytes/1e9:.1f} GB) vs checkpoint {total_size/1e9:.1f} GB, "
        f"deviation {deviation*100:.1f}% > {tol*100:.0f}%"
    )
    if reason:
        msg += f" [exception: {reason}]"
    assert deviation <= tol, msg

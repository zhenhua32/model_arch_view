#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Serve a local model architecture viewer for model_configs."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parent
MODEL_CONFIGS_DIR = ROOT_DIR / "model_configs"
WEB_DIR = ROOT_DIR / "web"


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_simple_yaml_file(path: Path) -> dict[str, Any]:
    """Read scalar YAML mappings without requiring PyYAML at runtime."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in lines:
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith(("#", "-")) or ":" not in stripped:
            continue
        indent = len(raw_line) - len(stripped)
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        stripped_value = raw_value.strip()
        if stripped_value.startswith("!"):
            child = {"_tag": stripped_value}
            parent[key] = child
            stack.append((indent, child))
            continue
        if stripped_value:
            parent[key] = _parse_simple_yaml_scalar(raw_value)
            continue
        child: dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))
    return root


def supplemental_yaml_config(model_dir: Path) -> dict[str, Any]:
    model_dir = architecture_config_dir(model_dir)
    for name in ("config.yaml", "config.yml", "cosyvoice2.yaml"):
        path = model_dir / name
        if path.exists():
            return read_simple_yaml_file(path)
    return {}


@lru_cache(maxsize=None)
def model_card_base_reference(model_dir: Path) -> str | None:
    readme = model_dir / "README.md"
    try:
        lines = readme.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    if not lines or lines[0].strip() != "---":
        return None
    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line)
    for index, line in enumerate(frontmatter):
        if not line.startswith("base_model:"):
            continue
        inline = line.split(":", 1)[1].strip()
        if inline:
            return inline.strip("'\"")
        for nested in frontmatter[index + 1:]:
            if nested and not nested[0].isspace() and not nested.lstrip().startswith("-"):
                break
            candidate = nested.strip()
            if candidate.startswith("-"):
                return candidate[1:].strip().strip("'\"")
    return None


@lru_cache(maxsize=None)
def local_base_model_dir(model_dir: Path) -> Path | None:
    reference = model_card_base_reference(model_dir)
    if not reference:
        return None
    normalized = reference.split("@", 1)[0].strip().replace("/", "__")
    if not normalized or normalized in {".", ".."}:
        return None
    candidate = MODEL_CONFIGS_DIR / normalized
    if candidate.is_dir() and candidate.resolve() != model_dir.resolve():
        return candidate
    normalized_folded = normalized.casefold()
    for possible in MODEL_CONFIGS_DIR.iterdir():
        if possible.is_dir() and possible.name.casefold() == normalized_folded and possible.resolve() != model_dir.resolve():
            return possible
    return None


@lru_cache(maxsize=None)
def architecture_config_dir(model_dir: Path) -> Path:
    current = model_dir
    seen: set[Path] = set()
    for _ in range(4):
        resolved = current.resolve()
        if resolved in seen:
            break
        seen.add(resolved)
        base_dir = local_base_model_dir(current)
        if not base_dir:
            break
        current = base_dir
    return current


def direct_primary_config(model_dir: Path) -> dict[str, Any]:
    for name in ("config.json", "configuration.json"):
        path = model_dir / name
        if path.exists():
            data = read_json_file(path)
            if data:
                return data
    return {}


def primary_config(model_dir: Path) -> dict[str, Any]:
    return direct_primary_config(architecture_config_dir(model_dir))


def first_defined(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            continue
        return value
    return None


def clamp_int(value: Any, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    if parsed < minimum:
        parsed = minimum
    if maximum is not None and parsed > maximum:
        parsed = maximum
    return parsed


def scalar_int(value: Any, default: int = 1) -> int:
    """Coerce a possibly list/tuple-valued config field to a single int.

    Video diffusion transformers often express patch_size as [t, h, w]; for
    spatial token accounting we want the spatial (last) dimension. Falls back to
    ``default`` when the value cannot be parsed.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[-1]
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def spatial_pair(value: Any, default: int = 1) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return scalar_int(value[-2], default), scalar_int(value[-1], default)
        if len(value) == 1:
            size = scalar_int(value[0], default)
            return size, size
        return default, default
    size = scalar_int(value, default)
    return size, size


def temporal_patch_size(value: Any, default: int = 1) -> int:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return scalar_int(value[0], default)
    return default


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return numerator
    return (numerator + denominator - 1) // denominator


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "-"
    return str(value)


def detail(label: str, value: Any) -> dict[str, str]:
    return {"label": label, "value": format_value(value)}


def section(title: str, items: list[dict[str, str]]) -> dict[str, Any]:
    return {"title": title, "items": items}


def shape(*dims: Any) -> str:
    rendered = ["?" if dim is None else str(dim) for dim in dims]
    return "[" + ", ".join(rendered) + "]"


def human_model_name(model_id: str) -> str:
    return model_id.replace("__", "/")


def count_model_files(model_dir: Path) -> int:
    return sum(1 for path in model_dir.rglob("*") if path.is_file())


# Substantive diffusers pipeline component keys (i.e. actual model sub-directories).
# A bare model_index.json is not sufficient: some LLM checkpoints also ship a
# model_index.json (e.g. a single "model"/"processor"/"scheduler" map) without any
# real diffusers component directory, and must be treated as an LLM instead. We
# require a *substantive* component -- tokenizers/schedulers alone are too generic
# because many non-diffusers checkpoints carry those keys.
_DIFFUSERS_COMPONENT_KEYS = {
    "transformer",
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "controlnet",
    "mllm",
}


_DIFFUSION_TRANSFORMER_CONFIG_PATHS = (
    "config.json",
    "transformer/config.json",
    "base_model/config.json",
    "low_noise_model/config.json",
    "high_noise_model/config.json",
)


def is_diffusion_transformer_config(config: dict[str, Any]) -> bool:
    class_name = str(config.get("_class_name") or "").lower()
    has_width = any(config.get(key) for key in ("hidden_size", "dim", "model_dim", "num_attention_heads", "num_heads"))
    has_depth = any(config.get(key) for key in ("num_hidden_layers", "num_layers", "depth", "num_transformer_blocks"))
    diffusion_class = any(token in class_name for token in ("wanmodel", "transformer2d", "transformer3d", "videotransformer", "ditmodel"))
    return bool(has_width and has_depth and (diffusion_class or config.get("_diffusers_version")))


def discover_diffusion_transformer_configs(model_dir: Path) -> list[tuple[dict[str, Any], str]]:
    discovered: list[tuple[dict[str, Any], str]] = []
    for relative_path in _DIFFUSION_TRANSFORMER_CONFIG_PATHS:
        path = model_dir / relative_path
        if not path.exists():
            continue
        config = read_json_file(path)
        if is_diffusion_transformer_config(config):
            discovered.append((config, relative_path))
    return discovered


def has_llm_dimensions(config: dict[str, Any], params: dict[str, Any] | None = None) -> bool:
    params = params or {}
    hidden_size = first_defined(config.get("hidden_size"), params.get("dim"))
    num_layers = first_defined(config.get("num_hidden_layers"), config.get("num_layers"), params.get("n_layers"))
    try:
        return int(hidden_size or 0) > 0 and int(num_layers or 0) > 0
    except (TypeError, ValueError):
        return False


def _model_identity(config: dict[str, Any], yaml_config: dict[str, Any] | None = None) -> str:
    values: list[Any] = [config.get("model_type"), config.get("architecture"), config.get("task")]
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        values.extend(architectures)
    elif architectures:
        values.append(architectures)
    if yaml_config:
        values.extend([yaml_config.get("model"), yaml_config.get("encoder"), yaml_config.get("audio_encoder")])
    return " ".join(str(value).lower() for value in values if value)


def is_asr_model_config(config: dict[str, Any], yaml_config: dict[str, Any] | None = None) -> bool:
    identity = _model_identity(config, yaml_config)
    if any(token in identity for token in ("asr", "transcrib", "sensevoice", "speechrecognition")):
        return True
    if not yaml_config:
        return False
    has_audio_encoder = bool(yaml_config.get("audio_encoder") or yaml_config.get("encoder"))
    has_frontend = isinstance(yaml_config.get("frontend_conf"), dict)
    dataset = str(yaml_config.get("dataset") or "").lower()
    return has_audio_encoder and has_frontend and "ctc" in dataset


def is_tts_model_config(config: dict[str, Any]) -> bool:
    identity = _model_identity(config)
    return "tts" in identity or "texttospeech" in identity or "text-to-speech" in identity or bool(config.get("audio_vae_config") and config.get("lm_config"))


def is_sam_video_config(config: dict[str, Any]) -> bool:
    identity = _model_identity(config)
    return "sam3video" in identity.replace("_", "") or "sam3_video" in identity


def is_hy_world_bundle(model_dir: Path) -> bool:
    return (model_dir / "HY-Pano-2.0" / "config.json").exists() and (model_dir / "HY-WorldMirror-2.0" / "config.json").exists()


def is_nemotron_streaming_asr(model_dir: Path) -> bool:
    if "nemotron-3.5-asr-streaming" not in model_dir.name.lower():
        return False
    try:
        readme = (model_dir / "README.md").read_text(encoding="utf-8-sig")
    except OSError:
        return False
    return "FastConformer-CacheAware-RNNT" in readme and "24 encoder layers" in readme


def processor_has_modal_inputs(processor: dict[str, Any]) -> bool:
    if any(key in processor for key in ("image_processor", "video_processor", "feature_extractor")):
        return True
    processor_class = str(processor.get("processor_class") or "").lower()
    return any(token in processor_class for token in ("image", "video", "vision", "audio", "speech", "asr", "tts"))


def classify_model_dir(model_dir: Path) -> str:
    if is_hy_world_bundle(model_dir) or is_nemotron_streaming_asr(model_dir):
        return "multimodal"
    config_dir = architecture_config_dir(model_dir)
    model_index = config_dir / "model_index.json"
    if model_index.exists():
        index_data = read_json_file(model_index) or {}
        # Only treat it as a diffusers pipeline when it actually declares a
        # substantive component directory. Otherwise it is most likely an LLM
        # checkpoint that happens to ship a model_index.json.
        if any(key in index_data for key in _DIFFUSERS_COMPONENT_KEYS):
            return "diffusers"

    if discover_diffusion_transformer_configs(config_dir):
        return "diffusers"

    config = primary_config(model_dir)
    yaml_config = supplemental_yaml_config(model_dir)
    params = read_json_file(config_dir / "params.json")
    processor = read_json_file(config_dir / "processor_config.json")

    if is_asr_model_config(config, yaml_config) or is_tts_model_config(config) or is_sam_video_config(config):
        return "multimodal"

    thinker_config = config.get("thinker_config") if isinstance(config.get("thinker_config"), dict) else {}
    if (
        config.get("vision_config")
        or config.get("audio_config")
        or thinker_config.get("audio_config")
        or config.get("image_token_id")
        or config.get("video_token_id")
        or config.get("audio_token_id")
        or params.get("vision_encoder")
        or processor_has_modal_inputs(processor)
    ):
        return "multimodal"

    task = str(config.get("task") or "").lower()
    if task and not any(token in task for token in ("text-generation", "feature-extraction", "sentence-similarity")):
        return "unknown"

    if has_llm_dimensions(config, params):
        return "llm"

    return "unknown"


def infer_architecture_name(model_dir: Path, model_type: str) -> str:
    if is_hy_world_bundle(model_dir):
        return "HYWorld2Pipeline"
    if is_nemotron_streaming_asr(model_dir):
        return "FastConformerCacheAwareRNNT"
    config_dir = architecture_config_dir(model_dir)
    if model_type == "diffusers":
        model_index = read_json_file(config_dir / "model_index.json")
        if model_index.get("_class_name"):
            return str(model_index["_class_name"])
        discovered = discover_diffusion_transformer_configs(config_dir)
        if discovered:
            return str(discovered[0][0].get("_class_name") or "DiffusionTransformer")
        return "DiffusersPipeline"

    config = primary_config(model_dir)
    if config.get("architectures"):
        return str(config["architectures"][0])
    if config.get("architecture"):
        return str(config["architecture"])

    params = read_json_file(config_dir / "params.json")
    if params.get("vision_encoder"):
        return "VisionLanguageModel"
    if params:
        return "TransformerModel"

    yaml_config = supplemental_yaml_config(model_dir)
    yaml_architecture = first_defined(yaml_config.get("model"), yaml_config.get("audio_encoder"), yaml_config.get("encoder"))
    if yaml_architecture:
        return str(yaml_architecture)
    if is_tts_model_config(config):
        return human_model_name(model_dir.name).split("/")[-1]

    return "UnknownModel"


def build_model_catalog() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if not MODEL_CONFIGS_DIR.exists():
        return models

    for model_dir in sorted(MODEL_CONFIGS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue

        model_type = classify_model_dir(model_dir)
        models.append(
            {
                "id": model_dir.name,
                "name": human_model_name(model_dir.name),
                "type": model_type,
                "architecture": infer_architecture_name(model_dir, model_type),
                "fileCount": count_model_files(model_dir),
            }
        )

    return models


def resolve_model_dir(model_id: str) -> Path:
    root = MODEL_CONFIGS_DIR.resolve()
    candidate = (MODEL_CONFIGS_DIR / model_id).resolve()
    if candidate.parent != root or not candidate.exists() or not candidate.is_dir():
        raise FileNotFoundError(model_id)
    return candidate


def append_source_file(model_dir: Path, sources: list[str], relative_path: str) -> None:
    path = model_dir / relative_path
    if path.exists():
        normalized = relative_path.replace("\\", "/")
        if normalized not in sources:
            sources.append(normalized)


def base_model_payload(
    model_id: str,
    model_type: str,
    architecture: str,
    headline: str,
    summary: list[dict[str, str]],
    controls: list[dict[str, Any]],
    parameters: dict[str, Any],
    graph: dict[str, Any],
    warnings: list[str],
    sources: list[str],
    selected_node_id: str,
) -> dict[str, Any]:
    return {
        "model": {
            "id": model_id,
            "name": human_model_name(model_id),
            "type": model_type,
            "architecture": architecture,
            "headline": headline,
            "summary": summary,
            "sources": sources,
        },
        "controls": controls,
        "parameters": parameters,
        "graph": graph,
        "warnings": warnings,
        "selectedNodeId": selected_node_id,
    }


def build_node(
    node_id: str,
    lane: str,
    order: int,
    label: str,
    subtitle: str,
    description: str,
    input_shape: str,
    output_shape: str,
    badges: list[str],
    details: list[dict[str, str]],
    sections: list[dict[str, Any]],
    accent: str,
    micro_flow: list[str] | None = None,
    parent_id: str | None = None,
    view_modes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "lane": lane,
        "order": order,
        "label": label,
        "subtitle": subtitle,
        "description": description,
        "inputShape": input_shape,
        "outputShape": output_shape,
        "badges": badges,
        "details": details,
        "sections": sections,
        "accent": accent,
        "microFlow": micro_flow or [],
        "parentId": parent_id,
        "viewModes": view_modes or ["summary", "expanded"],
    }


def build_graph(lanes: list[tuple[str, str]], nodes: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "lanes": [{"id": lane_id, "label": label} for lane_id, label in lanes],
        "nodes": nodes,
        "edges": edges,
    }


def derive_vae_scale(vae_config: dict[str, Any]) -> int:
    dim_mult = vae_config.get("dim_mult")
    if isinstance(dim_mult, list) and dim_mult:
        return 2 ** max(len(dim_mult) - 1, 0)

    block_out_channels = vae_config.get("block_out_channels")
    if isinstance(block_out_channels, list) and block_out_channels:
        return 2 ** max(len(block_out_channels) - 1, 0)

    return 8


def derive_vae_spatial_scales(vae_config: dict[str, Any]) -> tuple[int, int]:
    base_scale = derive_vae_scale(vae_config)
    patch_height, patch_width = spatial_pair(vae_config.get("patch_size"), 1)
    explicit = vae_config.get("spatial_compression_ratio")
    if isinstance(explicit, (int, float)) and explicit > 0:
        return int(explicit), int(explicit)
    return base_scale * patch_height, base_scale * patch_width


def derive_vae_temporal_scale(vae_config: dict[str, Any]) -> int:
    explicit = first_defined(
        vae_config.get("temporal_compression_ratio"),
        vae_config.get("time_compression_ratio"),
        vae_config.get("temporal_downsample_factor"),
    )
    if isinstance(explicit, (int, float)) and explicit > 0:
        return int(explicit)
    downsample_flags = first_defined(vae_config.get("temporal_downsample"), vae_config.get("temperal_downsample"))
    if isinstance(downsample_flags, list):
        return 2 ** sum(bool(item) for item in downsample_flags)
    return 1


def build_edge(source: str, target: str, label: str, view_modes: list[str] | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "label": label,
        "viewModes": view_modes or ["summary", "expanded"],
    }


def max_nested_numeric(mapping: dict[str, Any], key: str) -> int | None:
    values: list[int] = []
    for value in mapping.values():
        if isinstance(value, dict):
            numeric = value.get(key)
            if isinstance(numeric, (int, float)):
                values.append(int(numeric))
    return max(values) if values else None


def summarize_layer_pattern(layer_types: Any) -> str | None:
    if not isinstance(layer_types, list) or not layer_types:
        return None
    full_attention = sum(1 for item in layer_types if "full" in str(item))
    sliding_attention = sum(1 for item in layer_types if "sliding" in str(item) or "local" in str(item))
    linear_attention = sum(1 for item in layer_types if "linear" in str(item))
    return f"linear={linear_attention}, sliding={sliding_attention}, full={full_attention}"


_ENCODER_ONLY_MODEL_TYPES = {
    "bert",
    "deberta",
    "deberta-v2",
    "electra",
    "roberta",
    "xlm-roberta",
}


def _first_architecture(config: dict[str, Any]) -> str:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return ""


def _attention_layer_counts(layer_types: Any, num_layers: int, use_sliding_window: bool) -> tuple[int, int, int]:
    if isinstance(layer_types, list) and layer_types:
        normalized = [str(item).lower() for item in layer_types[:num_layers]]
        full = sum("full" in item for item in normalized)
        sliding_candidates = sum("sliding" in item or "local" in item for item in normalized)
        sliding = sliding_candidates if use_sliding_window else 0
        full += 0 if use_sliding_window else sliding_candidates
        linear = sum("linear" in item for item in normalized)
        unclassified = max(num_layers - full - sliding - linear, 0)
        return full + unclassified, sliding, linear
    if use_sliding_window:
        return 0, num_layers, 0
    return num_layers, 0, 0


def parse_llm_dims(config: dict[str, Any], params_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse the architecture dimensions used for parameter/KV/FLOPs estimation.

    Single source of truth for both ``build_llm_payload`` and the test-suite so the
    numbers rendered in the UI are exactly what the tests assert on. All aliases
    (Qwen/DeepSeek/GLM/LongCat/params.json-style) are resolved here.
    """
    params_config = params_config or {}
    moe_config = params_config.get("moe") if isinstance(params_config.get("moe"), dict) else {}
    architecture = _first_architecture(config)
    model_type = str(config.get("model_type") or "").lower()
    is_deepseek_v4 = model_type == "deepseek_v4"
    encoder_only = model_type in _ENCODER_ONLY_MODEL_TYPES or (
        architecture.endswith("Model") and not any(marker in architecture for marker in ("CausalLM", "ConditionalGeneration", "LMHead"))
    )

    hidden_size = int(first_defined(config.get("hidden_size"), params_config.get("dim"), 0) or 0)
    num_layers = int(first_defined(config.get("num_hidden_layers"), config.get("num_layers"), params_config.get("n_layers"), 0) or 0)
    num_heads = int(first_defined(config.get("num_attention_heads"), params_config.get("n_heads"), 0) or 0)
    num_kv_heads = int(first_defined(config.get("num_key_value_heads"), params_config.get("n_kv_heads"), num_heads) or num_heads or 0)
    head_dim = int(first_defined(config.get("head_dim"), params_config.get("head_dim"), hidden_size // num_heads if hidden_size and num_heads else 0) or 0)
    ffn_hidden = int(first_defined(config.get("intermediate_size"), params_config.get("hidden_dim"), config.get("moe_intermediate_size"), 0) or 0)
    vocab_size = int(first_defined(config.get("vocab_size"), params_config.get("vocab_size"), 0) or 0)

    num_experts = int(first_defined(config.get("n_routed_experts"), config.get("num_experts"), config.get("num_local_experts"), moe_config.get("num_experts"), 0) or 0)
    experts_per_tok = scalar_int(first_defined(config.get("num_experts_per_tok"), config.get("moe_topk"), moe_config.get("num_experts_per_tok"), moe_config.get("moe_topk")), 0)
    moe_ffn_hidden = scalar_int(first_defined(config.get("moe_intermediate_size"), config.get("expert_ffn_hidden_size"), moe_config.get("moe_intermediate_size")), 0)
    explicit_shared_dim = first_defined(
        config.get("shared_expert_intermediate_size"),
        config.get("moe_shared_expert_intermediate_size"),
        moe_config.get("shared_expert_intermediate_size"),
        moe_config.get("moe_shared_expert_intermediate_size"),
    )
    n_shared_experts = scalar_int(first_defined(
        config.get("n_shared_experts"),
        config.get("num_shared_experts"),
        config.get("num_shared_expert"),
        moe_config.get("n_shared_experts"),
        moe_config.get("num_shared_experts"),
        1 if explicit_shared_dim is not None else 0,
    ), 0)
    shared_ffn_hidden = scalar_int(first_defined(explicit_shared_dim, moe_ffn_hidden), 0)
    first_k_dense = int(first_defined(config.get("first_k_dense_replace"), config.get("n_dense_layers"), 0) or 0)
    tie_word_embeddings = bool(first_defined(config.get("tie_word_embeddings"), params_config.get("tie_word_embeddings"), False))
    ffn_projection_count = 2 if encoder_only else 3
    active_expert_multiplier = max(int(first_defined(config.get("cli_factor"), 1) or 1), 1)
    mtp_module_layers = int(first_defined(config.get("mtp_transformer_layers"), 1) or 1)
    mtp_module_count = int(first_defined(config.get("num_mtp_modules"), 0) or 0)
    mtp_layers = int(first_defined(config.get("mtp_num_layers"), mtp_module_count * mtp_module_layers if mtp_module_count else None, config.get("num_nextn_predict_layers"), 0) or 0)
    include_mtp_in_total = bool(config.get("mtp_num_layers") or (config.get("use_mtp") and mtp_module_count))

    max_position_embeddings = int(first_defined(config.get("max_position_embeddings"), params_config.get("max_position_embeddings"), 0) or 0)
    position_embedding_params = hidden_size * max_position_embeddings if encoder_only and config.get("position_embedding_type", "absolute") == "absolute" else 0
    token_type_vocab_size = int(first_defined(config.get("type_vocab_size"), 0) or 0)
    token_type_embedding_params = hidden_size * token_type_vocab_size if encoder_only else 0

    sliding_window = int(first_defined(config.get("sliding_window"), config.get("sliding_window_size"), params_config.get("sliding_window_size"), 0) or 0)
    layer_types = config.get("layer_types")
    layer_types_enable_sliding = isinstance(layer_types, list) and any("sliding" in str(item).lower() or "local" in str(item).lower() for item in layer_types)
    explicit_sliding = config.get("use_sliding_window")
    use_sliding_window = bool(explicit_sliding) if explicit_sliding is not None else layer_types_enable_sliding
    full_attention_layers, sliding_attention_layers, linear_attention_layers = _attention_layer_counts(
        layer_types, num_layers, use_sliding_window
    )
    dsa_layers = config.get("dsa_layers") if isinstance(config.get("dsa_layers"), list) else []
    swa_layers = config.get("swa_layers") if isinstance(config.get("swa_layers"), list) else []
    sparse_attention_layers = 0
    sparse_topk = 0
    if dsa_layers or swa_layers:
        sparse_attention_layers = sum(isinstance(index, int) and 0 <= index < num_layers for index in dsa_layers)
        sliding_attention_layers = sum(isinstance(index, int) and 0 <= index < num_layers for index in swa_layers)
        full_attention_layers = max(num_layers - sparse_attention_layers - sliding_attention_layers, 0)
        linear_attention_layers = 0
        sparse_topk = int(first_defined(config.get("index_topk"), 0) or 0)
        use_sliding_window = sliding_attention_layers > 0

    q_lora_rank = int(first_defined(config.get("q_lora_rank"), 0) or 0)
    kv_lora_rank = int(first_defined(config.get("kv_lora_rank"), config.get("kv_lora_a"), 0) or 0)
    qk_rope_head_dim = int(first_defined(config.get("qk_rope_head_dim"), 0) or 0)
    qk_nope_head_dim = int(first_defined(
        config.get("qk_nope_head_dim"),
        (head_dim - qk_rope_head_dim) if (head_dim and qk_rope_head_dim) else head_dim,
        0,
    ) or 0)
    qk_head_dim = int(first_defined(
        config.get("qk_head_dim"),
        (qk_nope_head_dim + qk_rope_head_dim) if (qk_nope_head_dim and qk_rope_head_dim) else head_dim,
        0,
    ) or 0)
    v_head_dim = int(first_defined(config.get("v_head_dim"), head_dim) or 0)

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "ffn_hidden": ffn_hidden,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "experts_per_tok": experts_per_tok,
        "moe_ffn_hidden": moe_ffn_hidden,
        "n_shared_experts": n_shared_experts,
        "shared_ffn_hidden": shared_ffn_hidden,
        "first_k_dense": first_k_dense,
        "tie_word_embeddings": tie_word_embeddings,
        "ffn_projection_count": ffn_projection_count,
        "has_output_head": not encoder_only,
        "position_embedding_params": position_embedding_params,
        "token_type_embedding_params": token_type_embedding_params,
        "active_expert_multiplier": active_expert_multiplier,
        "mtp_layers": mtp_layers,
        "include_mtp_in_total": include_mtp_in_total,
        "sliding_window": sliding_window if (use_sliding_window or is_deepseek_v4) else 0,
        "use_sliding_window": use_sliding_window,
        "full_attention_layers": full_attention_layers,
        "sliding_attention_layers": sliding_attention_layers,
        "linear_attention_layers": linear_attention_layers,
        "sparse_attention_layers": sparse_attention_layers,
        "sparse_topk": sparse_topk,
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_head_dim": qk_rope_head_dim,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "is_mla": not is_deepseek_v4 and (q_lora_rank > 0 or kv_lora_rank > 0 or bool(config.get("use_mla")) or str(config.get("attention_method", "")).upper() == "MLA"),
        "is_deepseek_v4": is_deepseek_v4,
        "o_lora_rank": int(first_defined(config.get("o_lora_rank"), 0) or 0),
        "o_groups": int(first_defined(config.get("o_groups"), 1) or 1),
        "index_head_dim": int(first_defined(config.get("index_head_dim"), 0) or 0),
        "index_n_heads": int(first_defined(config.get("index_n_heads"), 0) or 0),
        "index_topk": int(first_defined(config.get("index_topk"), 0) or 0),
        "compress_ratios": [int(value) for value in config.get("compress_ratios", [])[:num_layers] if isinstance(value, (int, float))],
        "hc_mult": int(first_defined(config.get("hc_mult"), 1) or 1),
    }


def _visible_token_pairs(seq_len: int, window: int = 0) -> int:
    if seq_len <= 0:
        return 0
    if window <= 0 or window >= seq_len:
        return seq_len * (seq_len + 1) // 2
    return window * (window + 1) // 2 + (seq_len - window) * window


def _compressed_visible_pairs(seq_len: int, ratio: int, limit: int = 0) -> int:
    if seq_len <= 0 or ratio <= 0:
        return 0
    uncapped_len = seq_len if limit <= 0 else min(seq_len, limit * ratio - 1)
    quotient, remainder = divmod(uncapped_len, ratio)
    total = ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)
    if limit > 0 and seq_len > uncapped_len:
        total += (seq_len - uncapped_len) * limit
    return total


def _deepseek_v4_attention_params(dims: dict[str, Any]) -> tuple[int, int]:
    hidden_size = dims["hidden_size"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    q_lora_rank = dims["q_lora_rank"]
    o_lora_rank = dims["o_lora_rank"]
    o_groups = dims["o_groups"]
    index_head_dim = dims["index_head_dim"]
    index_n_heads = dims["index_n_heads"]
    hc_mult = dims["hc_mult"]
    ratios = dims["compress_ratios"] or [0] * dims["num_layers"]

    base = hidden_size * q_lora_rank
    base += q_lora_rank * num_heads * head_dim
    base += hidden_size * head_dim
    base += num_heads * head_dim * o_lora_rank
    base += o_groups * o_lora_rank * hidden_size
    base += q_lora_rank + head_dim + num_heads
    mix_width = (2 + hc_mult) * hc_mult
    hyper_connection = 2 * (mix_width * hc_mult * hidden_size + mix_width + 3)

    layer_params: list[int] = []
    for ratio in ratios:
        compressor = 0
        if ratio == 4:
            compressor += 4 * hidden_size * head_dim + 2 * ratio * head_dim + head_dim
            compressor += 4 * hidden_size * index_head_dim + 2 * ratio * index_head_dim + index_head_dim
            compressor += q_lora_rank * index_n_heads * index_head_dim + hidden_size * index_n_heads
        elif ratio > 0:
            compressor += 2 * hidden_size * head_dim + ratio * head_dim + head_dim
        layer_params.append(base + compressor + hyper_connection)
    total = sum(layer_params)
    average = round(total / len(layer_params)) if layer_params else base + hyper_connection
    return average, total


def _context_layer_tokens(metrics: dict[str, Any], seq_len: int) -> int:
    if metrics.get("is_deepseek_v4"):
        ratios = metrics.get("compress_ratios") or []
        window = int(metrics.get("sliding_window", 0) or 0)
        head_dim = int(metrics.get("head_dim", 0) or 0)
        index_topk = int(metrics.get("index_topk", 0) or 0)
        if not ratios or not head_dim:
            return 0
        local_tokens = min(seq_len, window or seq_len)
        total = 0
        for ratio in ratios:
            total += local_tokens
            if ratio == 4:
                total += min(index_topk, ceil_div(seq_len, ratio))
            elif ratio > 0:
                total += ceil_div(seq_len, ratio)
        return total
    full_layers = int(metrics.get("full_attention_layers", metrics.get("num_layers", 0)) or 0)
    sliding_layers = int(metrics.get("sliding_attention_layers", 0) or 0)
    sparse_layers = int(metrics.get("sparse_attention_layers", 0) or 0)
    sparse_topk = int(metrics.get("sparse_topk", 0) or 0)
    window = int(metrics.get("sliding_window", 0) or 0)
    return full_layers * seq_len + sliding_layers * min(seq_len, window or seq_len) + sparse_layers * min(seq_len, sparse_topk or seq_len)


def _indexer_flops_per_token(metrics: dict[str, Any], seq_len: int) -> int:
    index_width = int(metrics.get("index_n_heads", 0) or 0) * int(metrics.get("index_head_dim", 0) or 0)
    if index_width <= 0 or seq_len <= 0:
        return 0
    if metrics.get("is_deepseek_v4"):
        csa_layers = sum(ratio == 4 for ratio in metrics.get("compress_ratios") or [])
        return 2 * index_width * ceil_div(seq_len, 4) * csa_layers
    sparse_layers = int(metrics.get("sparse_attention_layers", 0) or 0)
    return 2 * index_width * seq_len * sparse_layers


def _kv_cache_bytes(metrics: dict[str, Any], seq_len: int, batch: int = 1) -> int:
    if metrics.get("is_deepseek_v4"):
        ratios = metrics.get("compress_ratios") or []
        window = int(metrics.get("sliding_window", 0) or 0)
        head_dim = int(metrics.get("head_dim", 0) or 0)
        index_head_dim = int(metrics.get("index_head_dim", 0) or 0)
        local_tokens = min(seq_len, window or seq_len)
        cached_elements = 0
        for ratio in ratios:
            cached_elements += local_tokens * head_dim
            if ratio > 0:
                compressed_tokens = ceil_div(seq_len, ratio)
                cached_elements += compressed_tokens * head_dim
                if ratio == 4:
                    cached_elements += compressed_tokens * index_head_dim
        return cached_elements * 2 * batch
    per_layer = int(metrics.get("kv_bytes_per_token_per_layer", 0) or 0)
    if per_layer <= 0:
        return int(metrics.get("kv_bytes_per_token", 0) or 0) * seq_len * batch
    full_layers = int(metrics.get("full_attention_layers", metrics.get("num_layers", 0)) or 0)
    sparse_layers = int(metrics.get("sparse_attention_layers", 0) or 0)
    sliding_layers = int(metrics.get("sliding_attention_layers", 0) or 0)
    window = int(metrics.get("sliding_window", 0) or 0)
    cached_layer_tokens = (full_layers + sparse_layers) * seq_len + sliding_layers * min(seq_len, window or seq_len)
    return per_layer * cached_layer_tokens * batch


def estimate_llm_metrics(dims: dict[str, Any], seq_len: int = 0) -> dict[str, Any] | None:
    """Estimate parameter count, KV-cache size and per-token FLOPs from parsed dims.

    Returns ``None`` when the config lacks the minimum required fields
    (``hidden_size`` and ``num_layers``). All returned figures are raw numbers:
      - total_params / active_params: parameter counts
      - kv_cache_mb_per_1k: MiB of KV cache per 1K tokens per batch element
      - gflops_per_token: linear weights plus context-dependent QK/AV work
    plus a per-component breakdown for white-box assertions.
    """
    hidden_size = dims["hidden_size"]
    num_layers = dims["num_layers"]
    if not hidden_size or not num_layers:
        return None

    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    ffn_hidden = dims["ffn_hidden"]
    vocab_size = dims["vocab_size"]
    num_experts = dims["num_experts"]
    experts_per_tok = dims["experts_per_tok"]
    moe_ffn_hidden = dims["moe_ffn_hidden"]
    n_shared_experts = dims["n_shared_experts"]
    shared_ffn_hidden = dims["shared_ffn_hidden"]
    first_k_dense = dims["first_k_dense"]
    tie_word_embeddings = dims["tie_word_embeddings"]
    ffn_projection_count = dims["ffn_projection_count"]
    has_output_head = dims["has_output_head"]
    active_expert_multiplier = dims["active_expert_multiplier"]
    mtp_layers = dims["mtp_layers"]
    include_mtp_in_total = dims["include_mtp_in_total"]
    is_mla = dims["is_mla"]
    q_lora_rank = dims["q_lora_rank"]
    kv_lora_rank = dims["kv_lora_rank"]
    qk_rope_head_dim = dims["qk_rope_head_dim"]
    qk_nope_head_dim = dims["qk_nope_head_dim"]
    qk_head_dim = dims["qk_head_dim"]
    v_head_dim = dims["v_head_dim"]
    is_deepseek_v4 = dims["is_deepseek_v4"]

    special_attn_total = 0
    if is_deepseek_v4:
        attn_params, special_attn_total = _deepseek_v4_attention_params(dims)
    elif is_mla:
        if q_lora_rank:
            q_params = hidden_size * q_lora_rank + q_lora_rank * num_heads * (qk_head_dim or head_dim)
        else:
            q_params = hidden_size * num_heads * (qk_head_dim or head_dim)
        kv_down = hidden_size * ((kv_lora_rank or head_dim) + qk_rope_head_dim)
        kv_up = (kv_lora_rank or head_dim) * num_heads * ((qk_nope_head_dim or head_dim) + (v_head_dim or head_dim))
        o_params = num_heads * (v_head_dim or head_dim) * hidden_size
        attn_params = q_params + kv_down + kv_up + o_params
    else:
        q_dim = (num_heads * head_dim) if (num_heads and head_dim) else hidden_size
        kv_dim = (num_kv_heads * head_dim) if (num_kv_heads and head_dim) else q_dim
        attn_params = (
            hidden_size * q_dim          # q_proj
            + hidden_size * kv_dim       # k_proj
            + hidden_size * kv_dim       # v_proj
            + q_dim * hidden_size        # o_proj
        )

    # ---- FFN / MoE params ----
    dense_ffn_dim = ffn_hidden or (hidden_size * 4)
    dense_ffn_params = ffn_projection_count * hidden_size * dense_ffn_dim
    # Per-component FFN totals (total-parameter convention) for the breakdown chart.
    expert_params = 0
    n_dense_layers = 0
    n_moe_layers = 0
    routed_experts_total = 0
    shared_experts_total = 0
    dense_ffn_total = 0
    routed_experts_active = 0
    shared_experts_active = 0
    dense_ffn_active = 0
    if num_experts:
        expert_dim = moe_ffn_hidden or dense_ffn_dim
        expert_params = ffn_projection_count * hidden_size * expert_dim
        routed_total = expert_params * num_experts
        routed_active = expert_params * (experts_per_tok or 1) * active_expert_multiplier
        shared_expert_params = ffn_projection_count * hidden_size * (shared_ffn_hidden or expert_dim)
        shared_params = shared_expert_params * n_shared_experts
        n_dense_layers = min(first_k_dense, num_layers) if first_k_dense else 0
        n_moe_layers = num_layers - n_dense_layers
        routed_experts_total = n_moe_layers * routed_total
        shared_experts_total = n_moe_layers * shared_params
        dense_ffn_total = n_dense_layers * dense_ffn_params
        routed_experts_active = n_moe_layers * routed_active
        shared_experts_active = n_moe_layers * shared_params
        dense_ffn_active = n_dense_layers * dense_ffn_params
        ffn_total = dense_ffn_total + routed_experts_total + shared_experts_total
        ffn_active = dense_ffn_active + routed_experts_active + shared_experts_active
    else:
        ffn_total = ffn_active = num_layers * dense_ffn_params
        dense_ffn_total = ffn_total
        dense_ffn_active = ffn_active
        shared_expert_params = 0

    embed_params = hidden_size * (vocab_size or 0)
    output_head_params = embed_params if has_output_head else 0
    output_head_total = 0 if tie_word_embeddings else output_head_params
    embed_total = embed_params + output_head_total + dims["position_embedding_params"] + dims["token_type_embedding_params"]
    attn_total = special_attn_total or attn_params * num_layers
    mtp_ffn_params = routed_total + shared_params if num_experts else dense_ffn_params
    mtp_params = mtp_layers * (attn_params + mtp_ffn_params)
    mtp_included = mtp_params if include_mtp_in_total else 0
    total_params = attn_total + ffn_total + embed_total + mtp_included
    active_params = attn_total + ffn_active + output_head_params
    non_expert_active_params = active_params - routed_experts_active

    # ---- KV cache (per 1K tokens per batch element) ----
    kv_bytes_per_token = 0  # raw per-token KV bytes (bf16), reused by memory estimator
    kv_bytes_per_token_per_layer = 0
    cache_layer_count = dims["full_attention_layers"] + dims["sliding_attention_layers"] + dims["sparse_attention_layers"]
    if has_output_head and is_deepseek_v4:
        kv_bytes_per_token = 0
    elif has_output_head and is_mla:
        kv_bytes_per_token_per_layer = ((kv_lora_rank or head_dim) + qk_rope_head_dim) * 2
        kv_bytes_per_token = cache_layer_count * kv_bytes_per_token_per_layer
    elif has_output_head and num_kv_heads and head_dim:
        kv_bytes_per_token_per_layer = 2 * num_kv_heads * head_dim * 2
        kv_bytes_per_token = cache_layer_count * kv_bytes_per_token_per_layer

    qk_width = num_heads * ((qk_head_dim or head_dim) if is_mla else head_dim)
    v_width = num_heads * ((v_head_dim or head_dim) if is_mla else head_dim)
    linear_flops_per_token = 2 * active_params
    full_attention_layers = dims["full_attention_layers"]
    sliding_attention_layers = dims["sliding_attention_layers"]
    sparse_attention_layers = dims["sparse_attention_layers"]
    sparse_topk = dims["sparse_topk"]
    sliding_window = dims["sliding_window"]
    context_layer_tokens = full_attention_layers * seq_len
    context_layer_tokens += sliding_attention_layers * min(seq_len, sliding_window or seq_len)
    context_layer_tokens += sparse_attention_layers * min(seq_len, sparse_topk or seq_len)
    if is_deepseek_v4:
        context_layer_tokens = _context_layer_tokens(
            {
                "is_deepseek_v4": True,
                "compress_ratios": dims["compress_ratios"],
                "sliding_window": sliding_window,
                "head_dim": head_dim,
                "index_topk": dims["index_topk"],
            },
            seq_len,
        )
    attention_flops_per_token = 2 * (qk_width + v_width) * context_layer_tokens if seq_len else 0

    metrics = {
        "total_params": total_params,
        "active_params": active_params,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
        "linear_flops_per_token": linear_flops_per_token,
        "attention_flops_per_token": attention_flops_per_token,
        "gflops_per_token": (linear_flops_per_token + attention_flops_per_token) / 1e9,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "head_dim": head_dim,
        "qk_width": qk_width,
        "v_width": v_width,
        "full_attention_layers": full_attention_layers,
        "sliding_attention_layers": sliding_attention_layers,
        "linear_attention_layers": dims["linear_attention_layers"],
        "sparse_attention_layers": sparse_attention_layers,
        "sparse_topk": sparse_topk,
        "sliding_window": sliding_window,
        "causal_attention": has_output_head,
        "is_deepseek_v4": is_deepseek_v4,
        "compress_ratios": dims["compress_ratios"],
        "index_head_dim": dims["index_head_dim"],
        "index_n_heads": dims["index_n_heads"],
        "index_topk": dims["index_topk"],
        "mtp_params": mtp_params,
        "mtp_included": mtp_included,
        "mtp_layers": mtp_layers,
        "output_head_params": output_head_params,
        "position_embedding_params": dims["position_embedding_params"],
        "token_type_embedding_params": dims["token_type_embedding_params"],
        "shared_expert_params": shared_expert_params,
        "effective_experts_per_tok": (experts_per_tok or 1) * active_expert_multiplier if num_experts else 0,
        # component breakdown (white-box + chart). sum(breakdown) == total_params.
        "attn_params_per_layer": attn_params,
        "ffn_total": ffn_total,
        "ffn_active": ffn_active,
        "routed_experts_active": routed_experts_active,
        "shared_experts_active": shared_experts_active,
        "dense_ffn_active": dense_ffn_active,
        "non_expert_active_params": non_expert_active_params,
        "embed_params": embed_params,
        "embed_total": embed_total,
        "expert_params": expert_params,
        "dense_ffn_params": dense_ffn_params,
        "n_dense_layers": n_dense_layers,
        "n_moe_layers": n_moe_layers,
        "breakdown": {
            "attention": attn_total,
            "routed_experts": routed_experts_total,
            "shared_experts": shared_experts_total,
            "dense_ffn": dense_ffn_total,
            "embedding": embed_total,
            "mtp": mtp_included,
        },
    }
    if seq_len:
        metrics["attention_flops_per_token"] += _indexer_flops_per_token(metrics, seq_len)
        metrics["gflops_per_token"] = (linear_flops_per_token + metrics["attention_flops_per_token"]) / 1e9
    kv_cache_bytes_per_1k = _kv_cache_bytes(metrics, 1024)
    metrics["kv_cache_mb_per_1k"] = kv_cache_bytes_per_1k / (1024 * 1024) if (kv_bytes_per_token_per_layer or is_deepseek_v4) else None
    return metrics


# ---- Deployment estimation reference tables --------------------------------
# Approximate spec sheet values; centralised so they are easy to tweak.
# bf16_tflops = dense bf16 tensor-core peak; bw_gbs = HBM bandwidth (GB/s).
GPU_REFERENCE = [
    {
        "name": "RTX 4090", "mem_gb": 24, "bf16_tflops": 165, "bw_gbs": 1008,
        "compute_tops": {"bf16": 165, "fp16": 165, "fp8": 330, "int8": 661, "int4": 1321},
    },
    {
        "name": "A100 80G", "mem_gb": 80, "bf16_tflops": 312, "bw_gbs": 2039,
        "compute_tops": {"bf16": 312, "fp16": 312, "fp8": 312, "int8": 624, "int4": 1248},
    },
    {
        "name": "H100 80G", "mem_gb": 80, "bf16_tflops": 990, "bw_gbs": 3350,
        "compute_tops": {"bf16": 990, "fp16": 990, "fp8": 1979, "int8": 1979, "int4": 3958},
    },
    {
        "name": "H200 141G", "mem_gb": 141, "bf16_tflops": 990, "bw_gbs": 4800,
        "compute_tops": {"bf16": 990, "fp16": 990, "fp8": 1979, "int8": 1979, "int4": 3958},
    },
]

# Bytes per stored parameter for each weight precision.
PRECISION_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}

# Fraction of usable VRAM (framework/fragmentation overhead) and model FLOPs
# utilisation used for the theoretical throughput ceiling.
_VRAM_USABLE = 0.90
_MFU = 0.40
# Coarse single-layer prefill workspace factor. Intermediate buffers are reused
# between layers during inference, so this deliberately does not multiply by L.
_ACT_FACTOR = 8


def infer_weight_precision(config: dict[str, Any], params_config: dict[str, Any] | None = None) -> str:
    params_config = params_config or {}
    quant_payload = {
        "quantization_config": config.get("quantization_config"),
        "compression_config": config.get("compression_config"),
        "quantization": params_config.get("quantization"),
        "torch_dtype": config.get("torch_dtype"),
        "dtype": first_defined(config.get("dtype"), params_config.get("dtype")),
    }
    text = json.dumps(quant_payload, ensure_ascii=False).lower()
    if any(marker in text for marker in ("mxfp4", "fp4", "int4", "w4a16", "4-bit", "4bit")):
        return "int4"
    if any(marker in text for marker in ("int8", "w8a8", "int-quantized")):
        return "int8"
    if any(marker in text for marker in ("fp8", "float8", "e4m3", "e5m2")):
        return "fp8"
    if "float16" in text and "bfloat16" not in text:
        return "fp16"
    return "bf16"


def infer_native_weight_profile(config: dict[str, Any], metrics: dict[str, Any], precision: str) -> dict[str, Any]:
    quantization = config.get("quantization_config") if isinstance(config.get("quantization_config"), dict) else {}
    quant_method = str(quantization.get("quant_method") or "").lower()
    store_dtype = str(quantization.get("store_dtype") or "").lower()
    checkpoint_bytes = int(metrics.get("checkpoint_bytes", 0) or 0)
    total_params = int(metrics.get("total_params", 0) or 0)
    checkpoint_bpp = checkpoint_bytes / total_params if checkpoint_bytes and total_params else None
    has_routed_experts = int(metrics.get("routed_experts_active", 0) or 0) > 0
    explicit_mixed_fp4 = "fp4" in store_dtype and any(token in quant_method for token in ("fp8", "float8"))
    inferred_mixed_fp4 = has_routed_experts and precision == "fp8" and checkpoint_bpp is not None and checkpoint_bpp < 0.75

    profile = {
        "weight_format": precision.upper(),
        "checkpoint_bytes_per_param": checkpoint_bpp,
        "native_compute_precision": precision,
        "native_active_weight_bytes": None,
        "expert_bytes_per_param": PRECISION_BYTES[precision],
        "non_expert_bytes_per_param": PRECISION_BYTES[precision],
    }
    if explicit_mixed_fp4 or inferred_mixed_fp4:
        expert_bpp = PRECISION_BYTES["int4"]
        non_expert_bpp = PRECISION_BYTES["fp8"]
        routed_active = int(metrics.get("routed_experts_active", 0) or 0)
        non_expert_active = int(metrics.get("non_expert_active_params", 0) or 0)
        profile.update(
            {
                "weight_format": "混合 FP4 experts + FP8 core",
                "native_compute_precision": "fp8",
                "native_active_weight_bytes": routed_active * expert_bpp + non_expert_active * non_expert_bpp,
                "expert_bytes_per_param": expert_bpp,
                "non_expert_bytes_per_param": non_expert_bpp,
            }
        )
    elif precision == "fp8" and quantization.get("modules_to_not_convert"):
        profile["weight_format"] = "FP8（部分层保留高精度）"
        if checkpoint_bpp is not None and checkpoint_bpp >= 1.75:
            profile["weight_format"] = "checkpoint≈BF16 / runtime FP8"
    return profile


def _gpu_compute_tops(gpu: dict[str, Any], precision: str) -> float:
    compute_tops = gpu.get("compute_tops") if isinstance(gpu.get("compute_tops"), dict) else {}
    return float(compute_tops.get(precision, gpu["bf16_tflops"]))


def _prefill_attention_flops(metrics: dict[str, Any], seq_len: int) -> int:
    qk_width = int(metrics.get("qk_width", 0) or 0)
    v_width = int(metrics.get("v_width", 0) or 0)
    if not qk_width and not v_width:
        return 0
    if metrics.get("is_deepseek_v4"):
        ratios = metrics.get("compress_ratios") or []
        window = int(metrics.get("sliding_window", 0) or 0)
        index_topk = int(metrics.get("index_topk", 0) or 0)
        visible_pairs = len(ratios) * _visible_token_pairs(seq_len, window)
        for ratio in ratios:
            if ratio == 4:
                visible_pairs += _compressed_visible_pairs(seq_len, ratio, index_topk)
            elif ratio > 0:
                visible_pairs += _compressed_visible_pairs(seq_len, ratio)
        core_flops = 2 * (qk_width + v_width) * visible_pairs
        index_width = int(metrics.get("index_n_heads", 0) or 0) * int(metrics.get("index_head_dim", 0) or 0)
        index_pairs = sum(ratio == 4 for ratio in ratios) * _compressed_visible_pairs(seq_len, 4)
        return core_flops + 2 * index_width * index_pairs
    full_layers = int(metrics.get("full_attention_layers", metrics.get("num_layers", 0)) or 0)
    sliding_layers = int(metrics.get("sliding_attention_layers", 0) or 0)
    sparse_layers = int(metrics.get("sparse_attention_layers", 0) or 0)
    sparse_topk = int(metrics.get("sparse_topk", 0) or 0)
    window = int(metrics.get("sliding_window", 0) or 0)
    visible_pairs = full_layers * _visible_token_pairs(seq_len)
    visible_pairs += sliding_layers * _visible_token_pairs(seq_len, window)
    visible_pairs += sparse_layers * _visible_token_pairs(seq_len, sparse_topk)
    core_flops = 2 * (qk_width + v_width) * visible_pairs
    index_width = int(metrics.get("index_n_heads", 0) or 0) * int(metrics.get("index_head_dim", 0) or 0)
    index_flops = 2 * index_width * sparse_layers * _visible_token_pairs(seq_len)
    return core_flops + index_flops


def estimate_memory_footprint(
    metrics: dict[str, Any],
    precision: str = "bf16",
    batch: int = 1,
    seq_len: int = 2048,
) -> dict[str, Any]:
    """Estimate inference VRAM footprint and how many of each GPU are needed.

    All figures are theoretical approximations:
      - weights  = total_params * bytes/param(precision)
      - kv_cache = kv_bytes_per_token * seq_len * batch (KV kept in bf16)
      - activation ~= batch * seq_len * hidden_size * workspace_factor * 2B
    """
    precision = precision if precision in PRECISION_BYTES else "bf16"
    batch = max(int(batch or 1), 1)
    seq_len = max(int(seq_len or 1), 1)

    total_params = metrics.get("total_params", 0) or 0
    hidden_size = metrics.get("hidden_size", 0) or 0

    checkpoint_bytes = int(metrics.get("checkpoint_bytes", 0) or 0)
    checkpoint_precision = str(metrics.get("checkpoint_precision") or "")
    use_checkpoint_bytes = checkpoint_bytes > 0 and precision == checkpoint_precision
    weights_bytes = checkpoint_bytes if use_checkpoint_bytes else total_params * PRECISION_BYTES[precision]
    kv_bytes = _kv_cache_bytes(metrics, seq_len, batch)
    activation_bytes = batch * seq_len * hidden_size * _ACT_FACTOR * 2
    total_bytes = weights_bytes + kv_bytes + activation_bytes

    gib = 1024 ** 3
    total_gb = total_bytes / gib
    gpu_fit = []
    for gpu in GPU_REFERENCE:
        usable = gpu["mem_gb"] * _VRAM_USABLE
        count = math.ceil(total_gb / usable) if usable > 0 else 0
        gpu_fit.append({"name": gpu["name"], "mem_gb": gpu["mem_gb"], "count": count})

    return {
        "precision": precision,
        "weight_format": metrics.get("weight_format", precision.upper()) if precision == checkpoint_precision else precision.upper(),
        "batch": batch,
        "seq_len": seq_len,
        "bytes_per_param": PRECISION_BYTES[precision],
        "checkpoint_bytes_per_param": weights_bytes / total_params if use_checkpoint_bytes and total_params else None,
        "weights_bytes": weights_bytes,
        "weight_source": "checkpoint" if use_checkpoint_bytes else "parameter_estimate",
        "kv_bytes": kv_bytes,
        "kv_bytes_per_token": metrics.get("kv_bytes_per_token", 0) or 0,
        "activation_bytes": activation_bytes,
        "activation_factor": _ACT_FACTOR,
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "gpu_fit": gpu_fit,
    }


def estimate_throughput(
    metrics: dict[str, Any],
    precision: str = "bf16",
    seq_len: int = 2048,
    batch: int = 1,
    gpu_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Estimate per-GPU decode throughput (tok/s) and prefill first-token latency.

    Decode is the smaller of the compute ceiling and the memory-bandwidth ceiling
    (decode is usually bandwidth bound). Prefill (TTFT) is compute bound.
    All values are theoretical upper-bound approximations.
    """
    precision = precision if precision in PRECISION_BYTES else "bf16"
    seq_len = max(int(seq_len or 1), 1)
    batch = max(int(batch or 1), 1)
    active_params = metrics.get("active_params", 0) or 0
    if active_params <= 0:
        return []

    bytes_per_param = PRECISION_BYTES[precision]
    use_native_profile = precision == str(metrics.get("checkpoint_precision") or "")
    compute_precision = str(metrics.get("native_compute_precision") or precision) if use_native_profile else precision
    linear_flops = int(metrics.get("linear_flops_per_token", 2 * active_params) or 0)
    attention_flops = 2 * (int(metrics.get("qk_width", 0) or 0) + int(metrics.get("v_width", 0) or 0)) * _context_layer_tokens(metrics, seq_len)
    attention_flops += _indexer_flops_per_token(metrics, seq_len)
    decode_flops = linear_flops + attention_flops
    prefill_flops = batch * (linear_flops * seq_len + _prefill_attention_flops(metrics, seq_len))
    native_active_weight_bytes = metrics.get("native_active_weight_bytes") if use_native_profile else None
    active_weight_bytes = float(native_active_weight_bytes) if isinstance(native_active_weight_bytes, (int, float)) else active_params * bytes_per_param
    active_bytes_per_param = active_weight_bytes / active_params
    kv_read_bytes = _kv_cache_bytes(metrics, seq_len, 1)
    bytes_per_output_token = active_weight_bytes / batch + kv_read_bytes
    gpu_counts = gpu_counts or {}
    rows = []
    for gpu in GPU_REFERENCE:
        gpu_count = max(int(gpu_counts.get(gpu["name"], 1) or 1), 1)
        per_gpu_tops = _gpu_compute_tops(gpu, compute_precision)
        effective_tops = per_gpu_tops * gpu_count
        peak_flops = effective_tops * 1e12 * _MFU
        bw = gpu["bw_gbs"] * 1e9 * gpu_count
        compute_tps = peak_flops / decode_flops
        bandwidth_tps = bw / bytes_per_output_token
        if bandwidth_tps <= compute_tps:
            decode_tps, bound = bandwidth_tps, "带宽"
        else:
            decode_tps, bound = compute_tps, "算力"
        ttft_ms = prefill_flops / peak_flops * 1000
        rows.append({
            "name": gpu["name"],
            "decode_tps": decode_tps,
            "ttft_ms": ttft_ms,
            "bound": bound,
            "effective_tops": effective_tops,
            "per_gpu_tops": per_gpu_tops,
            "gpu_count": gpu_count,
            "compute_tps": compute_tps,
            "bandwidth_tps": bandwidth_tps,
            "decode_flops": decode_flops,
            "attention_flops": attention_flops,
            "active_weight_bytes": active_weight_bytes,
            "active_bytes_per_param": active_bytes_per_param,
            "compute_precision": compute_precision,
            "kv_read_bytes": kv_read_bytes,
            "bytes_per_output_token": bytes_per_output_token,
            "batch": batch,
        })
    return rows


def build_llm_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    params_config = read_json_file(model_dir / "params.json")
    architecture = infer_architecture_name(model_dir, "llm")
    dims = parse_llm_dims(config, params_config)

    hidden_size = dims["hidden_size"]
    num_layers = dims["num_layers"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    ffn_hidden = dims["ffn_hidden"]
    vocab_size = dims["vocab_size"]
    max_position = int(first_defined(config.get("max_position_embeddings"), params_config.get("max_position_embeddings"), 4096) or 4096)
    sliding_window = dims["sliding_window"]

    num_experts = dims["num_experts"]
    experts_per_tok = dims["experts_per_tok"]
    moe_ffn_hidden = dims["moe_ffn_hidden"]
    n_shared_experts = dims["n_shared_experts"]
    first_k_dense = dims["first_k_dense"]
    tie_word_embeddings = dims["tie_word_embeddings"]
    quant = first_defined(
        config.get("quantization_config", {}).get("quant_method") if isinstance(config.get("quantization_config"), dict) else None,
        params_config.get("quantization", {}).get("qformat_weight") if isinstance(params_config.get("quantization"), dict) else None,
        config.get("torch_dtype"),
        params_config.get("dtype"),
    )
    rope = first_defined(
        config.get("rope_scaling", {}).get("type") if isinstance(config.get("rope_scaling"), dict) else None,
        params_config.get("yarn", {}).get("factor") if isinstance(params_config.get("yarn"), dict) else None,
        config.get("rope_theta"),
        params_config.get("rope_theta"),
    )

    # MLA detection
    q_lora_rank = dims["q_lora_rank"]
    kv_lora_rank = dims["kv_lora_rank"]
    qk_rope_head_dim = dims["qk_rope_head_dim"]
    qk_nope_head_dim = dims["qk_nope_head_dim"]
    qk_head_dim = dims["qk_head_dim"]
    v_head_dim = dims["v_head_dim"]
    o_lora_rank = int(first_defined(config.get("o_lora_rank"), 0) or 0)
    is_mla = dims["is_mla"]
    is_deepseek_v4 = dims["is_deepseek_v4"]

    batch = clamp_int(query.get("batch", [1])[0], 1)
    seq_len = clamp_int(query.get("seq_len", [min(max_position, 2048)])[0], min(max_position, 2048), maximum=max_position)

    hidden_shape = shape(batch, seq_len, hidden_size or "hidden")
    logits_shape = shape(batch, seq_len, vocab_size or "vocab")
    display_qk_head_dim = qk_head_dim if is_mla else head_dim
    attention_shape = shape(batch, seq_len, num_heads or "heads", display_qk_head_dim or "head_dim")
    kv_shape = (
        f"latent {shape(batch, seq_len, kv_lora_rank or head_dim or 'kv_lora')} + rope {shape(batch, seq_len, qk_rope_head_dim or 'rope_dim')}"
        if is_mla
        else shape(batch, seq_len, num_kv_heads or "kv_heads", head_dim or "head_dim")
    )
    score_shape = (
        shape(batch, num_heads or "heads", seq_len, f"local≤{sliding_window}+compressed")
        if is_deepseek_v4
        else shape(batch, num_heads or "heads", seq_len, seq_len)
    )
    rope_q_shape = f"Q_rope {attention_shape}; K_rope {kv_shape}"

    warnings = [
        "Shape 基于配置文件和当前输入参数推导，不是逐算子运行时真实张量。",
    ]
    if not hidden_size or not num_layers:
        warnings.append("该模型的层数或隐藏维度信息不完整，图中会保留摘要级展示。")
    if dims["linear_attention_layers"]:
        warnings.append("线性注意力层不套用标准 KV cache 与 QK/AV 二次项；其固定状态开销未计入。")
    if dims["sparse_attention_layers"]:
        warnings.append(f"稀疏注意力按 {dims['sparse_attention_layers']} 个 DSA 层计算低维全序列 indexer 与 top-{dims['sparse_topk']} 主注意力；KV cache 仍保留这些层的完整序列。")
    if is_deepseek_v4:
        warnings.append("DeepSeek V4 按共享 K=V、q 低秩、grouped-o 低秩及 CSA/HCA 压缩率计算参数、注意力 FLOPs 与 KV cache。")

    lanes = [
        ("inputs", "输入"),
        ("embedding", "嵌入"),
        ("core", "主干"),
        ("head", "输出头"),
        ("output", "输出"),
    ]

    core_badges = [f"{num_layers or '?'} 层", architecture]
    if num_experts:
        core_badges.append("MoE")
    if quant:
        core_badges.append(str(quant))

    attention_badges = [f"{num_heads or '?'} heads", f"head {head_dim or '?'}"]
    if num_kv_heads:
        attention_badges.append(f"kv {num_kv_heads}")

    ffn_badges = ["MoE" if num_experts else "Dense FFN", f"hidden {ffn_hidden or '?'}"]
    if num_experts:
        ffn_badges.append(f"top-{experts_per_tok}")

    repeat_nodes: list[dict[str, Any]] = []
    repeat_edges: list[dict[str, Any]] = []

    if num_layers <= 1:
        repeat_specs = [
            ("decoder_layer_1", 1, "Layer 1", "唯一的一层 decoder block", "该模型只有一个 decoder block，因此重复层摘要退化为单层视图。", 1, 1, 1),
        ]
    elif num_layers == 2:
        repeat_specs = [
            ("decoder_layer_1", 1, "Layer 1", "首层 decoder block", "表示第一个 decoder block 的摘要。", 1, 1, 1),
            ("decoder_layer_2", 2, f"Layer {num_layers}", "末层 decoder block", "表示最后一个 decoder block 的摘要。", num_layers, num_layers, 1),
        ]
    else:
        repeat_specs = [
            ("decoder_layer_1", 1, "Layer 1", "首层 decoder block", "表示第一个 decoder block 的摘要。", 1, 1, 1),
            (
                "decoder_layer_mid",
                2,
                f"Layers 2..{num_layers - 1}",
                f"中间 {num_layers - 2} 层的重复摘要",
                "把中间重复出现的大部分 decoder blocks 折叠为一个摘要节点。",
                2,
                num_layers - 1,
                num_layers - 2,
            ),
            ("decoder_layer_last", 3, f"Layer {num_layers}", "末层 decoder block", "表示最后一个 decoder block 的摘要。", num_layers, num_layers, 1),
        ]

    previous_repeat_node_id: str | None = None
    for node_id, order, label, subtitle, description, layer_start, layer_end, repeat_count in repeat_specs:
        if layer_start == 1 and layer_end == num_layers:
            layer_mask_note = "该模型只有一个 block，mask 规则只应用一次后直接进入输出头。"
        elif layer_start == 1:
            layer_mask_note = "首层首次在 embedding 输出上施加 mask 规则，定义后续层共享的可见范围。"
        elif layer_end == num_layers:
            layer_mask_note = "末层沿用相同的 mask 规则，最终可见上下文直接送入 LM Head。"
        else:
            layer_mask_note = f"中间 {repeat_count} 层重复相同的 mask 规则，不改变可见范围，只重复利用同一注意力约束。"

        mask_window_note = (
            f"启用 sliding window，每个 query 最多保留最近 {sliding_window} 个历史 token。"
            if sliding_window
            else "未启用 sliding window，causal mask 之后保留全部历史 token。"
        )

        repeat_nodes.append(
            build_node(
                node_id,
                "core",
                order,
                label,
                subtitle,
                description,
                hidden_shape,
                hidden_shape,
                [f"{repeat_count}x block", f"layers {layer_start}..{layer_end}" if layer_start != layer_end else f"layer {layer_start}"],
                [
                    detail("layer_start", layer_start),
                    detail("layer_end", layer_end),
                    detail("repeat_count", repeat_count),
                    detail("hidden_size", hidden_size),
                    detail("num_attention_heads", num_heads),
                ],
                [
                    section(
                        "块摘要",
                        [
                            detail("block input", hidden_shape),
                            detail("attention sub-block", attention_shape),
                            detail("ffn / moe sub-block", shape(batch, seq_len, ffn_hidden or "ffn")),
                            detail("block output", hidden_shape),
                        ],
                    ),
                    section(
                        "推导公式",
                        [
                            detail("per layer preserve", f"[B, T, H] -> block -> [B, T, H] = {hidden_shape}"),
                            detail("repeat summary", f"该节点代表 {repeat_count} 个 decoder blocks 的重复摘要"),
                            detail("layer span", f"layers {layer_start}..{layer_end}" if layer_start != layer_end else f"layer {layer_start}"),
                        ],
                    ),
                    section(
                        "Mask 摘要",
                        [
                            detail("causal rule", "始终阻止当前 token 看到未来位置。"),
                            detail("window rule", mask_window_note),
                            detail("layer role", layer_mask_note),
                        ],
                    ),
                ],
                "core",
                micro_flow=["Attention", "FFN / MoE", "Residual"],
                parent_id="decoder_stack",
                view_modes=["repeat"],
            )
        )

        if previous_repeat_node_id is None:
            repeat_edges.append(build_edge("token_embedding", node_id, "layer stream", ["repeat"]))
        else:
            repeat_edges.append(build_edge(previous_repeat_node_id, node_id, "next repeated block", ["repeat"]))
        previous_repeat_node_id = node_id

    if previous_repeat_node_id is not None:
        repeat_edges.append(build_edge(previous_repeat_node_id, "lm_head", "stack output", ["repeat"]))

    nodes = [
        build_node(
            "token_input",
            "inputs",
            0,
            "Token 输入",
            "离散 token ids",
            "语言模型接收的 token 序列。",
            shape(batch, seq_len),
            shape(batch, seq_len),
            ["text", f"ctx {max_position}"],
            [detail("batch", batch), detail("seq_len", seq_len), detail("max_position_embeddings", max_position)],
            [section("说明", [detail("输入语义", "token ids 进入 embedding lookup")])],
            "input",
        ),
        build_node(
            "token_embedding",
            "embedding",
            0,
            "Token Embedding",
            "词表查表与位置混合",
            "将 token ids 映射到隐藏向量空间。",
            shape(batch, seq_len),
            hidden_shape,
            [f"vocab {vocab_size or '?'}", f"hidden {hidden_size or '?'}"],
            [detail("vocab_size", vocab_size), detail("hidden_size", hidden_size)],
            [
                section("输出张量", [detail("embedding", hidden_shape)]),
                section("推导公式", [detail("lookup", f"[B, T] -> [B, T, H] = {hidden_shape}")]),
            ],
            "text",
            view_modes=["summary", "expanded", "repeat"],
        ),
        build_node(
            "decoder_stack",
            "core",
            0,
            "Decoder Stack",
            f"重复 {num_layers or '?'} 次的主干层",
            "表示一个典型 decoder-only block 的重复执行以及层间残差连接。",
            hidden_shape,
            hidden_shape,
            core_badges,
            [
                detail("num_hidden_layers", num_layers),
                detail("num_attention_heads", num_heads),
                detail("num_key_value_heads", num_kv_heads),
                detail("head_dim", head_dim),
                detail("ffn_hidden", ffn_hidden),
            ],
            [
                section(
                    "层内形状",
                    [
                        detail("hidden stream", hidden_shape),
                        detail("attention qkv", attention_shape),
                        detail("kv cache view", kv_shape),
                        detail("ffn / moe intermediate", shape(batch, seq_len, ffn_hidden or "ffn")),
                    ],
                ),
                section(
                    "结构要点",
                    [
                        detail("rope", rope),
                        detail("quantization", quant),
                        detail("experts", num_experts or "dense"),
                        detail("top-k experts", experts_per_tok or "-"),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("residual stream", f"[B, T, H] = {hidden_shape}"),
                        detail("attention view", f"[B, T, n_heads, head_dim] = {attention_shape}"),
                        detail("kv cache view", f"[B, T, kv_heads, head_dim] = {kv_shape}"),
                        detail(
                            "routing / ffn",
                            f"[B, T, top_k] = {shape(batch, seq_len, experts_per_tok or '-')} over {num_experts} experts" if num_experts else f"[B, T, ffn_hidden] = {shape(batch, seq_len, ffn_hidden or '?')}",
                        ),
                        detail("stack repeat", f"{num_layers or '?'} x block with output shape preserved"),
                    ],
                ),
            ],
            "core",
            micro_flow=[
                f"Self-Attention {num_heads or '?'} x {head_dim or '?'}",
                f"KV Heads {num_kv_heads or '?'}",
                f"{'MoE 路由 top-' + str(experts_per_tok) if num_experts else 'Dense FFN'}",
                "Residual + Norm",
            ],
            view_modes=["summary"],
        ),
        build_node(
            "decoder_q_proj",
            "core",
            1,
            "Q Projection",
            "query 投影",
            "把残差流映射为多头 query 视图。",
            hidden_shape,
            attention_shape,
            [f"{num_heads or '?'} heads", f"head {head_dim or '?'}", "Q"],
            [
                detail("num_hidden_layers", num_layers),
                detail("num_attention_heads", num_heads),
                detail("head_dim", head_dim),
            ],
            [
                section(
                    "子块形状",
                    [
                        detail("input hidden", hidden_shape),
                        detail("query view", attention_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("query reshape", f"[B, T, H] -> [B, T, n_heads, head_dim] = {attention_shape}"),
                        detail("block repeat", f"Q projection appears in each of {num_layers or '?'} layers"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Linear Q", "Reshape heads"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_k_proj",
            "core",
            2,
            "K Projection",
            "key 投影",
            "把残差流映射为 key 视图，通常与 KV heads 对齐。",
            hidden_shape,
            kv_shape,
            [f"kv {num_kv_heads or '?'}", f"head {head_dim or '?'}", "K"],
            [
                detail("num_hidden_layers", num_layers),
                detail("num_key_value_heads", num_kv_heads),
                detail("head_dim", head_dim),
            ],
            [
                section(
                    "子块形状",
                    [
                        detail("input hidden", hidden_shape),
                        detail("key view", kv_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("key reshape", f"[B, T, H] -> [B, T, kv_heads, head_dim] = {kv_shape}"),
                        detail("kv sharing", f"K 与 V 通常共享 {num_kv_heads or '?'} 个 KV heads"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Linear K", "KV cache view"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_v_proj",
            "core",
            3,
            "V Projection",
            "value 投影",
            "把残差流映射为 value 视图，并作为注意力加权聚合的值张量。",
            hidden_shape,
            kv_shape,
            [f"kv {num_kv_heads or '?'}", f"head {head_dim or '?'}", "V"],
            [detail("num_hidden_layers", num_layers), detail("num_key_value_heads", num_kv_heads), detail("head_dim", head_dim)],
            [
                section(
                    "子块形状",
                    [
                        detail("input hidden", hidden_shape),
                        detail("value view", kv_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("value reshape", f"[B, T, H] -> [B, T, kv_heads, head_dim] = {kv_shape}"),
                        detail("attention value", "V 将与 softmax(QK^T) 权重相乘后聚合"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Linear V", "Value cache view"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_qk_rope",
            "core",
            4,
            "RoPE / Positional QK",
            "对 Q / K 应用位置编码",
            "在进行打分前，对 Q 和 K 应用旋转位置编码或同类位置编码变换。",
            f"Q {attention_shape}; K {kv_shape}",
            rope_q_shape,
            ["RoPE", f"heads {num_heads or '?'}"],
            [detail("rope", rope), detail("q shape", attention_shape), detail("k shape", kv_shape)],
            [
                section(
                    "子块形状",
                    [
                        detail("q input", attention_shape),
                        detail("k input", kv_shape),
                        detail("encoded qk", rope_q_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("rope apply", f"RoPE(Q), RoPE(K) 保持张量阶不变 -> {rope_q_shape}"),
                        detail("position encoding", "该步骤注入相对或旋转位置关系，再进入 score matrix 计算"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Apply RoPE", "Position-aware Q/K"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_qk_score",
            "core",
            5,
            "QK Score",
            "query-key 打分",
            "对每个 query head 计算 QK^T 分数矩阵。",
            rope_q_shape,
            score_shape,
            [f"{num_heads or '?'} heads", "QK^T"],
            [detail("score_shape", score_shape), detail("seq_len", seq_len), detail("num_attention_heads", num_heads)],
            [
                section(
                    "子块形状",
                    [
                        detail("q input", rope_q_shape),
                        detail("score matrix", score_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("score build", f"RoPE(Q) x RoPE(K)^T -> [B, heads, T, T] = {score_shape}"),
                        detail("block repeat", f"score computation appears in each of {num_layers or '?'} layers"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Dot product", "Scale"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_causal_mask",
            "core",
            6,
            "Causal Mask",
            "阻止看到未来 token",
            "在 softmax 前先应用下三角 causal mask，确保 decoder 只能访问当前及历史位置。",
            score_shape,
            score_shape,
            ["causal", f"ctx {seq_len}"],
            [detail("mask_shape", score_shape), detail("causal_mask", True), detail("future_tokens", "masked")],
            [
                section(
                    "子块形状",
                    [
                        detail("score input", score_shape),
                        detail("causal-masked score", score_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("causal rule", "对未来位置加上 -inf，使 query 只能看见当前及过去 token"),
                        detail("masked score", f"score + causal_mask -> {score_shape}"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Lower-triangular mask", "Future blocked"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_sliding_mask",
            "core",
            7,
            "Sliding Window Mask",
            "限制局部历史窗口",
            "若模型启用 sliding window，则在 causal mask 之后进一步裁剪可见历史范围；否则该节点表示该约束未启用。",
            score_shape,
            score_shape,
            ["local window", f"window {sliding_window}" if sliding_window else "disabled"],
            [detail("mask_shape", score_shape), detail("sliding_window", sliding_window or "disabled"), detail("mode", "local attention" if sliding_window else "full history after causal mask")],
            [
                section(
                    "子块形状",
                    [
                        detail("causal-masked score", score_shape),
                        detail("window-masked score", score_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail(
                            "window rule",
                            f"每个 query 最多保留最近 {sliding_window} 个历史 token" if sliding_window else "当前配置未启用 sliding window，本层不改变 causal mask 结果",
                        ),
                        detail("masked score", f"causal_score + window_mask -> {score_shape}" if sliding_window else f"causal_score -> {score_shape}"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Apply local window" if sliding_window else "No-op window", "Pass to softmax"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_softmax",
            "core",
            8,
            "Softmax",
            "注意力权重归一化",
            "对 QK 分数矩阵按最后一个维度做 softmax，得到注意力权重。",
            score_shape,
            score_shape,
            ["attn weights", f"T={seq_len}"],
            [detail("weights_shape", score_shape), detail("normalize_dim", "last seq axis")],
            [
                section(
                    "子块形状",
                    [
                        detail("score input", score_shape),
                        detail("weights output", score_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("softmax", f"softmax(score, axis=-1) -> {score_shape}"),
                        detail("weight meaning", "每个 query 位置对所有 key 位置的归一化权重"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Normalize scores", "Attention weights"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_weighted_v",
            "core",
            9,
            "Weighted V",
            "注意力加权聚合",
            "用注意力权重对 V 做加权聚合，得到每个 head 的上下文向量。",
            f"W {score_shape}; V {kv_shape}",
            attention_shape,
            ["weighted sum", f"head {head_dim or '?'}"],
            [detail("weights", score_shape), detail("value", kv_shape), detail("context", attention_shape)],
            [
                section(
                    "子块形状",
                    [
                        detail("weights input", score_shape),
                        detail("value input", kv_shape),
                        detail("context output", attention_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("weighted sum", f"softmax(QK^T) x V -> [B, T, n_heads, head_dim] = {attention_shape}"),
                        detail("per head context", "每个 query head 产生一个上下文向量"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Matrix multiply", "Context vectors"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_out_proj",
            "core",
            10,
            "Output Projection",
            "concat heads -> hidden",
            "拼接各个注意力 head 的上下文后投影回隐藏维度。",
            attention_shape,
            hidden_shape,
            attention_badges,
            [detail("attention_output", hidden_shape), detail("num_attention_heads", num_heads), detail("head_dim", head_dim)],
            [
                section(
                    "子块形状",
                    [
                        detail("context input", attention_shape),
                        detail("output hidden", hidden_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("concat heads", f"[B, T, n_heads, head_dim] -> [B, T, H] = {hidden_shape}"),
                        detail("output projection", "线性投影回 residual stream 维度"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Concat heads", "Linear out"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_ffn",
            "core",
            11,
            "MoE / FFN",
            "每层内的前馈子块",
            "展示每层前馈网络，若模型为 MoE 则突出 routed experts 与 top-k 路由。",
            hidden_shape,
            hidden_shape,
            ffn_badges,
            [
                detail("ffn_hidden", ffn_hidden),
                detail("experts", num_experts or "dense"),
                detail("top-k experts", experts_per_tok or "-"),
            ],
            [
                section(
                    "子块形状",
                    [
                        detail("input hidden", hidden_shape),
                        detail("intermediate", shape(batch, seq_len, ffn_hidden or "ffn")),
                        detail("output hidden", hidden_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("dense ffn", f"[B, T, H] -> [B, T, ffn_hidden] -> [B, T, H] = {hidden_shape}"),
                        detail(
                            "moe routing",
                            f"[B, T, top_k] = {shape(batch, seq_len, experts_per_tok or '-')} across {num_experts} experts" if num_experts else "该模型未启用 routed experts",
                        ),
                    ],
                ),
            ],
            "core",
            micro_flow=["Gate / router" if num_experts else "Up projection", "Experts / FFN", "Down projection"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        build_node(
            "decoder_residual",
            "core",
            12,
            "Residual + Norm",
            "每层输出回写到残差流",
            "表示 attention / FFN 子块之后的残差叠加与归一化。",
            hidden_shape,
            hidden_shape,
            [f"{num_layers or '?'} repeats", "residual stream"],
            [detail("residual stream", hidden_shape), detail("norm", "RMSNorm / LayerNorm variant")],
            [
                section(
                    "子块形状",
                    [
                        detail("residual in", hidden_shape),
                        detail("residual out", hidden_shape),
                    ],
                ),
                section(
                    "推导公式",
                    [
                        detail("shape preserve", f"[B, T, H] + sub-block output -> [B, T, H] = {hidden_shape}"),
                        detail("stack handoff", "output becomes next block input or final LM head input"),
                    ],
                ),
            ],
            "core",
            micro_flow=["Residual add", "Normalization", "Pass to next layer"],
            parent_id="decoder_stack",
            view_modes=["block"],
        ),
        *repeat_nodes,
        build_node(
            "lm_head",
            "head",
            0,
            "LM Head",
            "隐藏态映射到词表",
            "将最终隐藏态投影到词表维度。",
            hidden_shape,
            logits_shape,
            ["projection", f"vocab {vocab_size or '?'}"],
            [detail("input_hidden", hidden_size), detail("vocab_size", vocab_size)],
            [
                section("输出张量", [detail("logits", logits_shape)]),
                section("推导公式", [detail("projection", f"[B, T, H] -> [B, T, vocab] = {logits_shape}")]),
            ],
            "head",
            view_modes=["summary", "expanded", "repeat"],
        ),
        build_node(
            "logits",
            "output",
            0,
            "Logits / Next Token",
            "每个位置的词表分布",
            "通常再经过 sampling 或 greedy decode 得到输出 token。",
            logits_shape,
            logits_shape,
            ["output"],
            [detail("logits", logits_shape)],
            [section("解码", [detail("common strategy", "greedy / top-k / top-p")])],
            "output",
        ),
    ]

    edges = [
        build_edge("token_input", "token_embedding", "lookup"),
        build_edge("token_embedding", "decoder_stack", "hidden stream", ["summary"]),
        build_edge("decoder_stack", "lm_head", "final hidden", ["summary"]),
        build_edge("token_embedding", "decoder_q_proj", "query branch", ["block"]),
        build_edge("token_embedding", "decoder_k_proj", "key branch", ["block"]),
        build_edge("token_embedding", "decoder_v_proj", "value branch", ["block"]),
        build_edge("decoder_q_proj", "decoder_qk_rope", "Q", ["block"]),
        build_edge("decoder_k_proj", "decoder_qk_rope", "K", ["block"]),
        build_edge("decoder_qk_rope", "decoder_qk_score", "positional QK", ["block"]),
        build_edge("decoder_qk_score", "decoder_causal_mask", "raw scores", ["block"]),
        build_edge("decoder_causal_mask", "decoder_sliding_mask", "causal scores", ["block"]),
        build_edge("decoder_sliding_mask", "decoder_softmax", "masked scores", ["block"]),
        build_edge("decoder_softmax", "decoder_weighted_v", "normalized weights", ["block"]),
        build_edge("decoder_v_proj", "decoder_weighted_v", "V", ["block"]),
        build_edge("decoder_weighted_v", "decoder_out_proj", "context", ["block"]),
        build_edge("decoder_out_proj", "decoder_ffn", "attention output", ["block"]),
        build_edge("decoder_ffn", "decoder_residual", "ffn output", ["block"]),
        build_edge("decoder_residual", "lm_head", "stack output", ["block"]),
        *repeat_edges,
        build_edge("lm_head", "logits", "vocab projection"),
    ]

    if is_mla:
        # Switch standard block nodes/edges to block_gqa so they're hidden in block mode
        for node in nodes:
            if node.get("viewModes") == ["block"]:
                node["viewModes"] = ["block_gqa"]
        edges = [e if "block" not in e.get("viewModes", []) else {**e, "viewModes": ["block_gqa"]} for e in edges]

        kv_compress_dim = kv_lora_rank or head_dim or 0
        mla_q_shape = shape(batch, seq_len, num_heads or "heads", qk_head_dim or head_dim or "head_dim")
        mla_kv_latent = shape(batch, seq_len, kv_compress_dim)
        mla_rope_key = shape(batch, seq_len, qk_rope_head_dim or "rope_dim")
        mla_cached_shape = f"latent {mla_kv_latent} + K_rope {mla_rope_key}"
        mla_v_shape = shape(batch, seq_len, num_heads or "heads", v_head_dim or "head_dim")

        mla_block_nodes = [
            build_node(
                "mla_q_proj", "core", 1, "Q Low-rank Projection",
                f"hidden → {q_lora_rank} → heads",
                "Q 通过低秩投影压缩再展开，减少参数量。",
                hidden_shape, mla_q_shape,
                [f"q_lora {q_lora_rank}", f"{num_heads or '?'} heads"],
                [detail("q_lora_rank", q_lora_rank), detail("qk_head_dim", qk_head_dim)],
                [section("MLA Q", [detail("down", f"{hidden_size} → {q_lora_rank}"), detail("up", f"{q_lora_rank} → {num_heads}×{qk_head_dim}")])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_kv_compress", "core", 2, "KV Compression",
                f"hidden → {kv_compress_dim} (latent)",
                "KV 压缩为低秩 latent，cache 只存压缩表示。",
                hidden_shape, mla_cached_shape,
                [f"kv_lora {kv_compress_dim}", f"rope {qk_rope_head_dim}"],
                [detail("kv_compress_dim", kv_compress_dim), detail("cached_rope_dim", qk_rope_head_dim)],
                [section("MLA KV", [detail("compress", f"{hidden_size} → latent {kv_compress_dim} + rope {qk_rope_head_dim}"), detail("cache_saving", f"vs GQA: K+V {num_kv_heads}×{head_dim}×2 → {kv_compress_dim + qk_rope_head_dim}")])],
                "scheduler", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_kv_decompress", "core", 3, "KV Decompression",
                f"latent → K_nope + K_rope + V",
                "从压缩 latent 还原 K（nope + rope）和 V。",
                mla_cached_shape, f"K {mla_q_shape}; V {mla_v_shape}",
                [f"nope {qk_nope_head_dim}", f"rope {qk_rope_head_dim}"],
                [detail("qk_nope_head_dim", qk_nope_head_dim), detail("qk_rope_head_dim", qk_rope_head_dim), detail("v_head_dim", v_head_dim)],
                [section("解压", [detail("K_nope", qk_nope_head_dim), detail("K_rope", qk_rope_head_dim), detail("V", v_head_dim)])],
                "scheduler", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_qk_rope", "core", 4, "RoPE Decoupling",
                "Q/K split: nope + rope",
                "RoPE 仅施加在 rope 维度，nope 维度保持不变。",
                f"{mla_q_shape} + K", mla_q_shape,
                ["RoPE", f"rope_dim {qk_rope_head_dim}"],
                [detail("qk_rope_head_dim", qk_rope_head_dim)],
                [section("解耦 RoPE", [detail("nope_part", f"{qk_nope_head_dim} dims, no RoPE"), detail("rope_part", f"{qk_rope_head_dim} dims, RoPE")])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_score", "core", 5, "Attention Score",
                "[Q_nope||Q_rope] × [K_nope||K_rope]",
                "拼接 nope 和 rope 维度后计算注意力分数。",
                mla_q_shape, shape(batch, num_heads or "heads", seq_len, seq_len),
                ["score", f"dim {qk_head_dim}"],
                [detail("Q_dim", qk_head_dim), detail("K_dim", qk_head_dim)],
                [section("Score", [detail("QK", f"[B, heads, T, {qk_head_dim}] × [B, heads, T, {qk_head_dim}]ᵀ")])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_softmax", "core", 6, "Causal + Softmax",
                "mask → normalize",
                "施加 causal mask 后 softmax 归一化。",
                shape(batch, num_heads or "heads", seq_len, seq_len), shape(batch, num_heads or "heads", seq_len, seq_len),
                ["causal", "softmax"],
                [],
                [section("Mask + Softmax", [detail("window", sliding_window or "full")])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_weighted_v", "core", 7, "Weighted V",
                "scores × V → context",
                "用 softmax 权重对 V 加权求和。",
                f"score × {mla_v_shape}", mla_v_shape,
                ["attn output", f"v_dim {v_head_dim}"],
                [detail("v_head_dim", v_head_dim)],
                [section("Attention Output", [detail("context", mla_v_shape)])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
            build_node(
                "mla_out_proj", "core", 8, "Output Low-rank",
                f"context → {o_lora_rank or 'hidden'} → hidden",
                "通过低秩投影输出到隐藏维度。",
                mla_v_shape, hidden_shape,
                [f"o_lora {o_lora_rank}" if o_lora_rank else "direct"],
                [detail("o_lora_rank", o_lora_rank)],
                [section("Output", [detail("down", f"→ {o_lora_rank}" if o_lora_rank else "direct"), detail("up", f"→ {hidden_size}")])],
                "core", parent_id="decoder_stack", view_modes=["block"],
            ),
        ]

        mla_block_edges = [
            build_edge("token_embedding", "mla_q_proj", "Q input", ["block"]),
            build_edge("token_embedding", "mla_kv_compress", "KV input", ["block"]),
            build_edge("mla_q_proj", "mla_qk_rope", "Q", ["block"]),
            build_edge("mla_kv_compress", "mla_kv_decompress", "decompress", ["block"]),
            build_edge("mla_kv_decompress", "mla_qk_rope", "K", ["block"]),
            build_edge("mla_kv_decompress", "mla_weighted_v", "V", ["block"]),
            build_edge("mla_qk_rope", "mla_score", "QK", ["block"]),
            build_edge("mla_score", "mla_softmax", "raw scores", ["block"]),
            build_edge("mla_softmax", "mla_weighted_v", "weights", ["block"]),
            build_edge("mla_weighted_v", "mla_out_proj", "context", ["block"]),
            build_edge("mla_out_proj", "decoder_ffn", "attn output", ["block"]),
        ]

        nodes.extend(mla_block_nodes)
        edges.extend(mla_block_edges)

    if not dims["has_output_head"]:
        warnings.append("该配置是 encoder-only 模型；不使用 causal mask、生成式 LM Head 或自回归 KV cache。")
        nodes = [node for node in nodes if node["id"] not in {"decoder_causal_mask", "decoder_sliding_mask"}]
        edges = [
            edge for edge in edges
            if edge["source"] not in {"decoder_causal_mask", "decoder_sliding_mask"}
            and edge["target"] not in {"decoder_causal_mask", "decoder_sliding_mask"}
        ]
        edges.append(build_edge("decoder_qk_score", "decoder_softmax", "bidirectional scores", ["block"]))
        for node in nodes:
            if node["id"] == "decoder_stack":
                node.update({
                    "label": "Encoder Stack",
                    "subtitle": f"{num_layers or '?'} 层双向编码主干",
                    "description": "双向 self-attention 与 FFN 对输入序列进行编码。",
                    "sections": [
                        section("编码流", [
                            detail("hidden stream", hidden_shape),
                            detail("attention", "bidirectional full attention"),
                            detail("ffn", shape(batch, seq_len, ffn_hidden or "ffn")),
                        ])
                    ],
                })
            elif node["id"].startswith("decoder_layer_"):
                node.update({
                    "description": "重复执行双向 self-attention、FFN、残差连接与归一化。",
                    "sections": [section("层内摘要", [
                        detail("hidden stream", hidden_shape),
                        detail("attention", "bidirectional"),
                        detail("ffn", shape(batch, seq_len, ffn_hidden or "ffn")),
                    ])],
                    "microFlow": ["Bidirectional Attention", "FFN", "Residual"],
                })
            elif node["id"] == "lm_head":
                node.update({
                    "label": "Encoder 输出",
                    "subtitle": "contextual hidden states",
                    "description": "输出每个 token 的上下文化隐藏表示，不执行词表投影。",
                    "inputShape": hidden_shape,
                    "outputShape": hidden_shape,
                    "badges": ["hidden states"],
                    "details": [detail("output", hidden_shape)],
                    "sections": [section("输出", [detail("last hidden state", hidden_shape)])],
                })
            elif node["id"] == "logits":
                node.update({
                    "label": "Embedding 输出",
                    "subtitle": "pooling / token embeddings",
                    "description": "根据任务选择 token 表示或池化后的句向量。",
                    "inputShape": hidden_shape,
                    "outputShape": shape(batch, hidden_size or "hidden"),
                    "details": [detail("token embeddings", hidden_shape), detail("pooled", shape(batch, hidden_size or "hidden"))],
                    "sections": [section("编码结果", [detail("last hidden state", hidden_shape)])],
                })
        for edge in edges:
            if edge["source"] == "decoder_stack" and edge["target"] == "lm_head":
                edge["label"] = "final hidden"
            elif edge["source"] == "lm_head" and edge["target"] == "logits":
                edge["label"] = "pool / select"

    sources: list[str] = []
    append_source_file(model_dir, sources, "config.json")
    append_source_file(model_dir, sources, "configuration.json")
    append_source_file(model_dir, sources, "generation_config.json")
    append_source_file(model_dir, sources, "params.json")

    summary = [
        detail("类型", "LLM"),
        detail("隐藏维度", hidden_size),
        detail("层数", num_layers),
        detail("注意力头", num_heads),
        detail("上下文长度", max_position),
        detail("量化", quant),
    ]
    if is_mla:
        summary.append(detail("注意力", f"MLA (q_lora={q_lora_rank}, kv_lora={kv_lora_rank or head_dim}, rope_dim={qk_rope_head_dim})"))
    elif is_deepseek_v4:
        ratio_counts = {ratio: dims["compress_ratios"].count(ratio) for ratio in sorted(set(dims["compress_ratios"]))}
        summary.append(detail("注意力", f"Compressed MQA {ratio_counts} / local {sliding_window}"))
    if dims["sparse_attention_layers"]:
        summary.append(detail("层型", f"DSA {dims['sparse_attention_layers']} / SWA {dims['sliding_attention_layers']} / top-{dims['sparse_topk']}"))
    if num_experts:
        effective_top_k = (experts_per_tok or 1) * dims["active_expert_multiplier"]
        cli_note = f" (top-{experts_per_tok} × CLI {dims['active_expert_multiplier']})" if dims["active_expert_multiplier"] > 1 else ""
        summary.append(detail("MoE", f"{num_experts} experts / active {effective_top_k}{cli_note}"))

    # Runtime estimation (shared with the test-suite via estimate_llm_metrics).
    metrics = estimate_llm_metrics(dims, seq_len)
    default_precision = infer_weight_precision(config, params_config)
    if metrics is not None:
        checkpoint_index = read_json_file(model_dir / "model.safetensors.index.json")
        checkpoint_size = checkpoint_index.get("metadata", {}).get("total_size") if isinstance(checkpoint_index.get("metadata"), dict) else None
        if isinstance(checkpoint_size, (int, float)) and checkpoint_size > 0:
            metrics["checkpoint_bytes"] = int(checkpoint_size)
            metrics["checkpoint_precision"] = default_precision
        metrics.update(infer_native_weight_profile(config, metrics, default_precision))
        total_params = metrics["total_params"]
        active_params = metrics["active_params"]
        summary.append(detail("参数量", f"{total_params / 1e9:.2f}B" + (f" (active {active_params / 1e9:.2f}B)" if num_experts else "")))
        if metrics["mtp_params"]:
            mtp_scope = "已计入总参数" if metrics["mtp_included"] else "辅助预测层，未计入主干总参数"
            summary.append(detail("MTP 参数", f"{metrics['mtp_params'] / 1e9:.2f}B（{mtp_scope}）"))

        kv_per_1k = metrics["kv_cache_mb_per_1k"]
        if kv_per_1k is not None:
            summary.append(detail("KV cache", f"{kv_per_1k:.1f} MiB / 1K tokens / batch"))

        metric_label = "Decode FLOPs" if dims["has_output_head"] else "Encoder FLOPs"
        summary.append(detail(metric_label, f"{metrics['gflops_per_token']:.1f} GFLOPs / token @ seq {seq_len}"))

    precision = str(query.get("precision", [default_precision])[0] or default_precision)
    if precision not in PRECISION_BYTES:
        precision = "bf16"

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 16, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "Token 长度", "type": "number", "value": seq_len, "min": 1, "max": max_position, "step": 1, "help": "输入 token 数"},
        {"name": "precision", "label": "权重精度", "type": "select", "value": precision, "options": list(PRECISION_BYTES.keys()), "help": "用于显存/成本与吞吐估算"},
    ]

    headline = (
        f"Decoder-only 语言模型，隐藏维度 {hidden_size or '?'}，共 {num_layers or '?'} 层。"
        if dims["has_output_head"]
        else f"Encoder-only 表征模型，隐藏维度 {hidden_size or '?'}，共 {num_layers or '?'} 层。"
    )
    payload = base_model_payload(
        model_id,
        "llm",
        architecture,
        headline,
        summary,
        controls,
        {"batch": batch, "seq_len": seq_len, "precision": precision},
        build_graph(lanes, nodes, edges),
        warnings,
        sources,
        "decoder_stack",
    )

    if metrics is not None:
        memory = estimate_memory_footprint(metrics, precision, batch, seq_len)
        gpu_counts = {item["name"]: item["count"] for item in memory["gpu_fit"]}
        payload["metrics"] = {
            **metrics,
            "is_moe": bool(num_experts),
            "formula_terms": {
                "hidden_size": dims["hidden_size"],
                "num_layers": dims["num_layers"],
                "num_heads": dims["num_heads"],
                "num_kv_heads": dims["num_kv_heads"],
                "head_dim": dims["head_dim"],
                "ffn_hidden": dims["ffn_hidden"],
                "vocab_size": dims["vocab_size"],
                "num_experts": dims["num_experts"],
                "experts_per_tok": dims["experts_per_tok"],
                "effective_experts_per_tok": metrics["effective_experts_per_tok"],
                "moe_ffn_hidden": dims["moe_ffn_hidden"],
                "n_shared_experts": dims["n_shared_experts"],
                "shared_ffn_hidden": dims["shared_ffn_hidden"],
                "first_k_dense": dims["first_k_dense"],
                "tie_word_embeddings": dims["tie_word_embeddings"],
                "has_output_head": dims["has_output_head"],
                "ffn_projection_count": dims["ffn_projection_count"],
                "is_mla": dims["is_mla"],
                "is_deepseek_v4": dims["is_deepseek_v4"],
                "compress_ratios": dims["compress_ratios"],
                "sparse_attention_layers": dims["sparse_attention_layers"],
                "sparse_topk": dims["sparse_topk"],
                "attn_per_layer": metrics["attn_params_per_layer"],
                "expert_per": metrics["expert_params"],
                "shared_expert_per": metrics["shared_expert_params"],
                "dense_ffn_per": metrics["dense_ffn_params"],
                "n_dense_layers": metrics["n_dense_layers"],
                "n_moe_layers": metrics["n_moe_layers"],
                "embed_params": metrics["embed_params"],
                "embed_total": metrics["embed_total"],
                "mtp_params": metrics["mtp_params"],
                "mtp_layers": metrics["mtp_layers"],
            },
            "memory": memory,
            "throughput": estimate_throughput(metrics, precision, seq_len, batch, gpu_counts) if dims["has_output_head"] else [],
            "gpu_reference": GPU_REFERENCE,
            "throughput_terms": {
                "active_params": active_params,
                "bytes_per_param": PRECISION_BYTES[precision],
                "precision": precision,
                "weight_format": metrics.get("weight_format", precision.upper()),
                "native_active_weight_bytes": metrics.get("native_active_weight_bytes"),
                "native_compute_precision": metrics.get("native_compute_precision", precision),
                "expert_bytes_per_param": metrics.get("expert_bytes_per_param", PRECISION_BYTES[precision]),
                "non_expert_bytes_per_param": metrics.get("non_expert_bytes_per_param", PRECISION_BYTES[precision]),
                "uses_native_profile": precision == default_precision and metrics.get("native_active_weight_bytes") is not None,
                "seq_len": seq_len,
                "batch": batch,
                "mfu": _MFU,
                "two": 2,
            },
        }

    return payload


def build_nemotron_streaming_asr_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    chunk_options = [80, 160, 320, 560, 1120]
    right_context_by_chunk = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}
    try:
        requested_chunk = int(query.get("chunk_ms", [1120])[0])
    except (TypeError, ValueError):
        requested_chunk = 1120
    chunk_ms = requested_chunk if requested_chunk in chunk_options else 1120
    chunks = clamp_int(query.get("chunks", [1])[0], 1, maximum=10000)
    language_mode = str(query.get("language_mode", ["provided"])[0]).lower()
    if language_mode not in {"provided", "auto"}:
        language_mode = "provided"
    batch = clamp_int(query.get("batch", [1])[0], 1, maximum=256)
    base_frame_ms = 80
    left_context_frames = 56
    right_context_frames = right_context_by_chunk[chunk_ms]
    chunk_frames = 1 + right_context_frames
    left_cache_ms = left_context_frames * base_frame_ms
    total_audio_ms = chunks * chunk_ms
    encoder_layers = 24
    encoder_hidden = 512
    language_dim = 128
    fused_hidden = encoder_hidden + language_dim

    lanes = [("inputs", "输入"), ("features", "特征"), ("encoder", "流式编码"), ("decoder", "转录解码"), ("output", "输出")]
    nodes = [
        build_node("stream_audio", "inputs", 0, "流式音频", f"{chunk_ms} ms non-overlap chunks", "每次只输入新 chunk，不重复计算历史重叠窗口。", shape(batch, chunks, "waveform_chunk"), shape(batch, chunks, "waveform_chunk"), [f"chunks {chunks}", f"total {total_audio_ms / 1000:.2f}s"], [detail("chunk_ms", chunk_ms), detail("chunks", chunks), detail("total_audio_ms", total_audio_ms)], [section("流式输入", [detail("total duration", f"{chunks} * {chunk_ms}ms = {total_audio_ms}ms")])], "input"),
        build_node("language_prompt", "inputs", 1, "语言 ID 提示", "provided / auto detect", "将语言身份编码为 128 维 one-hot，并广播到每个声学时间步。", shape(batch, language_dim), shape(batch, chunk_frames, language_dim), [f"mode {language_mode}", f"dim {language_dim}"], [detail("language_dim", language_dim), detail("broadcast_frames", chunk_frames)], [section("输出", [detail("language features", shape(batch, chunk_frames, language_dim))])], "text"),
        build_node("stream_features", "features", 0, "流式声学特征", f"{base_frame_ms} ms encoder frames", "将当前 chunk 转换为 FastConformer 输入特征；模型卡未公开 mel bins，因此该维保持符号。", shape(batch, "waveform_chunk"), shape(batch, chunk_frames, "feature_bins"), [f"frames {chunk_frames}", f"right ctx {right_context_frames}"], [detail("base_frame_ms", base_frame_ms), detail("current_frames", 1), detail("right_context_frames", right_context_frames), detail("chunk_frames", chunk_frames)], [section("Chunk 推导", [detail("frames", f"1 current + {right_context_frames} right = {chunk_frames}"), detail("duration", f"{chunk_frames} * {base_frame_ms}ms = {chunk_ms}ms")])], "audio"),
        build_node("cache_aware_encoder", "encoder", 0, "Cache-Aware FastConformer", f"{encoder_layers} layers / D={encoder_hidden}", "所有自注意力与卷积层复用历史缓存，只编码当前非重叠 chunk。", shape(batch, chunk_frames, "feature_bins"), shape(batch, chunk_frames, encoder_hidden), [f"layers {encoder_layers}", f"left cache {left_context_frames}"], [detail("encoder_layers", encoder_layers), detail("encoder_hidden", encoder_hidden), detail("left_context_frames", left_context_frames), detail("left_cache_ms", left_cache_ms), detail("right_context_frames", right_context_frames)], [section("缓存窗口", [detail("left cache", f"{left_context_frames} * {base_frame_ms}ms = {left_cache_ms}ms"), detail("new encoder output", shape(batch, chunk_frames, encoder_hidden))])], "core"),
        build_node("prompt_fusion", "encoder", 1, "语言条件融合", f"{encoder_hidden} + {language_dim} = {fused_hidden}", "拼接声学表示与逐帧语言编码，再投影到 RNN-T 解码器。", shape(batch, chunk_frames, fused_hidden), shape(batch, chunk_frames, "rnnt_joint_dim"), [f"fused {fused_hidden}"], [detail("acoustic_dim", encoder_hidden), detail("language_dim", language_dim), detail("fused_dim", fused_hidden)], [section("维度推导", [detail("concat", f"{encoder_hidden} + {language_dim} = {fused_hidden}")])], "fusion"),
        build_node("rnnt_decoder", "decoder", 0, "RNN-T 解码器", "streaming transducer", "联合声学帧与预测网络状态，增量输出带标点和大小写的文本 token。", shape(batch, chunk_frames, "rnnt_joint_dim"), shape(batch, "emitted_tokens"), ["RNNT", "incremental"], [detail("encoder_frames", chunk_frames), detail("output_tokens", "data-dependent")], [section("输出", [detail("token stream", shape(batch, "emitted_tokens"))])], "head"),
        build_node("transcript", "output", 0, "多语言转录", "text + optional language tag", "输出文本，并按设置保留或移除自动语言标签。", shape(batch, "emitted_tokens"), shape(batch, "transcript"), ["40 language-locales", f"lang {language_mode}"], [detail("language_mode", language_mode), detail("punctuation", True), detail("capitalization", True)], [section("结果", [detail("transcript", shape(batch, "transcript"))])], "output"),
    ]
    edges = [
        build_edge("stream_audio", "stream_features", "new non-overlap chunk"),
        build_edge("stream_features", "cache_aware_encoder", "acoustic features"),
        build_edge("cache_aware_encoder", "prompt_fusion", "D=512 embeddings"),
        build_edge("language_prompt", "prompt_fusion", "K=128 language code"),
        build_edge("prompt_fusion", "rnnt_decoder", "projected joint features"),
        build_edge("rnnt_decoder", "transcript", "incremental tokens"),
    ]
    sources: list[str] = []
    append_source_file(model_dir, sources, "README.md")
    return base_model_payload(
        model_id,
        "multimodal",
        "FastConformerCacheAwareRNNT",
        f"流式 ASR 每个 {chunk_ms}ms chunk 包含 {chunk_frames} 个 80ms 帧，并复用 {left_cache_ms / 1000:.2f}s 左侧编码缓存。",
        [
            detail("类型", "流式语音识别"),
            detail("参数量", "约 600M"),
            detail("编码器", f"{encoder_layers} 层 FastConformer / D={encoder_hidden}"),
            detail("当前 chunk", f"{chunk_ms} ms / {chunk_frames} frames"),
            detail("左侧缓存", f"{left_context_frames} frames / {left_cache_ms / 1000:.2f}s"),
            detail("语言提示", f"{language_dim} dim / {language_mode}"),
        ],
        [
            {"name": "batch", "label": "并行流", "type": "number", "value": batch, "min": 1, "max": 256, "step": 1, "help": "并行实时音频流数量"},
            {"name": "chunk_ms", "label": "Chunk 时长", "type": "select", "value": chunk_ms, "options": chunk_options, "help": "模型卡公开支持的非重叠 chunk 时长"},
            {"name": "chunks", "label": "Chunk 数", "type": "number", "value": chunks, "min": 1, "max": 10000, "step": 1, "help": "连续处理的 chunk 数量"},
            {"name": "language_mode", "label": "语言模式", "type": "select", "value": language_mode, "options": ["provided", "auto"], "help": "显式语言 ID 或自动检测"},
        ],
        {
            "batch": batch,
            "chunk_ms": chunk_ms,
            "chunks": chunks,
            "language_mode": language_mode,
            "base_frame_ms": base_frame_ms,
            "chunk_frames": chunk_frames,
            "right_context_frames": right_context_frames,
            "left_context_frames": left_context_frames,
            "left_cache_ms": left_cache_ms,
            "total_audio_ms": total_audio_ms,
        },
        build_graph(lanes, nodes, edges),
        [
            "模型目录未包含 .nemo 配置；24 层、D=512、语言向量 K=128 与 chunk/cache 映射均来自随仓库保存的模型卡。",
            "模型卡未公开前端 mel bins、RNN-T joint 维度和 tokenizer 大小，因此这些 shape 保持符号值。",
        ],
        sources,
        "cache_aware_encoder",
    )


def build_asr_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    yaml_config = supplemental_yaml_config(model_dir)
    preprocessor = read_json_file(model_dir / "preprocessor_config.json")
    architecture = infer_architecture_name(model_dir, "multimodal")

    thinker_config = config.get("thinker_config") if isinstance(config.get("thinker_config"), dict) else {}
    thinker_audio = thinker_config.get("audio_config") if isinstance(thinker_config.get("audio_config"), dict) else {}
    config_audio = config.get("audio_config") if isinstance(config.get("audio_config"), dict) else {}
    config_encoder = config.get("encoder") if isinstance(config.get("encoder"), dict) else {}
    yaml_encoder = first_defined(yaml_config.get("audio_encoder_conf"), yaml_config.get("encoder_conf"), {}) or {}
    encoder_config = first_defined(config_encoder, thinker_audio, config_audio, yaml_encoder, {}) or {}

    transformer_decoder = config.get("transf_decoder") if isinstance(config.get("transf_decoder"), dict) else {}
    transformer_decoder_config = transformer_decoder.get("config_dict") if isinstance(transformer_decoder.get("config_dict"), dict) else {}
    thinker_text = thinker_config.get("text_config") if isinstance(thinker_config.get("text_config"), dict) else {}
    config_text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    language_config = config.get("language_config") if isinstance(config.get("language_config"), dict) else {}
    decoder_config = first_defined(thinker_text, config_text, language_config, transformer_decoder_config, {}) or {}
    adaptor_config = yaml_config.get("audio_adaptor_conf") if isinstance(yaml_config.get("audio_adaptor_conf"), dict) else {}
    frontend_config = yaml_config.get("frontend_conf") if isinstance(yaml_config.get("frontend_conf"), dict) else {}
    dataset_config = yaml_config.get("dataset_conf") if isinstance(yaml_config.get("dataset_conf"), dict) else {}

    encoder_hidden = scalar_int(
        first_defined(encoder_config.get("d_model"), encoder_config.get("hidden_size"), encoder_config.get("output_size")),
        512,
    )
    encoder_output_hidden = scalar_int(first_defined(encoder_config.get("output_dim"), encoder_config.get("output_size"), encoder_hidden), encoder_hidden)
    encoder_layers = scalar_int(
        first_defined(encoder_config.get("n_layers"), encoder_config.get("encoder_layers"), encoder_config.get("num_hidden_layers"), encoder_config.get("num_blocks")),
        1,
    )
    encoder_heads = scalar_int(
        first_defined(encoder_config.get("n_heads"), encoder_config.get("encoder_attention_heads"), encoder_config.get("num_attention_heads"), encoder_config.get("attention_heads")),
        1,
    )
    decoder_hidden = scalar_int(
        first_defined(decoder_config.get("hidden_size"), adaptor_config.get("llm_dim"), config.get("head", {}).get("hidden_size") if isinstance(config.get("head"), dict) else None, encoder_output_hidden),
        encoder_output_hidden,
    )
    decoder_layers = scalar_int(first_defined(decoder_config.get("num_hidden_layers"), decoder_config.get("num_layers")), 0)
    decoder_heads = scalar_int(first_defined(decoder_config.get("num_attention_heads"), decoder_config.get("n_heads")), 0)
    vocab_size = scalar_int(decoder_config.get("vocab_size"), 0)
    if not vocab_size:
        vocab_size = scalar_int(first_defined(config.get("vocab_size"), config.get("head", {}).get("num_classes") if isinstance(config.get("head"), dict) else None), 0)
    identity = _model_identity(config, yaml_config)
    has_decoder = bool(decoder_config or yaml_config.get("llm") or "conditionalgeneration" in identity)

    inferred_sample_rate = None
    chunk_length = first_defined(preprocessor.get("chunk_length"), config.get("max_audio_clip_s"))
    if isinstance(preprocessor.get("n_samples"), (int, float)) and isinstance(chunk_length, (int, float)) and chunk_length > 0:
        inferred_sample_rate = int(preprocessor["n_samples"] / chunk_length)
    sample_rate = scalar_int(first_defined(preprocessor.get("sampling_rate"), config.get("sample_rate"), frontend_config.get("fs"), inferred_sample_rate), 16000)
    frame_shift_ms = frontend_config.get("frame_shift")
    inferred_hop = int(sample_rate * float(frame_shift_ms) / 1000) if isinstance(frame_shift_ms, (int, float)) and frame_shift_ms > 0 else None
    hop_length = scalar_int(first_defined(preprocessor.get("hop_length"), preprocessor.get("n_window_stride"), inferred_hop), max(sample_rate // 100, 1))
    feature_size = scalar_int(
        first_defined(preprocessor.get("feature_size"), thinker_audio.get("num_mel_bins"), config_audio.get("num_mel_bins"), encoder_config.get("feat_in"), frontend_config.get("n_mels")),
        80,
    )

    ratio_subsampling = None
    max_feature_frames = preprocessor.get("nb_max_frames")
    max_source_positions = thinker_audio.get("max_source_positions")
    if isinstance(max_feature_frames, (int, float)) and isinstance(max_source_positions, (int, float)) and max_source_positions > 0:
        ratio_subsampling = ceil_div(int(max_feature_frames), int(max_source_positions))
    subsampling_factor = scalar_int(
        first_defined(
            encoder_config.get("subsampling_factor"),
            encoder_config.get("downsample_rate"),
            dataset_config.get("audio_encoder_downsample_rate"),
            frontend_config.get("lfr_n"),
            ratio_subsampling,
        ),
        1,
    )

    max_audio_seconds = scalar_int(first_defined(config.get("max_audio_clip_s"), preprocessor.get("chunk_length")), 30)
    batch = clamp_int(query.get("batch", [1])[0], 1, maximum=16)
    audio_seconds = clamp_int(query.get("audio_seconds", [min(max_audio_seconds, 30)])[0], min(max_audio_seconds, 30), maximum=max(max_audio_seconds, 1))
    sample_count = audio_seconds * sample_rate
    feature_frames = ceil_div(sample_count, hop_length)
    audio_tokens = ceil_div(feature_frames, subsampling_factor)
    max_output_tokens = scalar_int(
        first_defined(config.get("max_seq_len"), decoder_config.get("max_sequence_length"), decoder_config.get("max_position_embeddings")),
        1024,
    )
    output_tokens = clamp_int(query.get("output_tokens", [min(max_output_tokens, 256)])[0], min(max_output_tokens, 256), maximum=max_output_tokens) if has_decoder else 0
    decoder_context_tokens = audio_tokens + output_tokens
    projected_hidden = decoder_hidden if has_decoder else encoder_output_hidden

    lanes = [("inputs", "输入"), ("processing", "处理"), ("encoding", "编码"), ("backbone", "主干"), ("output", "输出")]
    nodes = [
        build_node(
            "audio_input",
            "inputs",
            0,
            "音频输入",
            f"{sample_rate} Hz waveform",
            "原始单声道波形。",
            shape(batch, sample_count),
            shape(batch, sample_count),
            [f"{audio_seconds}s", f"{sample_rate} Hz"],
            [detail("samples", sample_count), detail("duration_seconds", audio_seconds), detail("sample_rate", sample_rate)],
            [section("输入", [detail("waveform", shape(batch, sample_count))])],
            "input",
        ),
        build_node(
            "audio_features",
            "processing",
            0,
            "声学特征",
            "STFT / log-mel",
            "按 hop length 提取声学帧。",
            shape(batch, sample_count),
            shape(batch, feature_frames, feature_size),
            [f"frames {feature_frames}", f"mel {feature_size}"],
            [detail("hop_length", hop_length), detail("feature_frames", feature_frames), detail("feature_size", feature_size)],
            [section("推导公式", [detail("feature frames", f"ceil({sample_count}/{hop_length}) = {feature_frames}")])],
            "audio",
        ),
        build_node(
            "audio_encoder",
            "encoding",
            0,
            "音频编码器",
            f"{encoder_layers} layers / {encoder_heads} heads",
            "声学帧经编码器与下采样变为音频 token。",
            shape(batch, feature_frames, feature_size),
            shape(batch, audio_tokens, encoder_output_hidden),
            [f"hidden {encoder_hidden}", f"tokens {audio_tokens}"],
            [detail("encoder_hidden", encoder_hidden), detail("output_hidden", encoder_output_hidden), detail("layers", encoder_layers), detail("heads", encoder_heads), detail("subsampling_factor", subsampling_factor)],
            [section("推导公式", [detail("audio tokens", f"ceil({feature_frames}/{subsampling_factor}) = {audio_tokens}")])],
            "audio",
        ),
    ]
    edges = [build_edge("audio_input", "audio_features", "waveform"), build_edge("audio_features", "audio_encoder", "mel frames")]

    decoder_source = "audio_encoder"
    if has_decoder and encoder_output_hidden != decoder_hidden:
        nodes.append(
            build_node(
                "audio_projector",
                "encoding",
                1,
                "音频投影",
                "audio -> language hidden",
                "将音频编码维度映射到文本解码器维度。",
                shape(batch, audio_tokens, encoder_output_hidden),
                shape(batch, audio_tokens, decoder_hidden),
                [f"to {decoder_hidden}"],
                [detail("input_hidden", encoder_output_hidden), detail("output_hidden", decoder_hidden)],
                [section("投影", [detail("output", shape(batch, audio_tokens, decoder_hidden))])],
                "fusion",
            )
        )
        edges.append(build_edge("audio_encoder", "audio_projector", "audio hidden"))
        decoder_source = "audio_projector"

    if has_decoder:
        nodes.extend(
            [
                build_node(
                    "text_decoder",
                    "backbone",
                    0,
                    "文本解码器",
                    f"{decoder_layers or '?'} layers / {decoder_heads or '?'} heads",
                    "在音频条件上自回归生成转写 token。",
                    shape(batch, decoder_context_tokens, decoder_hidden),
                    shape(batch, output_tokens, decoder_hidden),
                    [f"hidden {decoder_hidden}", f"output {output_tokens}"],
                    [detail("audio_tokens", audio_tokens), detail("output_tokens", output_tokens), detail("context_tokens", decoder_context_tokens)],
                    [section("上下文", [detail("audio + generated", f"{audio_tokens} + {output_tokens} = {decoder_context_tokens}")])],
                    "core",
                ),
                build_node(
                    "asr_logits",
                    "output",
                    0,
                    "转写输出",
                    "token logits",
                    "输出文本词表上的转写分布。",
                    shape(batch, output_tokens, decoder_hidden),
                    shape(batch, output_tokens, vocab_size or "vocab"),
                    [f"vocab {vocab_size or '?'}"],
                    [detail("logits", shape(batch, output_tokens, vocab_size or "vocab"))],
                    [section("输出", [detail("token ids", shape(batch, output_tokens))])],
                    "output",
                ),
            ]
        )
        edges.extend([build_edge(decoder_source, "text_decoder", "audio tokens"), build_edge("text_decoder", "asr_logits", "decoder hidden")])
    else:
        nodes.append(
            build_node(
                "ctc_output",
                "output",
                0,
                "CTC / 标签输出",
                "frame-level predictions",
                "编码器直接输出逐帧标签或 CTC 分布。",
                shape(batch, audio_tokens, projected_hidden),
                shape(batch, audio_tokens, vocab_size or "classes"),
                ["encoder-only"],
                [detail("time_steps", audio_tokens), detail("classes", vocab_size or "config 未提供")],
                [section("输出", [detail("frame logits", shape(batch, audio_tokens, vocab_size or "classes"))])],
                "output",
            )
        )
        edges.append(build_edge("audio_encoder", "ctc_output", "encoded frames"))

    summary = [
        detail("类型", "语音识别"),
        detail("采样率", f"{sample_rate} Hz"),
        detail("声学编码维度", encoder_hidden),
        detail("编码层数", encoder_layers),
        detail("当前声学帧", feature_frames),
        detail("当前音频 tokens", audio_tokens),
    ]
    if has_decoder:
        summary.extend([detail("文本隐藏维度", decoder_hidden), detail("输出 tokens", output_tokens)])
    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 16, "step": 1, "help": "并行音频条数"},
        {"name": "audio_seconds", "label": "音频秒数", "type": "number", "value": audio_seconds, "min": 1, "max": max_audio_seconds, "step": 1, "help": "当前输入音频时长"},
    ]
    if has_decoder:
        controls.append({"name": "output_tokens", "label": "输出 token", "type": "number", "value": output_tokens, "min": 1, "max": max_output_tokens, "step": 1, "help": "预计生成的转写长度"})

    sources: list[str] = []
    for source_name in ("config.json", "configuration.json", "config.yaml", "preprocessor_config.json", "processor_config.json"):
        append_source_file(model_dir, sources, source_name)
    warnings = ["声学帧按 ceil(samples / hop_length) 估算；实际 STFT 的 padding 与边界策略可能造成少量帧差异。"]
    return base_model_payload(
        model_id,
        "multimodal",
        architecture,
        f"语音识别模型，{audio_seconds} 秒音频约形成 {audio_tokens} 个编码 token。",
        summary,
        controls,
        {
            "batch": batch,
            "audio_seconds": audio_seconds,
            "sample_rate": sample_rate,
            "sample_count": sample_count,
            "hop_length": hop_length,
            "feature_frames": feature_frames,
            "subsampling_factor": subsampling_factor,
            "audio_tokens": audio_tokens,
            "output_tokens": output_tokens,
        },
        build_graph(lanes, nodes, edges),
        warnings,
        sources,
        "audio_encoder",
    )


def build_sam_video_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    processor = read_json_file(model_dir / "processor_config.json")
    architecture = infer_architecture_name(model_dir, "multimodal")
    detector = config.get("detector_config") if isinstance(config.get("detector_config"), dict) else {}
    tracker = config.get("tracker_config") if isinstance(config.get("tracker_config"), dict) else {}
    vision_config = first_defined(detector.get("vision_config"), tracker.get("vision_config"), {}) or {}
    backbone_config = vision_config.get("backbone_config") if isinstance(vision_config.get("backbone_config"), dict) else vision_config
    text_config = detector.get("text_config") if isinstance(detector.get("text_config"), dict) else {}
    image_processor = processor.get("image_processor") if isinstance(processor.get("image_processor"), dict) else processor
    video_processor = processor.get("video_processor") if isinstance(processor.get("video_processor"), dict) else {}

    patch_size = scalar_int(first_defined(backbone_config.get("patch_size"), image_processor.get("patch_size")), 16)
    vision_hidden = scalar_int(backbone_config.get("hidden_size"), 1024)
    vision_layers = scalar_int(backbone_config.get("num_hidden_layers"), 1)
    vision_heads = scalar_int(backbone_config.get("num_attention_heads"), 1)
    fpn_hidden = scalar_int(vision_config.get("fpn_hidden_size"), 256)
    text_hidden = scalar_int(text_config.get("hidden_size"), 0)
    text_layers = scalar_int(text_config.get("num_hidden_layers"), 0)
    default_height, default_width = infer_default_image_size(config, backbone_config, image_processor)
    target_size = processor.get("target_size")
    if isinstance(target_size, (int, float)):
        default_height = default_width = clamp_int(target_size, default_height)
    mask_size_config = image_processor.get("mask_size") if isinstance(image_processor.get("mask_size"), dict) else {}
    mask_height = scalar_int(first_defined(mask_size_config.get("height"), config.get("low_res_mask_size")), 256)
    mask_width = scalar_int(first_defined(mask_size_config.get("width"), config.get("low_res_mask_size")), mask_height)

    batch = clamp_int(query.get("batch", [1])[0], 1, maximum=8)
    frames = clamp_int(query.get("frames", [video_processor.get("num_frames") or 8])[0], video_processor.get("num_frames") or 8, maximum=1024)
    image_height = clamp_int(query.get("image_height", [default_height])[0], default_height, maximum=4096)
    image_width = clamp_int(query.get("image_width", [default_width])[0], default_width, maximum=4096)
    object_count = clamp_int(query.get("objects", [1])[0], 1, maximum=256)
    prompt_tokens = clamp_int(query.get("prompt_tokens", [16])[0], 16, maximum=512)
    patch_rows = ceil_div(image_height, patch_size)
    patch_cols = ceil_div(image_width, patch_size)
    tokens_per_frame = patch_rows * patch_cols
    total_vision_tokens = frames * tokens_per_frame
    patch_width = 3 * patch_size * patch_size

    lanes = [("inputs", "输入"), ("processing", "处理"), ("encoding", "编码"), ("tracking", "检测与跟踪"), ("output", "输出")]
    nodes = [
        build_node("video_input", "inputs", 0, "视频输入", "frames x RGB", "待检测与分割的视频帧。", shape(batch, frames, 3, image_height, image_width), shape(batch, frames, 3, image_height, image_width), [f"frames {frames}", f"{image_height}x{image_width}"], [detail("frames", frames), detail("height", image_height), detail("width", image_width)], [section("输入", [detail("video", shape(batch, frames, 3, image_height, image_width))])], "input"),
        build_node("prompt_input", "inputs", 1, "提示输入", "text / points / masks", "文本、点、框或已有 mask 提示。", shape(batch, prompt_tokens), shape(batch, prompt_tokens), [f"prompts {prompt_tokens}", f"objects {object_count}"], [detail("prompt_tokens", prompt_tokens), detail("objects", object_count)], [section("提示", [detail("prompt sequence", shape(batch, prompt_tokens))])], "text"),
        build_node("frame_processor", "processing", 0, "帧预处理", "resize / normalize / patch", "逐帧缩放并切分视觉 patch。", shape(batch, frames, 3, image_height, image_width), shape(batch, frames, tokens_per_frame, patch_width), [f"patch {patch_size}", f"tokens/frame {tokens_per_frame}"], [detail("patch_rows", patch_rows), detail("patch_cols", patch_cols), detail("tokens_per_frame", tokens_per_frame), detail("total_vision_tokens", total_vision_tokens)], [section("推导公式", [detail("tokens/frame", f"ceil({image_height}/{patch_size}) * ceil({image_width}/{patch_size}) = {tokens_per_frame}"), detail("all frames", f"{frames} * {tokens_per_frame} = {total_vision_tokens}")])], "vision"),
        build_node("vision_encoder", "encoding", 0, "视觉主干", f"{vision_layers} layers / {vision_heads} heads", "逐帧提取视觉特征。", shape(batch, frames, tokens_per_frame, patch_width), shape(batch, frames, tokens_per_frame, vision_hidden), [f"hidden {vision_hidden}", f"layers {vision_layers}"], [detail("vision_hidden", vision_hidden), detail("layers", vision_layers), detail("heads", vision_heads)], [section("输出", [detail("frame features", shape(batch, frames, tokens_per_frame, vision_hidden))])], "vision"),
        build_node("feature_pyramid", "encoding", 1, "特征金字塔", "multi-scale FPN", "将主干特征映射到检测与 mask 解码宽度。", shape(batch, frames, tokens_per_frame, vision_hidden), shape(batch, frames, tokens_per_frame, fpn_hidden), [f"fpn {fpn_hidden}"], [detail("fpn_hidden", fpn_hidden), detail("tokens_per_frame", tokens_per_frame)], [section("输出", [detail("multi-scale features", shape(batch, frames, tokens_per_frame, fpn_hidden))])], "vision"),
        build_node("prompt_encoder", "encoding", 2, "提示编码器", f"text hidden {text_hidden or '?'}", "编码文本或几何提示。", shape(batch, prompt_tokens), shape(batch, prompt_tokens, fpn_hidden), [f"text layers {text_layers or '?'}"], [detail("text_hidden", text_hidden), detail("text_layers", text_layers), detail("prompt_output", shape(batch, prompt_tokens, fpn_hidden))], [section("提示特征", [detail("encoded prompts", shape(batch, prompt_tokens, fpn_hidden))])], "text"),
        build_node("detector_tracker", "tracking", 0, "检测与时序记忆", "DETR + memory attention", "融合帧特征、提示和跨帧记忆，维护对象轨迹。", shape(batch, frames, tokens_per_frame, fpn_hidden), shape(batch, frames, object_count, fpn_hidden), [f"objects {object_count}", "temporal memory"], [detail("tracked_objects", object_count), detail("memory_hidden", fpn_hidden)], [section("跟踪状态", [detail("object states", shape(batch, frames, object_count, fpn_hidden))])], "core"),
        build_node("mask_decoder", "output", 0, "Mask 解码器", f"low-res {mask_height}x{mask_width}", "生成低分辨率 mask 并上采样到输入尺寸。", shape(batch, frames, object_count, fpn_hidden), shape(batch, frames, object_count, image_height, image_width), [f"masks {object_count}", f"low-res {mask_height}x{mask_width}"], [detail("low_res_masks", shape(batch, frames, object_count, mask_height, mask_width)), detail("output_masks", shape(batch, frames, object_count, image_height, image_width))], [section("输出", [detail("segmentation masks", shape(batch, frames, object_count, image_height, image_width))])], "output"),
    ]
    edges = [
        build_edge("video_input", "frame_processor", "video frames"),
        build_edge("frame_processor", "vision_encoder", "patches"),
        build_edge("vision_encoder", "feature_pyramid", "backbone features"),
        build_edge("prompt_input", "prompt_encoder", "prompts"),
        build_edge("feature_pyramid", "detector_tracker", "frame features"),
        build_edge("prompt_encoder", "detector_tracker", "prompt features"),
        build_edge("detector_tracker", "mask_decoder", "object states"),
    ]
    sources: list[str] = []
    for source_name in ("config.json", "configuration.json", "processor_config.json"):
        append_source_file(model_dir, sources, source_name)
    return base_model_payload(
        model_id,
        "multimodal",
        architecture,
        f"视频检测与分割模型，每帧 {tokens_per_frame} 个视觉 patch token。",
        [detail("类型", "视频分割"), detail("视觉隐藏维度", vision_hidden), detail("视觉层数", vision_layers), detail("patch size", patch_size), detail("每帧视觉 tokens", tokens_per_frame), detail("总视觉 tokens", total_vision_tokens), detail("输出 masks", object_count)],
        [
            {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 8, "step": 1, "help": "并行视频数"},
            {"name": "frames", "label": "视频帧数", "type": "number", "value": frames, "min": 1, "max": 1024, "step": 1, "help": "当前处理帧数"},
            {"name": "image_height", "label": "帧高", "type": "number", "value": image_height, "min": patch_size, "max": 4096, "step": patch_size, "help": "预处理后的帧高"},
            {"name": "image_width", "label": "帧宽", "type": "number", "value": image_width, "min": patch_size, "max": 4096, "step": patch_size, "help": "预处理后的帧宽"},
            {"name": "objects", "label": "对象数", "type": "number", "value": object_count, "min": 1, "max": 256, "step": 1, "help": "同时跟踪的对象数量"},
            {"name": "prompt_tokens", "label": "提示 token", "type": "number", "value": prompt_tokens, "min": 1, "max": 512, "step": 1, "help": "文本或几何提示长度"},
        ],
        {"batch": batch, "frames": frames, "image_height": image_height, "image_width": image_width, "objects": object_count, "prompt_tokens": prompt_tokens, "patch_size": patch_size, "tokens_per_frame": tokens_per_frame, "total_vision_tokens": total_vision_tokens},
        build_graph(lanes, nodes, edges),
        ["视觉 token 数按 patch 网格计算；FPN 的多尺度特征图数量与内部重采样细节未重复累加为独立 token。"],
        sources,
        "detector_tracker",
    )


def discover_nested_text_backbone_config(model_dir: Path) -> tuple[dict[str, Any], str | None]:
    for path in sorted(model_dir.glob("*/config.json")):
        config = read_json_file(path)
        if config.get("hidden_size") and config.get("num_hidden_layers"):
            return config, str(path.relative_to(model_dir)).replace("\\", "/")
    return {}, None


def build_tts_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    yaml_config = supplemental_yaml_config(model_dir)
    architecture = infer_architecture_name(model_dir, "multimodal")
    if architecture == "UnknownModel":
        architecture = human_model_name(model_id).split("/")[-1]
    nested_text_config, nested_text_source = discover_nested_text_backbone_config(model_dir)
    yaml_gpt = yaml_config.get("gpt") if isinstance(yaml_config.get("gpt"), dict) else {}
    yaml_llm = yaml_config.get("llm") if isinstance(yaml_config.get("llm"), dict) else {}
    yaml_flow = yaml_config.get("flow") if isinstance(yaml_config.get("flow"), dict) else {}
    yaml_hift = yaml_config.get("hift") if isinstance(yaml_config.get("hift"), dict) else {}
    yaml_s2mel = yaml_config.get("s2mel") if isinstance(yaml_config.get("s2mel"), dict) else {}
    language_config = first_defined(
        config.get("language_config") if isinstance(config.get("language_config"), dict) else None,
        config.get("lm_config") if isinstance(config.get("lm_config"), dict) else None,
        yaml_gpt,
        nested_text_config,
        {},
    ) or {}
    hidden_size = scalar_int(first_defined(language_config.get("hidden_size"), language_config.get("model_dim"), yaml_config.get("llm_input_size")), 2048)
    num_layers = scalar_int(first_defined(language_config.get("num_hidden_layers"), language_config.get("layers")), 1)
    num_heads = scalar_int(first_defined(language_config.get("num_attention_heads"), language_config.get("heads")), 1)
    vocab_size = scalar_int(first_defined(language_config.get("vocab_size"), language_config.get("number_text_tokens")), 0)
    n_vq = scalar_int(config.get("n_vq"), 1)
    audio_vocab_size = scalar_int(first_defined(config.get("audio_vocab_size"), language_config.get("number_mel_codes"), yaml_llm.get("speech_token_size"), yaml_flow.get("vocab_size")), 0)
    dataset_config = yaml_config.get("dataset") if isinstance(yaml_config.get("dataset"), dict) else {}
    audio_vae_config = config.get("audio_vae_config") if isinstance(config.get("audio_vae_config"), dict) else {}
    s2mel_preprocess = yaml_s2mel.get("preprocess_params") if isinstance(yaml_s2mel.get("preprocess_params"), dict) else {}
    sample_rate = scalar_int(first_defined(config.get("sampling_rate"), audio_vae_config.get("out_sample_rate"), s2mel_preprocess.get("sr"), yaml_config.get("sample_rate"), dataset_config.get("sample_rate")), 24000)
    max_text_tokens = scalar_int(first_defined(language_config.get("max_text_tokens"), config.get("max_length"), language_config.get("max_position_embeddings")), 8192)
    max_audio_frames = scalar_int(first_defined(language_config.get("max_mel_tokens"), config.get("max_length")), 16384)
    batch = clamp_int(query.get("batch", [1])[0], 1, maximum=16)
    text_tokens = clamp_int(query.get("text_tokens", [min(max_text_tokens, 256)])[0], min(max_text_tokens, 256), maximum=max_text_tokens)
    audio_frames = clamp_int(query.get("audio_frames", [min(max_audio_frames, 750)])[0], min(max_audio_frames, 750), maximum=max_audio_frames)
    uses_delay_pattern = n_vq > 1 or "delay" in str(config.get("model_type") or "").lower()
    audio_steps = audio_frames + max(n_vq - 1, 0) if uses_delay_pattern else audio_frames
    total_steps = text_tokens + audio_steps
    prediction_heads = n_vq + 1 if uses_delay_pattern else 1
    dit_config = config.get("dit_config") if isinstance(config.get("dit_config"), dict) else {}
    encoder_config = config.get("encoder_config") if isinstance(config.get("encoder_config"), dict) else {}
    is_continuous_tts = bool(audio_vae_config and dit_config and not audio_vocab_size)
    token_frame_rate = scalar_int(yaml_config.get("token_frame_rate"), 0)
    token_mel_ratio = scalar_int(yaml_config.get("token_mel_ratio"), 1)
    mel_spec_config = yaml_config.get("mel_spec_transform1") if isinstance(yaml_config.get("mel_spec_transform1"), dict) else {}
    mel_hop = scalar_int(mel_spec_config.get("hop_size"), 0)
    mel_frames = audio_frames * token_mel_ratio if yaml_flow else 0
    waveform_samples = mel_frames * mel_hop if mel_frames and mel_hop else 0

    if uses_delay_pattern:
        sequence_name = "Delay Pattern"
        sequence_description = "对并行 RVQ 码本按 codebook 索引错位，形成延迟调度序列。"
        sequence_input_shape = shape(batch, audio_frames, n_vq)
        sequence_output_shape = shape(batch, audio_steps, n_vq)
        sequence_sections = [section("推导公式", [detail("delay steps", f"{audio_frames} + {n_vq} - 1 = {audio_steps}")])]
        sequence_kind = "RVQ codebooks"
    elif is_continuous_tts:
        latent_dim = scalar_int(first_defined(config.get("scalar_quantization_latent_dim"), audio_vae_config.get("latent_dim"), config.get("feat_dim")), 0)
        sequence_name = "声学潜变量序列"
        sequence_description = "组织连续或标量量化的声学潜变量 patch，供语言主干与残差生成器建模。"
        sequence_input_shape = shape(batch, audio_frames, latent_dim or "latent")
        sequence_output_shape = sequence_input_shape
        sequence_sections = [section("声学表示", [detail("latent_dim", latent_dim), detail("patch_size", config.get("patch_size"))])]
        sequence_kind = "continuous latents"
    else:
        sequence_name = "声学 Token 序列"
        sequence_description = "组织单码流语音 token 或 mel code；没有多码本错位，因此序列长度保持不变。"
        sequence_input_shape = shape(batch, audio_frames)
        sequence_output_shape = sequence_input_shape
        sequence_sections = [section("序列长度", [detail("acoustic steps", f"{audio_frames} = {audio_steps}")])]
        sequence_kind = "single acoustic stream"

    lanes = [("inputs", "输入"), ("processing", "处理"), ("backbone", "主干"), ("synthesis", "声学解码"), ("output", "输出")]
    nodes = [
        build_node("tts_text_input", "inputs", 0, "文本输入", "token ids", "待合成文本与控制提示。", shape(batch, text_tokens), shape(batch, text_tokens, hidden_size), [f"text {text_tokens}"], [detail("text_tokens", text_tokens), detail("hidden_size", hidden_size)], [section("输入", [detail("text hidden", shape(batch, text_tokens, hidden_size))])], "text"),
        build_node("delay_pattern" if uses_delay_pattern else "acoustic_sequence", "processing", 0, sequence_name, sequence_kind, sequence_description, sequence_input_shape, sequence_output_shape, [f"units {audio_frames}", f"steps {audio_steps}"], [detail("audio_units", audio_frames), detail("n_vq", n_vq if uses_delay_pattern else None), detail("sequence_steps", audio_steps)], sequence_sections, "audio"),
        build_node("tts_backbone", "backbone", 0, "Transformer 主干", f"{num_layers} layers / {num_heads} heads", "语言主干联合建模文本条件与自回归声学序列。", shape(batch, total_steps, hidden_size), shape(batch, total_steps, hidden_size), [f"hidden {hidden_size}", f"steps {total_steps}"], [detail("text_tokens", text_tokens), detail("acoustic_steps", audio_steps), detail("total_steps", total_steps)], [section("上下文", [detail("text + acoustic", f"{text_tokens} + {audio_steps} = {total_steps}")])], "core"),
    ]
    sequence_node_id = "delay_pattern" if uses_delay_pattern else "acoustic_sequence"
    prediction_node_id = "parallel_heads" if uses_delay_pattern else "acoustic_head"
    if uses_delay_pattern:
        prediction_shape = shape(batch, audio_steps, n_vq, audio_vocab_size or "audio_vocab")
        prediction_name = "并行预测头"
        prediction_subtitle = f"{prediction_heads} heads"
        prediction_description = "一个文本头加多个 RVQ 音频码本头。"
    elif is_continuous_tts:
        prediction_width = scalar_int(first_defined(config.get("scalar_quantization_latent_dim"), audio_vae_config.get("latent_dim"), config.get("feat_dim")), 0)
        prediction_shape = shape(batch, audio_steps, prediction_width or "latent")
        prediction_name = "声学潜变量预测"
        prediction_subtitle = f"latent {prediction_width or '?'}"
        prediction_description = "主语言模型预测粗粒度声学潜变量，再由残差语言模型与 DiT 细化。"
    else:
        prediction_shape = shape(batch, audio_steps, audio_vocab_size or "audio_vocab")
        prediction_name = "声学 Token 预测头"
        prediction_subtitle = f"vocab {audio_vocab_size or '?'}"
        prediction_description = "单一声学码流预测头；不额外虚构文本生成头或 RVQ 并行头。"
    nodes.append(build_node(prediction_node_id, "backbone", 1, prediction_name, prediction_subtitle, prediction_description, shape(batch, audio_steps, hidden_size), prediction_shape, [f"heads {prediction_heads}", f"steps {audio_steps}"], [detail("prediction_heads", prediction_heads), detail("prediction", prediction_shape), detail("text_vocab", vocab_size)], [section("输出", [detail("acoustic prediction", prediction_shape)])], "head"))

    edges = [
        build_edge("tts_text_input", "tts_backbone", "text hidden"),
        build_edge(sequence_node_id, "tts_backbone", "acoustic sequence"),
        build_edge("tts_backbone", prediction_node_id, "hidden states"),
    ]
    synthesis_label = "音频解码"
    warning = "配置未提供声学序列到波形时长的确定换算关系，因此不把序列长度强行换算成音频秒数。"
    if uses_delay_pattern:
        nodes.append(build_node("codec_decoder", "synthesis", 0, "音频 Codec 解码", f"waveform @ {sample_rate} Hz", "将并行 RVQ code 还原为波形；配置未提供 codec hop。", shape(batch, audio_frames, n_vq), shape(batch, "waveform_samples"), [f"{sample_rate} Hz"], [detail("codec_frames", audio_frames), detail("sample_rate", sample_rate)], [section("输出", [detail("waveform", shape(batch, "waveform_samples"))])], "output"))
        edges.append(build_edge(prediction_node_id, "codec_decoder", "RVQ codes"))
        synthesis_label = "RVQ Codec"
    elif yaml_flow:
        flow_encoder = yaml_flow.get("encoder") if isinstance(yaml_flow.get("encoder"), dict) else {}
        flow_decoder = yaml_flow.get("decoder") if isinstance(yaml_flow.get("decoder"), dict) else {}
        flow_estimator = flow_decoder.get("estimator") if isinstance(flow_decoder.get("estimator"), dict) else {}
        mel_channels = scalar_int(first_defined(yaml_flow.get("output_size"), flow_estimator.get("out_channels")), 80)
        nodes.extend([
            build_node("flow_decoder", "synthesis", 0, "因果 Flow Matching", f"{scalar_int(flow_encoder.get('num_blocks'), 0)} conformer + {scalar_int(flow_estimator.get('num_mid_blocks'), 0)} mid blocks", "将离散语音 token 上采样并生成 mel 频谱。", shape(batch, audio_steps), shape(batch, mel_frames or "mel_frames", mel_channels), [f"token:mel 1:{token_mel_ratio}", f"mel {mel_channels}"], [detail("token_frames", audio_frames), detail("token_mel_ratio", token_mel_ratio), detail("mel_frames", mel_frames), detail("mel_channels", mel_channels)], [section("帧数推导", [detail("mel frames", f"{audio_frames} * {token_mel_ratio} = {mel_frames}")])], "audio"),
            build_node("vocoder", "output", 0, "HiFT 声码器", f"waveform @ {sample_rate} Hz", "按配置中的 mel hop 将频谱还原为波形。", shape(batch, mel_frames, mel_channels), shape(batch, waveform_samples or "waveform_samples"), [f"hop {mel_hop or '?'}", f"{sample_rate} Hz"], [detail("mel_frames", mel_frames), detail("hop_size", mel_hop), detail("waveform_samples", waveform_samples), detail("duration_seconds", round(waveform_samples / sample_rate, 3) if waveform_samples else None)], [section("样本数推导", [detail("waveform samples", f"{mel_frames} * {mel_hop} = {waveform_samples}")])], "output"),
        ])
        edges.extend([build_edge(prediction_node_id, "flow_decoder", "speech tokens"), build_edge("flow_decoder", "vocoder", "mel spectrogram")])
        synthesis_label = "Causal CFM + HiFT"
        warning = "语音 token 帧率、token/mel 比和 mel hop 均来自配置，当前波形时长可精确由这些值推导。"
    elif yaml_s2mel:
        s2mel_dit = yaml_s2mel.get("DiT") if isinstance(yaml_s2mel.get("DiT"), dict) else {}
        mel_channels = scalar_int(s2mel_dit.get("in_channels"), 80)
        nodes.extend([
            build_node("s2mel_dit", "synthesis", 0, "S2Mel DiT", f"{scalar_int(s2mel_dit.get('depth'), 0)} layers / hidden {scalar_int(s2mel_dit.get('hidden_dim'), 0)}", "将语义码流、风格与条件特征映射为 mel 频谱。", shape(batch, audio_steps), shape(batch, "mel_frames", mel_channels), [f"mel {mel_channels}", "flow matching"], [detail("DiT_depth", s2mel_dit.get("depth")), detail("hidden_dim", s2mel_dit.get("hidden_dim")), detail("mel_channels", mel_channels)], [section("输出", [detail("mel", shape(batch, "mel_frames", mel_channels))])], "audio"),
            build_node("vocoder", "output", 0, "BigVGAN 声码器", f"waveform @ {sample_rate} Hz", "将 S2Mel 输出还原为波形；配置未声明语义码到 mel 帧的固定比例。", shape(batch, "mel_frames", mel_channels), shape(batch, "waveform_samples"), [f"{sample_rate} Hz"], [detail("vocoder", (yaml_config.get("vocoder") or {}).get("name") if isinstance(yaml_config.get("vocoder"), dict) else None), detail("sample_rate", sample_rate)], [section("输出", [detail("waveform", shape(batch, "waveform_samples"))])], "output"),
        ])
        edges.extend([build_edge(prediction_node_id, "s2mel_dit", "semantic codes"), build_edge("s2mel_dit", "vocoder", "mel spectrogram")])
        synthesis_label = "S2Mel DiT + BigVGAN"
    elif audio_vae_config and dit_config:
        residual_layers = scalar_int(config.get("residual_lm_num_layers"), 0)
        latent_dim = scalar_int(first_defined(config.get("scalar_quantization_latent_dim"), audio_vae_config.get("latent_dim"), config.get("feat_dim")), 0)
        nodes.extend([
            build_node("residual_lm", "synthesis", 0, "残差语言模型", f"{residual_layers} layers", "在主语言模型输出上继续细化声学潜变量。", prediction_shape, shape(batch, audio_steps, latent_dim or "latent"), [f"layers {residual_layers}", f"latent {latent_dim or '?'}"], [detail("layers", residual_layers), detail("latent_dim", latent_dim)], [section("输出", [detail("refined latents", shape(batch, audio_steps, latent_dim or "latent"))])], "core"),
            build_node("audio_dit", "synthesis", 1, "声学 DiT", f"{scalar_int(dit_config.get('num_layers'), 0)} layers / hidden {scalar_int(dit_config.get('hidden_dim'), 0)}", "用条件流匹配将潜变量映射到音频 VAE latent。", shape(batch, audio_steps, latent_dim or "latent"), shape(batch, audio_steps, scalar_int(audio_vae_config.get("latent_dim"), 0) or "vae_latent"), [f"heads {scalar_int(dit_config.get('num_heads'), 0)}"], [detail("layers", dit_config.get("num_layers")), detail("hidden_dim", dit_config.get("hidden_dim")), detail("encoder_layers", encoder_config.get("num_layers"))], [section("输出", [detail("VAE latents", shape(batch, audio_steps, scalar_int(audio_vae_config.get("latent_dim"), 0) or "vae_latent"))])], "audio"),
            build_node("audio_vae_decoder", "output", 0, "Audio VAE 解码", f"waveform @ {sample_rate} Hz", "将声学 DiT 输出解码为波形；配置未给出一个声学 patch 对应的固定样本数。", shape(batch, audio_steps, scalar_int(audio_vae_config.get("latent_dim"), 0) or "vae_latent"), shape(batch, "waveform_samples"), [f"{sample_rate} Hz"], [detail("decoder_dim", audio_vae_config.get("decoder_dim")), detail("decoder_rates", audio_vae_config.get("decoder_rates")), detail("sample_rate", sample_rate)], [section("输出", [detail("waveform", shape(batch, "waveform_samples"))])], "output"),
        ])
        edges.extend([build_edge(prediction_node_id, "residual_lm", "coarse latents"), build_edge("residual_lm", "audio_dit", "refined latents"), build_edge("audio_dit", "audio_vae_decoder", "VAE latents")])
        synthesis_label = "Residual LM + DiT + Audio VAE"
    else:
        nodes.append(build_node("acoustic_decoder", "output", 0, "声学解码器", f"waveform @ {sample_rate} Hz", "将声学表示还原为波形；配置未公开内部时长换算。", prediction_shape, shape(batch, "waveform_samples"), [f"{sample_rate} Hz"], [detail("sample_rate", sample_rate)], [section("输出", [detail("waveform", shape(batch, "waveform_samples"))])], "output"))
        edges.append(build_edge(prediction_node_id, "acoustic_decoder", "acoustic representation"))
    sources: list[str] = []
    for source_name in ("config.json", "configuration.json", "config.yaml", "cosyvoice2.yaml", "processor_config.json"):
        append_source_file(model_dir, sources, source_name)
    if nested_text_source and nested_text_source not in sources:
        sources.append(nested_text_source)
    headline = (
        f"延迟模式 TTS，{audio_frames} 个 codec 帧展开为 {audio_steps} 个解码步。"
        if uses_delay_pattern
        else f"语音合成模型，当前按 {text_tokens} 个文本 token 与 {audio_frames} 个声学单元展示；声学后端为 {synthesis_label}。"
    )
    summary = [detail("类型", "语音合成"), detail("语言隐藏维度", hidden_size), detail("主干层数", num_layers)]
    if uses_delay_pattern:
        summary.extend([detail("RVQ 码本数", n_vq), detail("并行预测头", prediction_heads)])
    else:
        summary.extend([detail("声学码流", 1), detail("声学预测头", prediction_heads)])
    summary.extend([detail("当前解码步", audio_steps), detail("声学后端", synthesis_label), detail("采样率", f"{sample_rate} Hz")])
    return base_model_payload(
        model_id,
        "multimodal",
        architecture,
        headline,
        summary,
        [
            {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 16, "step": 1, "help": "并行合成条数"},
            {"name": "text_tokens", "label": "文本 token", "type": "number", "value": text_tokens, "min": 1, "max": max_text_tokens, "step": 1, "help": "输入文本与提示长度"},
            {"name": "audio_frames", "label": "Codec 帧" if uses_delay_pattern else "声学单元", "type": "number", "value": audio_frames, "min": 1, "max": max_audio_frames, "step": 1, "help": "目标 RVQ codec 帧数" if uses_delay_pattern else "目标语音 token、mel code 或声学 latent 单元数"},
        ],
        {"batch": batch, "text_tokens": text_tokens, "audio_frames": audio_frames, "n_vq": n_vq, "audio_steps": audio_steps, "delayed_steps": audio_steps if uses_delay_pattern else None, "total_steps": total_steps, "sample_rate": sample_rate, "waveform_samples": waveform_samples or None},
        build_graph(lanes, nodes, edges),
        [warning],
        sources,
        "tts_backbone",
    )


def build_hy_world_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    pano_config = read_json_file(model_dir / "HY-Pano-2.0" / "config.json")
    generation_config = read_json_file(model_dir / "HY-Pano-2.0" / "generation_config.json")
    mirror_config = read_json_file(model_dir / "HY-WorldMirror-2.0" / "config.json")
    pano_vae = pano_config.get("vae") if isinstance(pano_config.get("vae"), dict) else {}
    pano_vit = pano_config.get("vit") if isinstance(pano_config.get("vit"), dict) else {}
    pano_dims = parse_llm_dims(pano_config)
    pano_metrics = estimate_llm_metrics(pano_dims)

    task = str(query.get("task", ["panorama"])[0]).lower()
    if task not in {"panorama", "reconstruction"}:
        task = "panorama"
    batch = 1
    prompt_tokens = clamp_int(query.get("seq_len", [1024])[0], 1024, maximum=int(pano_config.get("max_position_embeddings", 22800) or 22800))
    pano_height = clamp_int(query.get("image_height", [960])[0], 960, minimum=16, maximum=4096)
    pano_width = clamp_int(query.get("image_width", [1952])[0], 1952, minimum=16, maximum=8192)
    denoise_steps = clamp_int(query.get("steps", [generation_config.get("diff_infer_steps", 50)])[0], 50, maximum=200)
    blend_width = clamp_int(query.get("blend_width", [32])[0], 32, maximum=max(pano_width - 1, 1))
    output_width = max(pano_width - blend_width, 1)
    vae_scale_height, vae_scale_width = spatial_pair(first_defined(pano_config.get("vae_downsample_factor"), pano_vae.get("ffactor_spatial")), 16)
    pano_latent_height = ceil_div(pano_height, vae_scale_height)
    pano_latent_width = ceil_div(pano_width, vae_scale_width)
    pano_latent_channels = scalar_int(pano_vae.get("latent_channels"), 32)
    pano_latent_tokens = pano_latent_height * pano_latent_width
    pano_condition_tokens = scalar_int(pano_config.get("vit_token"), 64)

    views = clamp_int(query.get("views", [8])[0], 8, maximum=32)
    mirror_patch = scalar_int(mirror_config.get("patch_size"), 14)
    mirror_height = clamp_int(query.get("recon_height", [mirror_config.get("img_size", 518)])[0], scalar_int(mirror_config.get("img_size"), 518), minimum=mirror_patch, maximum=2048)
    mirror_width = clamp_int(query.get("recon_width", [mirror_config.get("img_size", 518)])[0], scalar_int(mirror_config.get("img_size"), 518), minimum=mirror_patch, maximum=2048)
    mirror_rows = ceil_div(mirror_height, mirror_patch)
    mirror_cols = ceil_div(mirror_width, mirror_patch)
    mirror_registers = scalar_int(mirror_config.get("num_register_tokens"), 0)
    mirror_tokens_per_view = mirror_rows * mirror_cols + mirror_registers
    mirror_total_tokens = views * mirror_tokens_per_view
    gaussian_count = views * mirror_height * mirror_width
    mirror_hidden = scalar_int(mirror_config.get("embed_dim"), 1024)
    mirror_layers = scalar_int(mirror_config.get("depth"), 24)
    mirror_heads = scalar_int(mirror_config.get("num_heads"), 16)

    lanes = [
        ("inputs", "输入"),
        ("conditioning", "条件编码"),
        ("panorama", "全景生成"),
        ("reconstruction", "三维重建"),
        ("output", "输出"),
    ]
    nodes = [
        build_node("world_prompt", "inputs", 0, "文本提示", "reasoning / recaption", "HY-Pano 可先进行思考与重写，再生成 360° 全景。", shape(batch, prompt_tokens), shape(batch, prompt_tokens, pano_dims["hidden_size"]), [f"tokens {prompt_tokens}"], [detail("prompt_tokens", prompt_tokens), detail("hidden_size", pano_dims["hidden_size"])], [section("输入", [detail("text hidden", shape(batch, prompt_tokens, pano_dims["hidden_size"]))])], "text"),
        build_node("reference_image", "inputs", 1, "参考图像", "single perspective image", "用于 image-to-panorama 的单视角参考图。", shape(batch, 3, "input_h", "input_w"), shape(batch, 3, "input_h", "input_w"), ["single view"], [detail("mode", "image-to-panorama")], [section("输入", [detail("RGB", shape(batch, 3, "input_h", "input_w"))])], "vision"),
        build_node("multiview_input", "inputs", 2, "多视图 / 视频帧", f"{views} views", "WorldMirror 使用多视图图像或抽取后的视频帧进行前馈重建。", shape(batch, views, 3, mirror_height, mirror_width), shape(batch, views, 3, mirror_height, mirror_width), [f"views {views}", f"{mirror_height}x{mirror_width}"], [detail("views", views), detail("height", mirror_height), detail("width", mirror_width)], [section("输入", [detail("multi-view RGB", shape(batch, views, 3, mirror_height, mirror_width))])], "vision"),
        build_node("pano_vision_encoder", "conditioning", 0, "SigLIP2 视觉条件", f"{pano_vit.get('num_hidden_layers', '?')} layers / hidden {pano_vit.get('hidden_size', '?')}", "视觉编码后通过两层对齐器压缩为固定数量的条件 token。", shape(batch, 3, "input_h", "input_w"), shape(batch, pano_condition_tokens, pano_dims["hidden_size"]), [f"tokens {pano_condition_tokens}", f"patch {pano_vit.get('patch_size', '?')}"], [detail("vision_hidden", pano_vit.get("hidden_size")), detail("vision_layers", pano_vit.get("num_hidden_layers")), detail("aligned_tokens", pano_condition_tokens), detail("aligned_hidden", pano_dims["hidden_size"])], [section("输出", [detail("image condition", shape(batch, pano_condition_tokens, pano_dims["hidden_size"]))])], "vision"),
        build_node("pano_moe_backbone", "conditioning", 1, "HY-Pano MoE 主干", f"{pano_dims['num_layers']} layers / {pano_dims['num_experts']} experts", "统一处理文本推理、重写提示与图像生成条件。", shape(batch, prompt_tokens + pano_condition_tokens, pano_dims["hidden_size"]), shape(batch, prompt_tokens + pano_condition_tokens, pano_dims["hidden_size"]), [f"top-{pano_dims['experts_per_tok']}", f"active {pano_metrics['active_params'] / 1e9:.2f}B" if pano_metrics else "active ?"], [detail("hidden_size", pano_dims["hidden_size"]), detail("layers", pano_dims["num_layers"]), detail("experts", pano_dims["num_experts"]), detail("experts_per_token", pano_dims["experts_per_tok"]), detail("total_params", pano_metrics["total_params"] if pano_metrics else None), detail("active_params", pano_metrics["active_params"] if pano_metrics else None)], [section("参数估算", [detail("total", f"{pano_metrics['total_params'] / 1e9:.2f}B" if pano_metrics else "?"), detail("active/token", f"{pano_metrics['active_params'] / 1e9:.2f}B" if pano_metrics else "?")])], "core"),
        build_node("pano_latent", "panorama", 0, "全景 Latent", f"VAE scale {vae_scale_height}x{vae_scale_width}", "输出画布经 VAE 空间压缩形成去噪 latent。", shape(batch, 3, pano_height, pano_width), shape(batch, pano_latent_channels, pano_latent_height, pano_latent_width), [f"tokens {pano_latent_tokens}", f"channels {pano_latent_channels}"], [detail("latent_height", pano_latent_height), detail("latent_width", pano_latent_width), detail("latent_tokens", pano_latent_tokens)], [section("推导公式", [detail("latent grid", f"ceil({pano_height}/{vae_scale_height}) x ceil({pano_width}/{vae_scale_width}) = {pano_latent_height} x {pano_latent_width}"), detail("tokens", f"{pano_latent_height} * {pano_latent_width} = {pano_latent_tokens}")])], "latent"),
        build_node("pano_denoiser", "panorama", 1, "条件扩散去噪", f"{denoise_steps} steps", "HY-Pano 主干以文本与参考图条件迭代更新全景 latent。", shape(batch, pano_latent_tokens, pano_latent_channels), shape(batch, pano_latent_tokens, pano_latent_channels), [f"steps {denoise_steps}", "joint full attention"], [detail("steps", denoise_steps), detail("latent_tokens", pano_latent_tokens), detail("condition_tokens", prompt_tokens + pano_condition_tokens)], [section("循环", [detail("denoise iterations", denoise_steps)])], "core"),
        build_node("pano_vae_decode", "panorama", 2, "3D VAE 解码", f"RGB {pano_height}x{pano_width}", "将去噪 latent 解码为带左右重叠区的 ERP 全景。", shape(batch, pano_latent_channels, pano_latent_height, pano_latent_width), shape(batch, 3, pano_height, pano_width), [f"scale {vae_scale_height}x{vae_scale_width}"], [detail("decoded", shape(batch, 3, pano_height, pano_width)), detail("blend_overlap", blend_width)], [section("输出", [detail("pre-blend panorama", shape(batch, 3, pano_height, pano_width))])], "output"),
        build_node("pano_blend", "output", 0, "环形边缘融合", f"remove {blend_width}px overlap", "将左右重叠区环形融合，得到无缝 360° ERP 图像。", shape(batch, 3, pano_height, pano_width), shape(batch, 3, pano_height, output_width), [f"final {pano_height}x{output_width}"], [detail("input_width", pano_width), detail("blend_width", blend_width), detail("output_width", output_width)], [section("宽度推导", [detail("final width", f"{pano_width} - {blend_width} = {output_width}")])], "output"),
        build_node("mirror_patch_embed", "reconstruction", 0, "多视图 Patch Embedding", f"patch {mirror_patch} + {mirror_registers} registers", "逐视图切分 patch，并加入 register token。", shape(batch, views, 3, mirror_height, mirror_width), shape(batch, views, mirror_tokens_per_view, mirror_hidden), [f"tokens/view {mirror_tokens_per_view}", f"total {mirror_total_tokens}"], [detail("patch_rows", mirror_rows), detail("patch_cols", mirror_cols), detail("register_tokens", mirror_registers), detail("tokens_per_view", mirror_tokens_per_view), detail("total_tokens", mirror_total_tokens)], [section("推导公式", [detail("tokens/view", f"ceil({mirror_height}/{mirror_patch}) * ceil({mirror_width}/{mirror_patch}) + {mirror_registers} = {mirror_tokens_per_view}"), detail("all views", f"{views} * {mirror_tokens_per_view} = {mirror_total_tokens}")])], "vision"),
        build_node("world_mirror", "reconstruction", 1, "WorldMirror Transformer", f"{mirror_layers} layers / {mirror_heads} heads", "通过几何上下文 Transformer 融合多视图与可选相机、深度先验。", shape(batch, views, mirror_tokens_per_view, mirror_hidden), shape(batch, views, mirror_tokens_per_view, mirror_hidden), [f"hidden {mirror_hidden}", "normalized RoPE"], [detail("hidden_size", mirror_hidden), detail("layers", mirror_layers), detail("heads", mirror_heads), detail("condition_strategy", mirror_config.get("condition_strategy"))], [section("主干", [detail("features", shape(batch, views, mirror_tokens_per_view, mirror_hidden))])], "core"),
        build_node("geometry_heads", "reconstruction", 2, "DPT 几何预测头", "depth / normals / points / cameras", "一次前向同时预测逐像素几何与逐视图相机参数。", shape(batch, views, mirror_tokens_per_view, mirror_hidden), shape(batch, views, mirror_height, mirror_width, 3), ["dense geometry", f"pixels {gaussian_count}"], [detail("depth", shape(batch, views, mirror_height, mirror_width, 1)), detail("normals", shape(batch, views, mirror_height, mirror_width, 3)), detail("pts3d", shape(batch, views, mirror_height, mirror_width, 3)), detail("camera_poses", shape(batch, views, 4, 4)), detail("camera_intrs", shape(batch, views, 3, 3))], [section("输出", [detail("point maps", shape(batch, views, mirror_height, mirror_width, 3)), detail("camera params", shape(batch, views, 9))])], "head"),
        build_node("gaussian_splats", "output", 1, "3D Gaussian Splats", f"N = {gaussian_count}", "每个输入像素产生一个候选 Gaussian，后续可按置信度过滤或压缩。", shape(batch, views, mirror_height, mirror_width, 3), shape(batch, gaussian_count, 3), [f"gaussians {gaussian_count}", "PLY / 3DGS"], [detail("N", gaussian_count), detail("means", shape(batch, gaussian_count, 3)), detail("scales", shape(batch, gaussian_count, 3)), detail("quaternions", shape(batch, gaussian_count, 4)), detail("opacities", shape(batch, gaussian_count)), detail("SH", shape(batch, gaussian_count, 1, 3))], [section("数量推导", [detail("N", f"{views} * {mirror_height} * {mirror_width} = {gaussian_count}")])], "output"),
    ]
    edges = [
        build_edge("world_prompt", "pano_moe_backbone", "prompt hidden"),
        build_edge("reference_image", "pano_vision_encoder", "reference RGB"),
        build_edge("pano_vision_encoder", "pano_moe_backbone", "64 image tokens"),
        build_edge("pano_moe_backbone", "pano_denoiser", "reasoned condition"),
        build_edge("pano_latent", "pano_denoiser", "initial latent"),
        build_edge("pano_denoiser", "pano_vae_decode", "denoised latent"),
        build_edge("pano_vae_decode", "pano_blend", "overlapped ERP"),
        build_edge("multiview_input", "mirror_patch_embed", "multi-view RGB"),
        build_edge("mirror_patch_embed", "world_mirror", "view tokens"),
        build_edge("world_mirror", "geometry_heads", "geometric features"),
        build_edge("geometry_heads", "gaussian_splats", "points + GS attributes"),
    ]
    summary = [
        detail("类型", "三维世界生成与重建"),
        detail("当前任务", "360° 全景生成" if task == "panorama" else "多视图三维重建"),
        detail("HY-Pano 主干参数", f"{pano_metrics['total_params'] / 1e9:.2f}B / active {pano_metrics['active_params'] / 1e9:.2f}B" if pano_metrics else "?"),
        detail("全景 latent tokens", pano_latent_tokens),
        detail("最终全景", f"{pano_height}x{output_width}"),
        detail("WorldMirror tokens", mirror_total_tokens),
        detail("候选 Gaussians", gaussian_count),
    ]
    controls = [
        {"name": "task", "label": "任务", "type": "select", "value": task, "options": ["panorama", "reconstruction"], "help": "选择当前重点查看的 HY-World 分支"},
        {"name": "seq_len", "label": "提示 token", "type": "number", "value": prompt_tokens, "min": 1, "max": int(pano_config.get("max_position_embeddings", 22800) or 22800), "step": 1, "help": "HY-Pano 推理与重写上下文长度"},
        {"name": "image_height", "label": "全景生成高度", "type": "number", "value": pano_height, "min": 16, "max": 4096, "step": 16, "help": "融合前 ERP 高度"},
        {"name": "image_width", "label": "全景生成宽度", "type": "number", "value": pano_width, "min": 16, "max": 8192, "step": 16, "help": "包含环形融合重叠区的宽度"},
        {"name": "steps", "label": "去噪步数", "type": "number", "value": denoise_steps, "min": 1, "max": 200, "step": 1, "help": "HY-Pano diffusion 迭代次数"},
        {"name": "blend_width", "label": "环形融合宽度", "type": "number", "value": blend_width, "min": 1, "max": max(pano_width - 1, 1), "step": 1, "help": "最终输出会移除的左右重叠宽度"},
        {"name": "views", "label": "重建视图数", "type": "number", "value": views, "min": 1, "max": 32, "step": 1, "help": "输入 WorldMirror 的图像或视频帧数量"},
        {"name": "recon_height", "label": "重建输入高", "type": "number", "value": mirror_height, "min": mirror_patch, "max": 2048, "step": mirror_patch, "help": "WorldMirror 输入高度"},
        {"name": "recon_width", "label": "重建输入宽", "type": "number", "value": mirror_width, "min": mirror_patch, "max": 2048, "step": mirror_patch, "help": "WorldMirror 输入宽度"},
    ]
    sources: list[str] = []
    for source_name in ("README.md", "DOCUMENTATION.md", "HY-Pano-2.0/config.json", "HY-Pano-2.0/generation_config.json", "HY-WorldMirror-2.0/config.json"):
        append_source_file(model_dir, sources, source_name)
    headline = (
        f"HY-Pano 以 {pano_latent_tokens} 个 latent 位置生成 {pano_height}x{output_width} 的无缝 ERP 全景。"
        if task == "panorama"
        else f"WorldMirror 将 {views} 个视图编码为 {mirror_total_tokens} 个 token，并产生 {gaussian_count} 个候选 Gaussian。"
    )
    return base_model_payload(
        model_id,
        "multimodal",
        "HYWorld2Pipeline",
        headline,
        summary,
        controls,
        {
            "task": task,
            "seq_len": prompt_tokens,
            "image_height": pano_height,
            "image_width": pano_width,
            "steps": denoise_steps,
            "blend_width": blend_width,
            "pano_latent_tokens": pano_latent_tokens,
            "pano_output_width": output_width,
            "views": views,
            "recon_height": mirror_height,
            "recon_width": mirror_width,
            "tokens_per_view": mirror_tokens_per_view,
            "mirror_total_tokens": mirror_total_tokens,
            "gaussian_count": gaussian_count,
        },
        build_graph(lanes, nodes, edges),
        [
            "HY-Pano 参数量只按配置中的统一 MoE 主干估算；VAE 与 SigLIP2 子模块未重复并入该数字。",
            "默认 1952 像素生成宽度包含 32 像素环形重叠，融合后的官方默认输出宽度为 1920。",
            "WorldMirror 的候选 Gaussian 数 N=S*H*W 是过滤和压缩前数量。",
        ],
        sources,
        "pano_denoiser" if task == "panorama" else "world_mirror",
    )


def infer_default_image_size(config: dict[str, Any], vision_config: dict[str, Any], image_processor: dict[str, Any]) -> tuple[int, int]:
    candidate_resolutions = config.get("candidate_resolutions")
    if isinstance(candidate_resolutions, list) and candidate_resolutions:
        pair = candidate_resolutions[0]
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return clamp_int(pair[0], 1024), clamp_int(pair[1], 1024)

    size_config = image_processor.get("size") if isinstance(image_processor, dict) else None
    if isinstance(size_config, (list, tuple)):
        return spatial_pair(size_config, 1024)
    if isinstance(size_config, dict):
        height = size_config.get("height")
        width = size_config.get("width")
        if isinstance(height, (int, float)) or isinstance(width, (int, float)):
            resolved_height = clamp_int(first_defined(height, width), 1024)
            resolved_width = clamp_int(first_defined(width, height), 1024)
            return resolved_height, resolved_width
        shortest = size_config.get("shortest_edge")
        if isinstance(shortest, (int, float)) and 1 <= shortest <= 4096:
            size = clamp_int(shortest, 1024, maximum=4096)
            return size, size

    scale_resolution = first_defined(image_processor.get("scale_resolution"), image_processor.get("target_size"))
    if isinstance(scale_resolution, (int, float)) and 1 <= scale_resolution <= 4096:
        size = clamp_int(scale_resolution, 1024, maximum=4096)
        return size, size

    image_size = vision_config.get("image_size")
    if isinstance(image_size, (list, tuple)):
        return spatial_pair(image_size, 1024)
    if isinstance(image_size, (int, float)) and 1 <= image_size <= 4096:
        size = clamp_int(image_size, 1024, maximum=4096)
        return size, size

    return 1024, 1024


def build_multimodal_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config_dir = architecture_config_dir(model_dir)
    inherited_base = model_card_base_reference(model_dir) if config_dir != model_dir else None
    config = primary_config(model_dir)
    params_config = read_json_file(config_dir / "params.json")
    processor_config = read_json_file(config_dir / "processor_config.json")
    image_processor_config = read_json_file(config_dir / "preprocessor_config.json")
    video_processor_config = read_json_file(config_dir / "video_preprocessor_config.json")
    architecture = infer_architecture_name(model_dir, "multimodal")

    text_config = first_defined(config.get("text_config"), config.get("language_config"), config, params_config) or {}
    vision_config = first_defined(config.get("vision_config"), params_config.get("vision_encoder"), {}) or {}
    projector_config = config.get("projector_config") if isinstance(config.get("projector_config"), dict) else {}
    image_processor = processor_config.get("image_processor") if isinstance(processor_config.get("image_processor"), dict) else processor_config or image_processor_config
    video_processor = processor_config.get("video_processor") if isinstance(processor_config.get("video_processor"), dict) else video_processor_config
    audio_feature_extractor = processor_config.get("feature_extractor") if isinstance(processor_config.get("feature_extractor"), dict) else {}
    image_compression = first_defined(
        image_processor.get("img_token_compression_config"),
        vision_config.get("img_token_compression_config"),
        config.get("img_token_compression_config"),
        {},
    ) or {}
    media_processor = image_processor.get("media_proc_cfg") if isinstance(image_processor.get("media_proc_cfg"), dict) else {}

    language_hidden = int(first_defined(text_config.get("hidden_size"), config.get("hidden_size"), params_config.get("dim"), projector_config.get("n_embed"), 2048) or 2048)
    num_layers = int(first_defined(text_config.get("num_hidden_layers"), config.get("num_hidden_layers"), params_config.get("n_layers"), 0) or 0)
    num_heads = int(first_defined(text_config.get("num_attention_heads"), config.get("num_attention_heads"), params_config.get("n_heads"), 0) or 0)
    head_dim = int(first_defined(text_config.get("head_dim"), params_config.get("head_dim"), language_hidden // num_heads if language_hidden and num_heads else 0) or 0)
    vocab_size = int(first_defined(text_config.get("vocab_size"), config.get("vocab_size"), params_config.get("vocab_size"), 0) or 0)
    max_position = int(first_defined(text_config.get("max_position_embeddings"), config.get("max_position_embeddings"), params_config.get("max_position_embeddings"), 8192) or 8192)

    nested_patch_size = max_nested_numeric(vision_config.get("width"), "patch_size") if isinstance(vision_config.get("width"), dict) else None
    patch_size = scalar_int(first_defined(image_processor.get("patch_size"), media_processor.get("patch_size"), vision_config.get("patch_size"), nested_patch_size), 16)
    merge_size = scalar_int(
        first_defined(
            image_processor.get("merge_size"),
            image_processor.get("merge_kernel_size"),
            media_processor.get("merge_kernel_size"),
            image_compression.get("spatial_merge_size"),
            vision_config.get("spatial_merge_size"),
            processor_config.get("spatial_merge_size"),
            config.get("spatial_merge_size"),
            processor_config.get("downsample_ratio"),
        ),
        1,
    )
    temporal_patch = scalar_int(
        first_defined(
            video_processor.get("temporal_patch_size"),
            image_processor.get("temporal_patch_size"),
            media_processor.get("temporal_merge_kernel_size"),
            image_compression.get("temporal_patch_size"),
            vision_config.get("temporal_patch_size"),
        ),
        1,
    )
    vision_hidden = int(
        first_defined(
            vision_config.get("hidden_size"),
            max_nested_numeric(vision_config.get("width"), "width") if isinstance(vision_config.get("width"), dict) else None,
            projector_config.get("input_dim"),
            language_hidden,
        )
        or language_hidden
    )
    vision_output_hidden = int(first_defined(vision_config.get("out_hidden_size"), projector_config.get("input_dim"), vision_hidden) or vision_hidden)
    projector_hidden = int(first_defined(projector_config.get("n_embed"), vision_config.get("out_hidden_size"), language_hidden) or language_hidden)
    vision_depth = int(
        first_defined(
            vision_config.get("depth"),
            vision_config.get("num_hidden_layers"),
            max_nested_numeric(vision_config.get("width"), "layers") if isinstance(vision_config.get("width"), dict) else None,
            0,
        )
        or 0
    )

    default_height, default_width = infer_default_image_size(config, vision_config, image_processor if isinstance(image_processor, dict) else {})
    has_video = bool(
        config.get("video_token_id")
        or config.get("video_token_index")
        or processor_config.get("video_token")
        or video_processor
        or video_processor_config
        or vision_config.get("temporal_patch_size")
        or image_compression.get("temporal_patch_size")
        or media_processor.get("temporal_merge_kernel_size")
    )
    has_audio = bool(audio_feature_extractor)
    modality_options = ["image"]
    if has_video:
        modality_options.extend(["video", "image+video"])
    if has_audio:
        modality_options.append("audio")
    requested_modality = str(query.get("modality", [modality_options[0]])[0])
    modality = requested_modality if requested_modality in modality_options else modality_options[0]

    batch = clamp_int(query.get("batch", [1])[0], 1)
    seq_len = clamp_int(query.get("seq_len", [min(max_position, 1024)])[0], min(max_position, 1024), maximum=max_position)
    image_height = clamp_int(query.get("image_height", [default_height])[0], default_height)
    image_width = clamp_int(query.get("image_width", [default_width])[0], default_width)
    frames = clamp_int(query.get("frames", [max(temporal_patch, 8)])[0], max(temporal_patch, 8))
    audio_ms_per_token = scalar_int(processor_config.get("audio_ms_per_token"), 40)
    max_audio_tokens = scalar_int(processor_config.get("audio_seq_length"), 0)
    max_audio_seconds = max(ceil_div(max_audio_tokens * audio_ms_per_token, 1000), 1) if max_audio_tokens else 60
    audio_seconds = clamp_int(query.get("audio_seconds", [min(max_audio_seconds, 30)])[0], min(max_audio_seconds, 30), maximum=max_audio_seconds)
    audio_sample_rate = scalar_int(audio_feature_extractor.get("sampling_rate"), 16000)
    audio_hop_length = scalar_int(audio_feature_extractor.get("hop_length"), max(audio_sample_rate // 100, 1))
    audio_feature_size = scalar_int(audio_feature_extractor.get("feature_size"), 80)
    audio_sample_count = audio_seconds * audio_sample_rate
    audio_feature_frames = ceil_div(audio_sample_count, audio_hop_length)
    calculated_audio_tokens = ceil_div(audio_seconds * 1000, audio_ms_per_token)
    audio_token_count = min(calculated_audio_tokens, max_audio_tokens) if max_audio_tokens else calculated_audio_tokens

    image_patch_rows = ceil_div(image_height, patch_size)
    image_patch_cols = ceil_div(image_width, patch_size)
    image_patch_count = image_patch_rows * image_patch_cols
    merged_patch_rows = ceil_div(image_patch_rows, max(merge_size, 1))
    merged_patch_cols = ceil_div(image_patch_cols, max(merge_size, 1))
    grid_image_token_count = merged_patch_rows * merged_patch_cols
    fixed_image_tokens = scalar_int(
        first_defined(
            config.get("image_seq_length"),
            config.get("vision_soft_tokens_per_image"),
            config.get("image_token_len"),
            image_processor.get("image_seq_length"),
            processor_config.get("image_seq_length"),
        ),
        0,
    )
    image_token_count = fixed_image_tokens or grid_image_token_count
    video_frame_groups = ceil_div(frames, temporal_patch)
    video_patch_count = video_frame_groups * image_patch_rows * image_patch_cols
    fixed_video_tokens_per_frame = scalar_int(first_defined(video_processor.get("max_soft_tokens"), video_processor.get("image_seq_length")), 0)
    if fixed_video_tokens_per_frame:
        video_token_count = frames * fixed_video_tokens_per_frame
    elif fixed_image_tokens:
        video_token_count = video_frame_groups * fixed_image_tokens
    else:
        video_token_count = video_frame_groups * grid_image_token_count
    selected_image_tokens = image_token_count if modality in {"image", "image+video"} else 0
    selected_video_tokens = video_token_count if modality in {"video", "image+video"} else 0
    selected_audio_tokens = audio_token_count if modality == "audio" else 0
    total_tokens = seq_len + selected_image_tokens + selected_video_tokens + selected_audio_tokens
    image_patch_width = 3 * patch_size * patch_size
    video_patch_width = image_patch_width * temporal_patch

    warnings = [
        "视觉 token 数根据 patch_size、merge_size 和当前分辨率估算，实际实现可能包含额外裁剪或动态采样。"
    ]
    if fixed_image_tokens:
        warnings.append(f"配置显式声明每图 {fixed_image_tokens} 个视觉 token，融合总量优先使用该值；patch 网格仅用于展示视觉编码过程。")
    layer_pattern = summarize_layer_pattern(text_config.get("layer_types"))
    if layer_pattern:
        warnings.append(f"文本主干层型摘要: {layer_pattern}。")

    lanes = [
        ("inputs", "输入"),
        ("processing", "处理"),
        ("encoding", "编码"),
        ("fusion", "融合"),
        ("backbone", "主干"),
        ("output", "输出"),
    ]

    nodes = [
        build_node(
            "text_input",
            "inputs",
            0,
            "文本输入",
            "token ids",
            "文本序列进入词嵌入前的离散 token。",
            shape(batch, seq_len),
            shape(batch, seq_len),
            ["text", f"ctx {max_position}"],
            [detail("batch", batch), detail("seq_len", seq_len)],
            [section("输入", [detail("token sequence", shape(batch, seq_len))])],
            "input",
        ),
        build_node(
            "image_input",
            "inputs",
            1,
            "图像输入",
            "RGB 图像",
            "视觉分支的空间输入。",
            shape(batch, 3, image_height, image_width),
            shape(batch, 3, image_height, image_width),
            ["image", f"patch {patch_size}"],
            [detail("height", image_height), detail("width", image_width), detail("patch_size", patch_size)],
            [section("输入", [detail("image tensor", shape(batch, 3, image_height, image_width))])],
            "vision",
        ),
        build_node(
            "text_embedding",
            "processing",
            0,
            "文本嵌入",
            "token -> hidden",
            "把文本 token 映射到语言主干的隐藏维度。",
            shape(batch, seq_len),
            shape(batch, seq_len, language_hidden),
            [f"hidden {language_hidden}", f"vocab {vocab_size or '?'}"],
            [detail("hidden_size", language_hidden), detail("vocab_size", vocab_size)],
            [section("输出张量", [detail("text hidden", shape(batch, seq_len, language_hidden))])],
            "text",
        ),
        build_node(
            "image_processor",
            "processing",
            1,
            "视觉预处理",
            "resize / patch / normalize",
            "将图像重采样并切分为 patch 序列。",
            shape(batch, 3, image_height, image_width),
            shape(batch, image_patch_count, image_patch_width),
            [f"patches {image_patch_count}", f"merge {merge_size}"],
            [detail("raw_patch_count", image_patch_count), detail("patch_feature_width", image_patch_width), detail("merged_token_count", image_token_count), detail("grid_merged_token_count", grid_image_token_count), detail("effective_token_count", image_token_count)],
            [
                section("预处理参数", [detail("merge_size", merge_size), detail("patch_size", patch_size)]),
                section(
                    "推导公式",
                    [
                        detail("raw patches", f"ceil({image_height}/{patch_size}) * ceil({image_width}/{patch_size}) = {image_patch_count}"),
                        detail("patch width", f"3 * {patch_size} * {patch_size} = {image_patch_width}"),
                        detail("grid merged tokens", f"ceil({image_patch_rows}/{max(merge_size, 1)}) * ceil({image_patch_cols}/{max(merge_size, 1)}) = {grid_image_token_count}"),
                        detail("effective tokens", fixed_image_tokens or "使用网格结果"),
                    ],
                ),
            ],
            "vision",
        ),
        build_node(
            "image_encoder",
            "encoding",
            1,
            "视觉编码器",
            f"{vision_depth or '?'} 层视觉主干",
            "将视觉 patch 编码到视觉隐藏空间。",
            shape(batch, image_patch_count, image_patch_width),
            shape(batch, image_token_count, vision_output_hidden),
            [f"vision {vision_hidden}", f"tokens {image_token_count}"],
            [detail("backbone_hidden", vision_hidden), detail("output_hidden", vision_output_hidden), detail("vision_depth", vision_depth), detail("merged_tokens", image_token_count)],
            [section("视觉流", [detail("encoder output", shape(batch, image_token_count, vision_output_hidden))])],
            "vision",
        ),
        build_node(
            "image_projector",
            "fusion",
            1,
            "多模态投影",
            "vision -> language hidden",
            "把视觉特征映射到语言主干维度。",
            shape(batch, image_token_count, vision_output_hidden),
            shape(batch, image_token_count, projector_hidden),
            [f"to {projector_hidden}", "projector"],
            [detail("projector_out", projector_hidden), detail("vision_tokens", image_token_count)],
            [
                section("投影结果", [detail("projected image tokens", shape(batch, image_token_count, projector_hidden))]),
                section("推导公式", [detail("projector", f"[B, V, vision_hidden] -> [B, V, language_hidden] = {shape(batch, image_token_count, projector_hidden)}")]),
            ],
            "fusion",
        ),
        build_node(
            "fusion_context",
            "fusion",
            2,
            "上下文拼接",
            "text + vision tokens",
            "把文本 token 与视觉 token 合并到统一序列。",
            shape(batch, seq_len, language_hidden),
            shape(batch, total_tokens, language_hidden),
            [f"total {total_tokens}", "fusion"],
            [detail("modality", modality), detail("text_tokens", seq_len), detail("image_tokens", selected_image_tokens), detail("video_tokens", selected_video_tokens), detail("audio_tokens", selected_audio_tokens), detail("total_tokens", total_tokens)],
            [
                section("融合后序列", [detail("combined hidden", shape(batch, total_tokens, language_hidden))]),
                section(
                    "推导公式",
                    [
                        detail("token budget", f"text {seq_len} + image {selected_image_tokens} + video {selected_video_tokens} + audio {selected_audio_tokens} = {total_tokens}"),
                        detail("fusion output", f"[B, total_tokens, H] = {shape(batch, total_tokens, language_hidden)}"),
                    ],
                ),
            ],
            "fusion",
            view_modes=["summary", "expanded", "repeat"],
        ),
        build_node(
            "language_backbone",
            "backbone",
            0,
            "语言主干",
            f"{num_layers or '?'} 层条件生成主干",
            "语言层对统一序列执行注意力与条件生成。",
            shape(batch, total_tokens, language_hidden),
            shape(batch, total_tokens, language_hidden),
            [f"layers {num_layers or '?'}", architecture],
            [detail("num_hidden_layers", num_layers), detail("num_attention_heads", num_heads), detail("head_dim", head_dim), detail("hidden_size", language_hidden)],
            [
                section(
                    "层内摘要",
                    [
                        detail("hidden stream", shape(batch, total_tokens, language_hidden)),
                        detail("attention view", shape(batch, total_tokens, num_heads or "heads", head_dim or "head_dim")),
                        detail("layer pattern", layer_pattern or "standard attention"),
                    ],
                )
            ],
            "core",
            micro_flow=[
                "Cross-modal context build",
                f"Attention {num_heads or '?'} x {head_dim or '?'}",
                "Conditional decoding",
            ],
        ),
        build_node(
            "lm_head",
            "output",
            0,
            "输出头",
            "hidden -> vocab",
            "将融合后的隐藏态映射到词表分布。",
            shape(batch, total_tokens, language_hidden),
            shape(batch, total_tokens, vocab_size or "vocab"),
            [f"vocab {vocab_size or '?'}", "head"],
            [detail("vocab_size", vocab_size), detail("output", shape(batch, total_tokens, vocab_size or "vocab"))],
            [section("输出", [detail("logits", shape(batch, total_tokens, vocab_size or "vocab"))])],
            "head",
            view_modes=["summary", "expanded", "repeat"],
        ),
    ]

    if has_video:
        nodes.extend(
            [
                build_node(
                    "video_input",
                    "inputs",
                    2,
                    "视频输入",
                    "frames x RGB",
                    "时序视觉分支输入。",
                    shape(batch, frames, 3, image_height, image_width),
                    shape(batch, frames, 3, image_height, image_width),
                    ["video", f"tp {temporal_patch}"],
                    [detail("frames", frames), detail("temporal_patch", temporal_patch)],
                    [section("输入", [detail("video tensor", shape(batch, frames, 3, image_height, image_width))])],
                    "vision",
                ),
                build_node(
                    "video_processor",
                    "processing",
                    2,
                    "视频预处理",
                    "sample / patch / normalize",
                    "将视频帧组切分为时空 patch。",
                    shape(batch, frames, 3, image_height, image_width),
                    shape(batch, video_patch_count, video_patch_width),
                    [f"patches {video_patch_count}", f"tokens {video_token_count}"],
                    [detail("frame_groups", video_frame_groups), detail("raw_patch_count", video_patch_count), detail("merged_token_count", video_token_count), detail("effective_token_count", video_token_count)],
                    [
                        section("时空 patch", [detail("patch width", video_patch_width), detail("temporal_patch", temporal_patch)]),
                        section(
                            "推导公式",
                            [
                                detail("frame groups", f"ceil({frames}/{temporal_patch}) = {video_frame_groups}"),
                                detail("raw patches", f"{video_frame_groups} * ceil({image_height}/{patch_size}) * ceil({image_width}/{patch_size}) = {video_patch_count}"),
                                detail("effective video tokens", video_token_count),
                            ],
                        ),
                    ],
                    "vision",
                ),
                build_node(
                    "video_encoder",
                    "encoding",
                    2,
                    "视频编码器",
                    "时空特征编码",
                    "与视觉编码器共享或复用类似结构，将视频 patch 编码为 token。",
                    shape(batch, video_patch_count, video_patch_width),
                    shape(batch, video_token_count, vision_output_hidden),
                    [f"vision {vision_hidden}", f"tokens {video_token_count}"],
                    [detail("backbone_hidden", vision_hidden), detail("output_hidden", vision_output_hidden), detail("video_tokens", video_token_count)],
                    [section("输出张量", [detail("video encoder output", shape(batch, video_token_count, vision_output_hidden))])],
                    "vision",
                ),
                build_node(
                    "video_projector",
                    "fusion",
                    0,
                    "视频投影",
                    "video -> language hidden",
                    "将视频特征映射到语言主干维度。",
                    shape(batch, video_token_count, vision_output_hidden),
                    shape(batch, video_token_count, projector_hidden),
                    [f"to {projector_hidden}", "projector"],
                    [detail("projector_out", projector_hidden), detail("video_tokens", video_token_count)],
                    [section("投影结果", [detail("projected video tokens", shape(batch, video_token_count, projector_hidden))])],
                    "fusion",
                ),
            ]
        )

    if has_audio:
        nodes.extend(
            [
                build_node(
                    "audio_input",
                    "inputs",
                    3,
                    "音频输入",
                    f"{audio_sample_rate} Hz waveform",
                    "通用多模态模型的音频输入分支。",
                    shape(batch, audio_sample_count),
                    shape(batch, audio_sample_count),
                    [f"{audio_seconds}s", f"{audio_sample_rate} Hz"],
                    [detail("samples", audio_sample_count), detail("duration_seconds", audio_seconds)],
                    [section("输入", [detail("waveform", shape(batch, audio_sample_count))])],
                    "audio",
                ),
                build_node(
                    "audio_processor",
                    "processing",
                    3,
                    "音频预处理",
                    "log-mel / soft tokens",
                    "先提取声学帧，再压缩到模型声明的音频 token 预算。",
                    shape(batch, audio_sample_count),
                    shape(batch, audio_feature_frames, audio_feature_size),
                    [f"frames {audio_feature_frames}", f"tokens {audio_token_count}"],
                    [detail("hop_length", audio_hop_length), detail("feature_frames", audio_feature_frames), detail("feature_size", audio_feature_size), detail("audio_ms_per_token", audio_ms_per_token), detail("audio_tokens", audio_token_count)],
                    [section("推导公式", [detail("feature frames", f"ceil({audio_sample_count}/{audio_hop_length}) = {audio_feature_frames}"), detail("soft tokens", f"ceil({audio_seconds}*1000/{audio_ms_per_token}) = {calculated_audio_tokens}, cap {max_audio_tokens or 'none'} -> {audio_token_count}")])],
                    "audio",
                ),
                build_node(
                    "audio_projector",
                    "fusion",
                    3,
                    "音频投影",
                    "audio -> language hidden",
                    "将声学表示映射为语言主干可消费的 soft tokens。",
                    shape(batch, audio_feature_frames, audio_feature_size),
                    shape(batch, audio_token_count, language_hidden),
                    [f"tokens {audio_token_count}", f"to {language_hidden}"],
                    [detail("audio_tokens", audio_token_count), detail("language_hidden", language_hidden)],
                    [section("输出", [detail("projected audio tokens", shape(batch, audio_token_count, language_hidden))])],
                    "fusion",
                ),
            ]
        )

    nodes.append(
        build_node(
            "logits",
            "output",
            1,
            "生成输出",
            "词表分布 / 解码结果",
            "最终经解码策略得到回答、OCR 文本或 agent 响应。",
            shape(batch, total_tokens, vocab_size or "vocab"),
            shape(batch, total_tokens, vocab_size or "vocab"),
            ["output"],
            [detail("logits", shape(batch, total_tokens, vocab_size or "vocab"))],
            [section("说明", [detail("decode", "sampling / beam / greedy")])],
            "output",
        )
    )

    lang_hidden_shape = shape(batch, total_tokens, language_hidden)
    vision_hidden_shape = shape(batch, image_token_count, vision_hidden)

    mm_block_nodes = [
        build_node(
            "vision_patch_embed", "encoding", 2,
            "Vision Patch Embedding",
            "patch → token",
            "将视觉 patch 编码为 token 序列。",
            shape(batch, image_patch_count, image_patch_width),
            vision_hidden_shape,
            [f"patch {patch_size}", f"tokens {image_token_count}"],
            [detail("patch_width", image_patch_width), detail("vision_hidden", vision_hidden)],
            [section("嵌入", [detail("output", vision_hidden_shape)])],
            "vision", parent_id="image_encoder", view_modes=["block"],
        ),
        build_node(
            "vision_self_attn", "encoding", 3,
            "Vision Self-Attention",
            f"{vision_depth or '?'} 层视觉注意力",
            "视觉编码器内部的自注意力层。",
            vision_hidden_shape, vision_hidden_shape,
            [f"depth {vision_depth or '?'}", "self-attn"],
            [detail("input", vision_hidden_shape), detail("output", vision_hidden_shape)],
            [section("注意力", [detail("Q/K/V", vision_hidden_shape)])],
            "vision", parent_id="image_encoder", view_modes=["block"],
        ),
        build_node(
            "vision_ffn", "encoding", 4,
            "Vision FFN",
            "视觉前馈网络",
            "视觉编码器内部的 FFN 层。",
            vision_hidden_shape, vision_hidden_shape,
            ["FFN"],
            [detail("input", vision_hidden_shape), detail("output", vision_hidden_shape)],
            [section("FFN", [detail("hidden", vision_hidden)])],
            "vision", parent_id="image_encoder", view_modes=["block"],
        ),
        build_node(
            "lang_self_attn", "backbone", 1,
            "Language Self-Attention",
            f"{num_heads or '?'} heads",
            "语言主干对统一序列做自注意力。",
            lang_hidden_shape, lang_hidden_shape,
            [f"heads {num_heads or '?'}", f"dim {head_dim or '?'}"],
            [detail("Q/K/V", lang_hidden_shape)],
            [section("自注意力", [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)])],
            "core", parent_id="language_backbone", view_modes=["block"],
        ),
        build_node(
            "lang_cross_attn", "backbone", 2,
            "Cross-Modal Attention",
            "text attends to vision",
            "语言 token 对视觉 token 做交叉注意力。",
            f"{lang_hidden_shape} + {vision_hidden_shape}", lang_hidden_shape,
            ["cross-attn", "vision→lang"],
            [detail("Q", lang_hidden_shape), detail("K/V", vision_hidden_shape)],
            [section("交叉注意力", [detail("vision_tokens", image_token_count), detail("output", lang_hidden_shape)])],
            "core", parent_id="language_backbone", view_modes=["block"],
        ),
        build_node(
            "lang_ffn", "backbone", 3,
            "Language FFN",
            "MLP / SwiGLU",
            "语言主干的前馈网络。",
            lang_hidden_shape, lang_hidden_shape,
            ["FFN", f"hidden {language_hidden}"],
            [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)],
            [section("FFN", [detail("hidden", language_hidden)])],
            "core", parent_id="language_backbone", view_modes=["block"],
        ),
    ]

    mm_repeat_nodes = []
    mm_repeat_edges = []
    if num_layers and num_layers > 0:
        layer_1_id = "lang_layer_1"
        layer_mid_id = "lang_layer_mid"
        layer_last_id = "lang_layer_last"

        if num_layers <= 2:
            for i in range(1, num_layers + 1):
                nid = f"lang_layer_{i}"
                mm_repeat_nodes.append(build_node(
                    nid, "backbone", 10 + i,
                    f"Layer {i}", f"第 {i} 层",
                    f"第 {i} 层语言主干，执行注意力与 FFN。",
                    lang_hidden_shape, lang_hidden_shape,
                    [f"layer {i}/{num_layers}"],
                    [detail("layer", i), detail("total_layers", num_layers)],
                    [section("层", [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)])],
                    "core", parent_id="language_backbone", view_modes=["repeat"],
                ))
            mm_repeat_edges = [build_edge("fusion_context", layer_1_id, "conditioned seq", ["repeat"])]
            for i in range(2, num_layers + 1):
                mm_repeat_edges.append(build_edge(f"lang_layer_{i-1}", f"lang_layer_{i}", "next layer", ["repeat"]))
            mm_repeat_edges.append(build_edge(f"lang_layer_{num_layers}", "lm_head", "hidden stream", ["repeat"]))
        else:
            mm_repeat_nodes = [
                build_node(
                    layer_1_id, "backbone", 11,
                    "Layer 1", "首层",
                    "第一层语言主干。",
                    lang_hidden_shape, lang_hidden_shape,
                    [f"layer 1/{num_layers}"],
                    [detail("layer", 1), detail("total_layers", num_layers)],
                    [section("层", [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)])],
                    "core", parent_id="language_backbone", view_modes=["repeat"],
                ),
                build_node(
                    layer_mid_id, "backbone", 12,
                    f"Layers 2..{num_layers - 1}", "中间层",
                    f"中间 {num_layers - 2} 层语言主干。",
                    lang_hidden_shape, lang_hidden_shape,
                    [f"layers 2..{num_layers - 1}"],
                    [detail("layer_range", f"2..{num_layers - 1}"), detail("total_layers", num_layers)],
                    [section("层", [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)])],
                    "core", parent_id="language_backbone", view_modes=["repeat"],
                ),
                build_node(
                    layer_last_id, "backbone", 13,
                    f"Layer {num_layers}", "末层",
                    f"第 {num_layers} 层语言主干。",
                    lang_hidden_shape, lang_hidden_shape,
                    [f"layer {num_layers}/{num_layers}"],
                    [detail("layer", num_layers), detail("total_layers", num_layers)],
                    [section("层", [detail("input", lang_hidden_shape), detail("output", lang_hidden_shape)])],
                    "core", parent_id="language_backbone", view_modes=["repeat"],
                ),
            ]
            mm_repeat_edges = [
                build_edge("fusion_context", layer_1_id, "conditioned seq", ["repeat"]),
                build_edge(layer_1_id, layer_mid_id, "next layer", ["repeat"]),
                build_edge(layer_mid_id, layer_last_id, "next layer", ["repeat"]),
                build_edge(layer_last_id, "lm_head", "hidden stream", ["repeat"]),
            ]

    nodes.extend(mm_block_nodes)
    nodes.extend(mm_repeat_nodes)

    edges = [
        {"source": "text_input", "target": "text_embedding", "label": "token ids"},
        {"source": "text_embedding", "target": "fusion_context", "label": "text hidden", "viewModes": ["summary", "expanded"]},
        {"source": "image_input", "target": "image_processor", "label": "image tensor", "viewModes": ["summary", "expanded"]},
        {"source": "image_processor", "target": "image_encoder", "label": "patches", "viewModes": ["summary", "expanded"]},
        {"source": "image_encoder", "target": "image_projector", "label": "vision hidden", "viewModes": ["summary", "expanded"]},
        {"source": "image_projector", "target": "fusion_context", "label": "image tokens", "viewModes": ["summary", "expanded"]},
        {"source": "fusion_context", "target": "language_backbone", "label": "conditioned sequence", "viewModes": ["summary", "expanded"]},
        {"source": "language_backbone", "target": "lm_head", "label": "hidden stream", "viewModes": ["summary", "expanded"]},
        {"source": "lm_head", "target": "logits", "label": "vocab projection"},
        build_edge("image_encoder", "vision_patch_embed", "patches", ["block"]),
        build_edge("vision_patch_embed", "vision_self_attn", "vision tokens", ["block"]),
        build_edge("vision_self_attn", "vision_ffn", "attn out", ["block"]),
        build_edge("vision_ffn", "image_projector", "vision hidden", ["block"]),
        build_edge("fusion_context", "lang_self_attn", "unified seq", ["block"]),
        build_edge("lang_self_attn", "lang_cross_attn", "self out", ["block"]),
        build_edge("image_projector", "lang_cross_attn", "vision K/V", ["block"]),
        build_edge("lang_cross_attn", "lang_ffn", "cross out", ["block"]),
        build_edge("lang_ffn", "lm_head", "hidden out", ["block"]),
        *mm_repeat_edges,
    ]

    if has_video:
        edges.extend(
            [
                {"source": "video_input", "target": "video_processor", "label": "video tensor"},
                {"source": "video_processor", "target": "video_encoder", "label": "space-time patches"},
                {"source": "video_encoder", "target": "video_projector", "label": "vision hidden"},
                {"source": "video_projector", "target": "fusion_context", "label": "video tokens"},
            ]
        )
    if has_audio:
        edges.extend(
            [
                {"source": "audio_input", "target": "audio_processor", "label": "waveform"},
                {"source": "audio_processor", "target": "audio_projector", "label": "audio features"},
                {"source": "audio_projector", "target": "fusion_context", "label": "audio tokens"},
            ]
        )

    sources: list[str] = []
    append_source_file(model_dir, sources, "README.md")
    append_source_file(config_dir, sources, "config.json")
    append_source_file(config_dir, sources, "configuration.json")
    append_source_file(config_dir, sources, "processor_config.json")
    append_source_file(config_dir, sources, "preprocessor_config.json")
    append_source_file(config_dir, sources, "video_preprocessor_config.json")
    append_source_file(config_dir, sources, "params.json")
    if inherited_base:
        warnings.append(f"当前目录没有独立架构配置，按模型卡 base_model={inherited_base} 复用主干结构；适配器或 GGUF 的存储格式不改变张量 shape。")

    summary = [
        detail("类型", "多模态"),
        detail("语言隐藏维度", language_hidden),
        detail("视觉隐藏维度", vision_hidden),
        detail("当前模态", modality),
        detail("主干层数", num_layers),
        detail("patch size", patch_size),
        detail("当前总 tokens", total_tokens),
    ]
    if has_video:
        summary.append(detail("视频", f"temporal patch {temporal_patch}"))
    if has_audio:
        summary.append(detail("音频", f"{audio_token_count} tokens / {audio_seconds}s"))

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 8, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "文本 token", "type": "number", "value": seq_len, "min": 1, "max": max_position, "step": 1, "help": "文本序列长度"},
        {"name": "image_height", "label": "图像高", "type": "number", "value": image_height, "min": patch_size, "max": 8192, "step": patch_size, "help": "输入图像高度"},
        {"name": "image_width", "label": "图像宽", "type": "number", "value": image_width, "min": patch_size, "max": 8192, "step": patch_size, "help": "输入图像宽度"},
    ]
    if len(modality_options) > 1:
        controls.insert(1, {"name": "modality", "label": "输入模态", "type": "select", "value": modality, "options": modality_options, "help": "只把当前选择的模态分支计入 token 总量"})
    if has_video:
        controls.append({"name": "frames", "label": "视频帧数", "type": "number", "value": frames, "min": temporal_patch, "max": 1024, "step": temporal_patch, "help": "采样到模型的帧数"})
    if has_audio:
        controls.append({"name": "audio_seconds", "label": "音频秒数", "type": "number", "value": audio_seconds, "min": 1, "max": max_audio_seconds, "step": 1, "help": "当前音频输入时长"})

    headline = f"多模态条件生成模型，语言隐藏维度 {language_hidden}，视觉 token 通过投影并入同一序列。"
    return base_model_payload(
        model_id,
        "multimodal",
        architecture,
        headline,
        summary,
        controls,
        {
            "batch": batch,
            "modality": modality,
            "seq_len": seq_len,
            "image_height": image_height,
            "image_width": image_width,
            "frames": frames if has_video else 0,
            "audio_seconds": audio_seconds if has_audio else 0,
            "audio_tokens": audio_token_count if has_audio else 0,
        },
        build_graph(lanes, nodes, edges),
        warnings,
        sources,
        "fusion_context",
    )


def load_component_config(model_dir: Path, component_name: str | None) -> tuple[dict[str, Any], str | None]:
    if not component_name:
        return {}, None

    component_dir = model_dir / component_name
    for filename in ("config.json", "configuration.json", "scheduler_config.json", "processor_config.json", "preprocessor_config.json"):
        path = component_dir / filename
        if path.exists():
            return read_json_file(path), f"{component_name}/{filename}"
    return {}, None


def infer_text_hidden(text_component_config: dict[str, Any], transformer_config: dict[str, Any]) -> int:
    nested = text_component_config.get("text_config") if isinstance(text_component_config.get("text_config"), dict) else {}
    return int(
        first_defined(
            nested.get("hidden_size"),
            text_component_config.get("hidden_size"),
            transformer_config.get("text_hidden_dim"),
            transformer_config.get("caption_channels"),
            transformer_config.get("text_dim"),
            0,
        )
        or 2048
    )


def infer_text_layers(text_component_config: dict[str, Any], transformer_config: dict[str, Any]) -> int:
    nested = text_component_config.get("text_config") if isinstance(text_component_config.get("text_config"), dict) else {}
    return int(first_defined(nested.get("num_hidden_layers"), text_component_config.get("num_hidden_layers"), transformer_config.get("num_text_layers"), 0) or 0)


def infer_transformer_width(transformer_config: dict[str, Any]) -> int:
    hidden_size = first_defined(
        transformer_config.get("hidden_size"),
        transformer_config.get("dim"),
        transformer_config.get("model_dim"),
        transformer_config.get("inner_dim"),
    )
    if isinstance(hidden_size, (int, float)):
        return int(hidden_size)

    num_heads = first_defined(
        transformer_config.get("num_attention_heads"),
        transformer_config.get("num_heads"),
        transformer_config.get("n_heads"),
        transformer_config.get("text_num_attention_heads"),
    )
    head_dim = first_defined(transformer_config.get("attention_head_dim"), transformer_config.get("head_dim"))
    if isinstance(num_heads, (int, float)) and isinstance(head_dim, (int, float)):
        return int(num_heads) * int(head_dim)

    in_channels = transformer_config.get("in_channels")
    if isinstance(in_channels, (int, float)):
        return int(in_channels)

    text_hidden = transformer_config.get("text_hidden_dim")
    if isinstance(text_hidden, (int, float)):
        return int(text_hidden)

    return 0


def infer_transformer_layers(transformer_config: dict[str, Any]) -> int:
    if isinstance(transformer_config.get("n_layers"), (int, float)):
        return int(transformer_config["n_layers"]) + int(transformer_config.get("n_refiner_layers", 0) or 0)
    if isinstance(transformer_config.get("depth"), (int, float)):
        return int(transformer_config["depth"])
    base_layers = int(first_defined(transformer_config.get("num_layers"), transformer_config.get("num_double_stream_layers"), 0) or 0)
    single_layers = int(transformer_config.get("num_single_layers", 0) or 0)
    return base_layers + single_layers


def infer_transformer_heads(transformer_config: dict[str, Any]) -> int:
    return int(first_defined(
        transformer_config.get("num_attention_heads"),
        transformer_config.get("num_heads"),
        transformer_config.get("n_heads"),
        0,
    ) or 0)


def build_diffusers_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config_dir = architecture_config_dir(model_dir)
    inherited_base = model_card_base_reference(model_dir) if config_dir != model_dir else None
    model_index = read_json_file(config_dir / "model_index.json")
    architecture = infer_architecture_name(model_dir, "diffusers")
    component_keys = list(model_index.keys())
    discovered_transformers = discover_diffusion_transformer_configs(config_dir)

    text_component_name = "text_encoder" if "text_encoder" in component_keys else "mllm" if "mllm" in component_keys else None
    processor_component_name = "processor" if "processor" in component_keys else "tokenizer" if "tokenizer" in component_keys else None
    scheduler_component_name = "scheduler" if "scheduler" in component_keys else None
    transformer_component_name = "transformer" if "transformer" in component_keys else None
    vae_component_name = "vae" if "vae" in component_keys else None

    text_component_config, text_source = load_component_config(config_dir, text_component_name)
    processor_component_config, processor_source = load_component_config(config_dir, processor_component_name)
    scheduler_config, scheduler_source = load_component_config(config_dir, scheduler_component_name)
    transformer_config, transformer_source = load_component_config(config_dir, transformer_component_name)
    vae_config, vae_source = load_component_config(config_dir, vae_component_name)
    if not transformer_config and discovered_transformers:
        transformer_config, transformer_source = discovered_transformers[0]
    uses_umt5_xxl_fallback = False
    if not text_component_config and (config_dir / "google" / "umt5-xxl").exists():
        text_component_config = {"hidden_size": 4096, "num_hidden_layers": 24}
        text_source = "README.md"
        uses_umt5_xxl_fallback = True
    if not scheduler_config and (config_dir / "scheduler" / "scheduler_config.json").exists():
        scheduler_config = read_json_file(config_dir / "scheduler" / "scheduler_config.json")
        scheduler_source = "scheduler/scheduler_config.json"

    text_hidden = infer_text_hidden(text_component_config, transformer_config)
    text_layers = infer_text_layers(text_component_config, transformer_config)
    text_nested = text_component_config.get("text_config") if isinstance(text_component_config.get("text_config"), dict) else {}
    max_position = int(first_defined(text_nested.get("max_position_embeddings"), text_component_config.get("max_position_embeddings"), transformer_config.get("model_max_length"), transformer_config.get("text_len"), 4096) or 4096)
    prompt_tokens = clamp_int(query.get("seq_len", [min(max_position, 256)])[0], min(max_position, 256), maximum=max_position)
    batch = clamp_int(query.get("batch", [1])[0], 1)

    sample_size = first_defined(vae_config.get("sample_size"), 1024)
    default_height, default_width = spatial_pair(sample_size, 1024)
    image_height = clamp_int(query.get("image_height", [default_height])[0], default_height)
    image_width = clamp_int(query.get("image_width", [default_width])[0], default_width)
    steps = clamp_int(query.get("steps", [28 if model_index.get("is_distilled") else 40])[0], 28 if model_index.get("is_distilled") else 40)

    transformer_class = str(transformer_config.get("_class_name") or "")
    metadata_config = direct_primary_config(model_dir)
    video_hint = f"{model_id} {architecture} {transformer_class} {transformer_config.get('model_type', '')} {metadata_config.get('task', '')}".lower()
    transformer_model_type = str(transformer_config.get("model_type") or "").lower()
    is_video = "video" in video_hint or "transformer3d" in video_hint or transformer_model_type in {"t2v", "i2v", "ti2v"}
    default_frames = int(first_defined(transformer_config.get("sample_frames"), model_index.get("num_frames"), 81) or 81)
    frames = clamp_int(query.get("frames", [default_frames if is_video else 1])[0], default_frames if is_video else 1)

    vae_scale_height, vae_scale_width = derive_vae_spatial_scales(vae_config)
    if is_video:
        vae_temporal_scale = derive_vae_temporal_scale(vae_config)
        if not vae_config and ("wanmodel" in transformer_class.lower() or "transformer3d" in transformer_class.lower()):
            vae_temporal_scale = scalar_int(transformer_config.get("vae_scale"), 4)
    else:
        vae_temporal_scale = 1
    if "wanmodel" in transformer_class.lower():
        latent_channels = int(first_defined(vae_config.get("latent_channels"), vae_config.get("z_dim"), 16) or 16)
    else:
        latent_channels = int(first_defined(vae_config.get("latent_channels"), vae_config.get("z_dim"), transformer_config.get("out_channels"), transformer_config.get("in_channels"), 4) or 4)
    latent_height = ceil_div(image_height, vae_scale_height)
    latent_width = ceil_div(image_width, vae_scale_width)
    latent_frames = ceil_div(frames, vae_temporal_scale) if is_video else 1
    transformer_patch_value = first_defined(
        transformer_config.get("patch_size"),
        transformer_config.get("all_patch_size"),
        model_index.get("patch_size"),
        [1, 2, 2] if "wanmodel" in transformer_class.lower() else None,
        1,
    )
    transformer_patch_height, transformer_patch_width = spatial_pair(transformer_patch_value, 1)
    transformer_patch_temporal = temporal_patch_size(transformer_patch_value, 1)
    latent_tokens = (
        ceil_div(latent_frames, transformer_patch_temporal)
        * ceil_div(latent_height, transformer_patch_height)
        * ceil_div(latent_width, transformer_patch_width)
    )
    transformer_width = infer_transformer_width(transformer_config)
    transformer_layers = infer_transformer_layers(transformer_config)
    transformer_heads = infer_transformer_heads(transformer_config)
    transformer_head_dim = int(first_defined(
        transformer_config.get("attention_head_dim"),
        transformer_config.get("head_dim"),
        transformer_width // transformer_heads if transformer_width and transformer_heads else 0,
    ) or 0)

    has_conditioning_image = bool(
        processor_component_name == "processor"
        or text_component_name == "mllm"
        or transformer_model_type in {"i2v", "ti2v"}
        or "image-to-video" in str(metadata_config.get("task") or "").lower()
    )
    conditioning_patch = int(first_defined(processor_component_config.get("patch_size"), text_component_config.get("vision_config", {}).get("patch_size") if isinstance(text_component_config.get("vision_config"), dict) else None, 16) or 16)
    conditioning_merge = int(first_defined(processor_component_config.get("merge_size"), text_component_config.get("vision_config", {}).get("spatial_merge_size") if isinstance(text_component_config.get("vision_config"), dict) else None, 1) or 1)
    conditioning_patch_rows = ceil_div(image_height, conditioning_patch)
    conditioning_patch_cols = ceil_div(image_width, conditioning_patch)
    conditioning_raw_patches = conditioning_patch_rows * conditioning_patch_cols
    conditioning_tokens = ceil_div(conditioning_patch_rows, max(conditioning_merge, 1)) * ceil_div(conditioning_patch_cols, max(conditioning_merge, 1))
    audio_channel = scalar_int(transformer_config.get("audio_channel"), 0)
    has_audio_condition = audio_channel > 0
    default_audio_tokens = scalar_int(transformer_config.get("context_tokens"), 32)
    audio_tokens = clamp_int(query.get("audio_tokens", [default_audio_tokens])[0], default_audio_tokens, maximum=4096)
    scheduler_seq_len = int(first_defined(scheduler_config.get("base_image_seq_len"), scheduler_config.get("seq_len"), latent_tokens) or latent_tokens)

    pixel_shape = shape(batch, frames, 3, image_height, image_width) if is_video else shape(batch, 3, image_height, image_width)
    latent_shape = shape(batch, latent_channels, latent_frames, latent_height, latent_width) if is_video else shape(batch, latent_channels, latent_height, latent_width)
    transformer_patch_label = (
        f"{transformer_patch_temporal}x{transformer_patch_height}x{transformer_patch_width}"
        if is_video
        else f"{transformer_patch_height}x{transformer_patch_width}"
    )

    warnings = [
        "Diffusers 图中的 latent token 数由 VAE 下采样倍率和 transformer patch_size 估算。",
        "Scheduler 节点展示的是步数和序列长度摘要，不是单个运行时张量。",
    ]
    if is_video:
        warnings.append("视频 latent token 已包含 VAE 时间压缩与 transformer temporal patch。")
    if len(discovered_transformers) > 1:
        warnings.append(f"检测到 {len(discovered_transformers)} 套去噪 transformer 配置；图中使用第一套共同维度作为单阶段代表，不重复累计 latent token。")
    if has_conditioning_image:
        warnings.append("该 pipeline 含图像条件分支，是否必须输入图像取决于具体调用方式。")
    if has_audio_condition:
        warnings.append("该 pipeline 含音频条件分支；页面使用配置声明的 audio_channel 与 context_tokens 展示条件 shape。")
    if uses_umt5_xxl_fallback:
        warnings.append("仓库只包含 UMT5-XXL tokenizer/权重标识，文本编码器按 UMT5-XXL 的 4096 隐藏维度与 24 层展示。")
    if inherited_base:
        warnings.append(f"当前目录没有独立 pipeline 配置，按模型卡 base_model={inherited_base} 复用基础 diffusion 结构；LoRA 仅改变权重，不改变 latent shape。")

    lanes = [
        ("inputs", "输入"),
        ("conditioning", "条件分支"),
        ("latent", "latent 空间"),
        ("denoise", "去噪主干"),
        ("decode", "解码"),
        ("output", "输出"),
    ]

    nodes = [
        build_node(
            "prompt_input",
            "inputs",
            0,
            "Prompt 输入",
            "文本条件",
            "文本 token 将作为 diffusion denoiser 的条件。",
            shape(batch, prompt_tokens),
            shape(batch, prompt_tokens),
            ["prompt", f"tokens {prompt_tokens}"],
            [detail("batch", batch), detail("prompt_tokens", prompt_tokens)],
            [section("说明", [detail("conditioning", "prompt tokens -> text encoder")])],
            "input",
        ),
        build_node(
            "text_condition",
            "conditioning",
            0,
            "文本条件编码",
            f"{text_layers or '?'} 层文本主干",
            "产生给 diffusion transformer 使用的文本条件隐藏态。",
            shape(batch, prompt_tokens),
            shape(batch, prompt_tokens, text_hidden),
            [f"hidden {text_hidden}", text_component_name or "text"],
            [detail("text_hidden", text_hidden), detail("text_layers", text_layers), detail("max_position", max_position)],
            [section("输出张量", [detail("text hidden", shape(batch, prompt_tokens, text_hidden))])],
            "text",
        ),
        build_node(
            "scheduler",
            "latent",
            0,
            "Scheduler",
            f"{steps} inference steps",
            "定义时间步、sigma 或 shift 策略，驱动去噪迭代。",
            shape(steps),
            shape(steps, batch, scheduler_seq_len),
            [scheduler_config.get("_class_name", "scheduler"), f"seq {scheduler_seq_len}"],
            [detail("inference_steps", steps), detail("base_image_seq_len", scheduler_seq_len), detail("num_train_timesteps", scheduler_config.get("num_train_timesteps"))],
            [section("时间调度", [detail("time_shift_type", scheduler_config.get("time_shift_type", scheduler_config.get("time_shift_version", "-"))), detail("dynamic_shift", first_defined(scheduler_config.get("use_dynamic_shifting"), scheduler_config.get("dynamic_time_shift"), False))])],
            "scheduler",
        ),
    ]

    if has_conditioning_image:
        nodes.extend(
            [
                build_node(
                    "image_condition_input",
                    "inputs",
                    1,
                    "条件图像",
                    "可选视觉输入",
                    "图像编辑或多模态 pipeline 可能需要该输入。",
                    shape(batch, 3, image_height, image_width),
                    shape(batch, 3, image_height, image_width),
                    ["image", f"patch {conditioning_patch}"],
                    [detail("height", image_height), detail("width", image_width)],
                    [section("输入", [detail("image tensor", shape(batch, 3, image_height, image_width))])],
                    "vision",
                ),
                build_node(
                    "image_condition_processor",
                    "conditioning",
                    1,
                    "条件图像处理",
                    "normalize / patch",
                    "将条件图像整理为 mLLM 或 processor 需要的视觉 token。",
                    shape(batch, 3, image_height, image_width),
                    shape(batch, conditioning_raw_patches, 3 * conditioning_patch * conditioning_patch),
                    [f"patches {conditioning_raw_patches}", f"tokens {conditioning_tokens}"],
                    [detail("raw_patch_count", conditioning_raw_patches), detail("merged_token_count", conditioning_tokens), detail("merge_size", conditioning_merge)],
                    [section("预处理", [detail("patch_size", conditioning_patch), detail("merge_size", conditioning_merge)])],
                    "vision",
                ),
            ]
        )

    if has_audio_condition:
        nodes.extend(
            [
                build_node(
                    "audio_condition_input",
                    "inputs",
                    2,
                    "音频条件",
                    "precomputed audio features",
                    "语音驱动视频模型的音频条件序列。",
                    shape(batch, audio_tokens, audio_channel),
                    shape(batch, audio_tokens, audio_channel),
                    [f"tokens {audio_tokens}", f"channels {audio_channel}"],
                    [detail("audio_tokens", audio_tokens), detail("audio_channel", audio_channel)],
                    [section("输入", [detail("audio features", shape(batch, audio_tokens, audio_channel))])],
                    "audio",
                ),
                build_node(
                    "audio_condition_encoder",
                    "conditioning",
                    2,
                    "音频条件编码",
                    f"window {transformer_config.get('audio_window', '?')}",
                    "将音频特征映射为去噪 transformer 的交叉条件。",
                    shape(batch, audio_tokens, audio_channel),
                    shape(batch, audio_tokens, transformer_width),
                    [f"to {transformer_width}"],
                    [detail("audio_blocks", transformer_config.get("audio_block")), detail("output", shape(batch, audio_tokens, transformer_width))],
                    [section("输出", [detail("audio condition", shape(batch, audio_tokens, transformer_width))])],
                    "audio",
                ),
            ]
        )

    latents_input_node = "latent_init"
    latent_input_label = "Latent 初始化"
    latent_input_description = "以高斯噪声 latent 开始迭代去噪。"
    spatial_scale_label = f"{vae_scale_height}x{vae_scale_width}"
    latent_input_badges = [f"spatial 1/{spatial_scale_label}", f"channels {latent_channels}"]
    if is_video:
        latent_input_badges.append(f"temporal 1/{vae_temporal_scale}")
    if has_conditioning_image:
        latent_input_label = "VAE 编码"
        latent_input_description = "将条件图像编码到 latent 空间后参与去噪。"
        latents_input_node = "vae_encode"

    nodes.extend(
        [
            build_node(
                latents_input_node,
                "latent",
                1,
                latent_input_label,
                f"latent {latent_channels} x " + (f"{latent_frames} x " if is_video else "") + f"{latent_height} x {latent_width}",
                latent_input_description,
                pixel_shape if has_conditioning_image else latent_shape,
                latent_shape,
                latent_input_badges,
                [detail("vae_spatial_scale", spatial_scale_label), detail("vae_temporal_scale", vae_temporal_scale), detail("latent_frames", latent_frames), detail("latent_height", latent_height), detail("latent_width", latent_width), detail("latent_channels", latent_channels)],
                [
                    section("latent 形状", [detail("latent tensor", latent_shape), detail("latent tokens", latent_tokens)]),
                    section(
                        "推导公式",
                        [
                            detail("latent frames", f"ceil({frames}/{vae_temporal_scale}) = {latent_frames}" if is_video else "image pipeline = 1"),
                            detail("latent height", f"ceil({image_height}/{vae_scale_height}) = {latent_height}"),
                            detail("latent width", f"ceil({image_width}/{vae_scale_width}) = {latent_width}"),
                            detail("latent tokens", f"ceil({latent_frames}/{transformer_patch_temporal}) * ceil({latent_height}/{transformer_patch_height}) * ceil({latent_width}/{transformer_patch_width}) = {latent_tokens}"),
                        ],
                    ),
                ],
                "latent",
                view_modes=["summary", "expanded", "repeat"],
            ),
            build_node(
                "transformer",
                "denoise",
                0,
                "Diffusion Transformer",
                f"{transformer_layers or '?'} 层去噪主干",
                "在 scheduler 提供的时间步上，结合文本条件对 latent token 进行去噪更新。",
                shape(batch, latent_tokens, transformer_width or "hidden"),
                shape(batch, latent_tokens, transformer_width or "hidden"),
                [architecture, f"layers {transformer_layers or '?'}"],
                [detail("transformer_width", transformer_width), detail("latent_tokens", latent_tokens), detail("num_attention_heads", transformer_heads), detail("head_dim", transformer_head_dim)],
                [
                    section(
                        "条件与 latent",
                        [
                            detail("text condition", shape(batch, prompt_tokens, text_hidden)),
                            detail("latent grid", latent_shape),
                            detail("tokenized latent", shape(batch, latent_tokens, transformer_width or "hidden")),
                        ],
                    ),
                    section(
                        "推导公式",
                        [
                            detail("per step stream", f"[B, latent_tokens, width] = {shape(batch, latent_tokens, transformer_width or 'hidden')}"),
                            detail("conditioning", f"[B, prompt_tokens, text_hidden] = {shape(batch, prompt_tokens, text_hidden)}"),
                            detail("scheduler repeat", f"{steps} steps over latent token stream"),
                        ],
                    ),
                ],
                "core",
                micro_flow=[
                    f"Scheduler x {steps}",
                    f"Attention {transformer_heads or '?'} x {transformer_head_dim or '?'}",
                    "Latent residual update",
                ],
            ),
            build_node(
                "vae_decode",
                "decode",
                0,
                "VAE Decode",
                "latent -> video" if is_video else "latent -> image",
                "把最终 latent 还原到视频像素空间。" if is_video else "把最终 latent 还原到像素空间。",
                latent_shape,
                pixel_shape,
                [f"spatial x{spatial_scale_label}", vae_config.get("_class_name", "vae")],
                [detail("latent_channels", latent_channels), detail("vae_spatial_scale", spatial_scale_label), detail("vae_temporal_scale", vae_temporal_scale), detail("output_size", pixel_shape)],
                [
                    section("解码结果", [detail("video tensor" if is_video else "image tensor", pixel_shape)]),
                    section("推导公式", [detail("decode", f"latent -> pixels = {pixel_shape}")]),
                ],
                "decode",
                view_modes=["summary", "expanded", "repeat"],
            ),
            build_node(
                "image_output",
                "output",
                0,
                "视频输出" if is_video else "图像输出",
                "像素空间结果",
                "生成或编辑后的视频。" if is_video else "生成或编辑后的图像。",
                pixel_shape,
                pixel_shape,
                ["output"],
                [detail("video" if is_video else "image", pixel_shape)],
                [section("说明", [detail("postprocess", "可进一步保存、显示或后处理")])],
                "output",
            ),
        ]
    )

    transformer_width_val = transformer_width or "hidden"
    token_shape = shape(batch, latent_tokens, transformer_width_val)
    text_cond_shape = shape(batch, prompt_tokens, text_hidden)

    block_nodes = [
        build_node(
            "transformer_patchify", "denoise", 1,
            "Latent Patchify",
            "latent → patch tokens",
            "将 latent 空间切分为 patch token 序列。",
            latent_shape, token_shape,
            [f"patch {transformer_patch_label}", f"tokens {latent_tokens}"],
            [detail("latent_shape", latent_shape), detail("token_shape", token_shape)],
            [section("推导公式", [detail("patchify", f"{latent_shape} -> [B, tokens, width] = {token_shape}")])],
            "core", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_timestep", "denoise", 2,
            "Timestep Embedding",
            "t → conditioning",
            "将时间步编码为调制向量。",
            shape(steps), shape(batch, transformer_width_val),
            [f"steps {steps}", "sinusoidal"],
            [detail("inference_steps", steps)],
            [section("输出", [detail("timestep emb", shape(batch, transformer_width_val))])],
            "scheduler", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_self_attn", "denoise", 3,
            "Self-Attention",
            "latent token 间注意力",
            "对 latent patch token 执行自注意力。",
            token_shape, token_shape,
            [f"heads {transformer_heads or '?'}", f"dim {transformer_head_dim or '?'}"],
            [detail("qkv", token_shape), detail("heads", transformer_heads)],
            [section("注意力", [detail("Q/K/V", token_shape), detail("output", token_shape)])],
            "core", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_cross_attn", "denoise", 4,
            "Cross-Attention",
            "latent attends to text",
            "latent token 对文本条件做交叉注意力。",
            f"{token_shape} + {text_cond_shape}", token_shape,
            ["cross-attn", f"text {prompt_tokens}"],
            [detail("Q", token_shape), detail("K/V", text_cond_shape)],
            [section("交叉注意力", [detail("text condition", text_cond_shape), detail("output", token_shape)])],
            "core", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_ffn", "denoise", 5,
            "Feed-Forward",
            "MLP / SwiGLU",
            "对每个 token 做前馈网络变换。",
            token_shape, token_shape,
            ["FFN", f"width {transformer_width_val}"],
            [detail("input", token_shape), detail("output", token_shape)],
            [section("FFN", [detail("hidden", transformer_width_val)])],
            "core", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_adaln", "denoise", 6,
            "AdaLN 调制",
            "timestep → scale/shift",
            "用时间步嵌入调制各子层的归一化参数。",
            shape(batch, transformer_width_val), shape(batch, transformer_width_val),
            ["modulation", "AdaLN"],
            [detail("timestep_emb", shape(batch, transformer_width_val))],
            [section("调制", [detail("scale_shift", "timestep emb → per-block scale/shift")])],
            "scheduler", parent_id="transformer", view_modes=["block"],
        ),
        build_node(
            "transformer_unpatchify", "denoise", 7,
            "Unpatchify",
            "patch tokens → latent",
            "将 patch token 序列还原为 latent 空间。",
            token_shape, latent_shape,
            [f"tokens {latent_tokens}", f"patch {transformer_patch_label}"],
            [detail("token_shape", token_shape), detail("latent_shape", latent_shape)],
            [section("推导公式", [detail("unpatchify", f"[B, tokens, width] -> [B, C, h, w] = {latent_shape}")])],
            "core", parent_id="transformer", view_modes=["block"],
        ),
    ]

    repeat_nodes = []
    repeat_edges = []
    if steps > 0:
        step_1_id = "denoise_step_1"
        step_mid_id = "denoise_step_mid"
        step_last_id = "denoise_step_last"

        if steps <= 2:
            for i in range(1, steps + 1):
                nid = f"denoise_step_{i}"
                repeat_nodes.append(build_node(
                    nid, "denoise", 10 + i,
                    f"Step {i}", f"去噪步 {i}",
                    f"第 {i} 步去噪迭代，transformer 对 latent 做一次更新。",
                    latent_shape, latent_shape,
                    [f"step {i}/{steps}"],
                    [detail("step", i), detail("total_steps", steps)],
                    [section("迭代", [detail("input", latent_shape), detail("output", latent_shape)])],
                    "core", parent_id="transformer", view_modes=["repeat"],
                ))
            repeat_edges = [build_edge(latents_input_node, step_1_id, "initial latent", ["repeat"])]
            for i in range(2, steps + 1):
                repeat_edges.append(build_edge(f"denoise_step_{i-1}", f"denoise_step_{i}", "next step", ["repeat"]))
            repeat_edges.append(build_edge(f"denoise_step_{steps}", "vae_decode", "denoised latent", ["repeat"]))
        else:
            repeat_nodes = [
                build_node(
                    step_1_id, "denoise", 11,
                    "Step 1", "首步去噪",
                    "第一步去噪迭代，从初始噪声或编码 latent 开始。",
                    latent_shape, latent_shape,
                    [f"step 1/{steps}"],
                    [detail("step", 1), detail("total_steps", steps)],
                    [section("迭代", [detail("input", latent_shape), detail("output", latent_shape)])],
                    "core", parent_id="transformer", view_modes=["repeat"],
                ),
                build_node(
                    step_mid_id, "denoise", 12,
                    f"Steps 2..{steps - 1}", "中间步去噪",
                    f"中间 {steps - 2} 步去噪迭代。",
                    latent_shape, latent_shape,
                    [f"steps 2..{steps - 1}"],
                    [detail("step_range", f"2..{steps - 1}"), detail("total_steps", steps)],
                    [section("迭代", [detail("input", latent_shape), detail("output", latent_shape)])],
                    "core", parent_id="transformer", view_modes=["repeat"],
                ),
                build_node(
                    step_last_id, "denoise", 13,
                    f"Step {steps}", "末步去噪",
                    f"第 {steps} 步去噪迭代，输出最终去噪 latent。",
                    latent_shape, latent_shape,
                    [f"step {steps}/{steps}"],
                    [detail("step", steps), detail("total_steps", steps)],
                    [section("迭代", [detail("input", latent_shape), detail("output", latent_shape)])],
                    "core", parent_id="transformer", view_modes=["repeat"],
                ),
            ]
            repeat_edges = [
                build_edge(latents_input_node, step_1_id, "initial latent", ["repeat"]),
                build_edge(step_1_id, step_mid_id, "next step", ["repeat"]),
                build_edge(step_mid_id, step_last_id, "next step", ["repeat"]),
                build_edge(step_last_id, "vae_decode", "denoised latent", ["repeat"]),
            ]

    nodes.extend(block_nodes)
    nodes.extend(repeat_nodes)

    edges = [
        {"source": "prompt_input", "target": "text_condition", "label": "prompt"},
        {"source": "text_condition", "target": "transformer", "label": "text hidden", "viewModes": ["summary", "expanded"]},
        {"source": "scheduler", "target": "transformer", "label": "timesteps", "viewModes": ["summary", "expanded"]},
        {"source": latents_input_node, "target": "transformer", "label": "latent tokens", "viewModes": ["summary", "expanded"]},
        {"source": "transformer", "target": "vae_decode", "label": "denoised latent", "viewModes": ["summary", "expanded"]},
        {"source": "vae_decode", "target": "image_output", "label": "pixels"},
        build_edge(latents_input_node, "transformer_patchify", "latent input", ["block"]),
        build_edge("scheduler", "transformer_timestep", "t → emb", ["block"]),
        build_edge("text_condition", "transformer_cross_attn", "text K/V", ["block"]),
        build_edge("transformer_patchify", "transformer_self_attn", "patch tokens", ["block"]),
        build_edge("transformer_timestep", "transformer_adaln", "timestep emb", ["block"]),
        build_edge("transformer_adaln", "transformer_self_attn", "modulate", ["block"]),
        build_edge("transformer_adaln", "transformer_cross_attn", "modulate", ["block"]),
        build_edge("transformer_adaln", "transformer_ffn", "modulate", ["block"]),
        build_edge("transformer_self_attn", "transformer_cross_attn", "self out", ["block"]),
        build_edge("transformer_cross_attn", "transformer_ffn", "cross out", ["block"]),
        build_edge("transformer_ffn", "transformer_unpatchify", "ffn out", ["block"]),
        build_edge("transformer_unpatchify", "vae_decode", "latent out", ["block"]),
        *repeat_edges,
    ]

    if has_conditioning_image:
        edges.extend(
            [
                {"source": "image_condition_input", "target": "image_condition_processor", "label": "conditioning image"},
                {"source": "image_condition_processor", "target": "vae_encode", "label": "prepared image"},
            ]
        )
    if has_audio_condition:
        edges.extend(
            [
                {"source": "audio_condition_input", "target": "audio_condition_encoder", "label": "audio features"},
                {"source": "audio_condition_encoder", "target": "transformer", "label": "audio hidden"},
            ]
        )

    sources: list[str] = []
    append_source_file(model_dir, sources, "README.md")
    append_source_file(config_dir, sources, "model_index.json")
    if text_source:
        append_source_file(config_dir, sources, text_source)
    if processor_source:
        append_source_file(config_dir, sources, processor_source)
    if scheduler_source:
        append_source_file(config_dir, sources, scheduler_source)
    if transformer_source:
        append_source_file(config_dir, sources, transformer_source)
    for _, discovered_source in discovered_transformers:
        append_source_file(config_dir, sources, discovered_source)
    if vae_source:
        append_source_file(config_dir, sources, vae_source)

    summary = [
        detail("类型", "Diffusers"),
        detail("Transformer 宽度", transformer_width),
        detail("去噪层数", transformer_layers),
        detail("推理步数", steps),
        detail("latent channels", latent_channels),
        detail("latent frames", latent_frames) if is_video else detail("VAE spatial scale", spatial_scale_label),
        detail("latent tokens", latent_tokens),
        detail("scheduler", scheduler_config.get("_class_name", "-")),
    ]
    if len(discovered_transformers) > 1:
        summary.append(detail("去噪阶段", len(discovered_transformers)))
    if has_audio_condition:
        summary.append(detail("音频条件", f"{audio_tokens} x {audio_channel}"))

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 8, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "Prompt token", "type": "number", "value": prompt_tokens, "min": 1, "max": max_position, "step": 1, "help": "文本条件 token 数"},
        {"name": "image_height", "label": "图像高", "type": "number", "value": image_height, "min": 64, "max": 4096, "step": 8, "help": "输出图像高度"},
        {"name": "image_width", "label": "图像宽", "type": "number", "value": image_width, "min": 64, "max": 4096, "step": 8, "help": "输出图像宽度"},
        {"name": "steps", "label": "推理步数", "type": "number", "value": steps, "min": 1, "max": 200, "step": 1, "help": "scheduler 迭代步数"},
    ]
    if is_video:
        controls.insert(4, {"name": "frames", "label": "视频帧数", "type": "number", "value": frames, "min": 1, "max": 1024, "step": 1, "help": "输出视频帧数"})
    if has_audio_condition:
        controls.append({"name": "audio_tokens", "label": "音频 token", "type": "number", "value": audio_tokens, "min": 1, "max": 4096, "step": 1, "help": "预计算音频条件长度"})

    headline = (
        "Video diffusion pipeline，文本条件与时空 latent 共同驱动 transformer 去噪，再通过 VAE 解码为视频。"
        if is_video
        else "Diffusion pipeline，文本条件与 latent 空间共同驱动 transformer 去噪，再通过 VAE 解码为图像。"
    )
    return base_model_payload(
        model_id,
        "diffusers",
        architecture,
        headline,
        summary,
        controls,
        {
            "batch": batch,
            "seq_len": prompt_tokens,
            "image_height": image_height,
            "image_width": image_width,
            "frames": frames,
            "steps": steps,
            "audio_tokens": audio_tokens if has_audio_condition else 0,
        },
        build_graph(lanes, nodes, edges),
        warnings,
        sources,
        "transformer",
    )


def build_fallback_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    _ = query
    architecture = infer_architecture_name(model_dir, "unknown")
    config = primary_config(model_dir)
    summary = [
        detail("类型", "未知"),
        detail("architecture", architecture),
        detail("顶层字段", sorted(config.keys())[:8]),
    ]
    lanes = [("raw", "原始配置")]
    nodes = [
        build_node(
            "raw_config",
            "raw",
            0,
            "原始配置摘要",
            "未适配的模型格式",
            "当前版本尚未针对该模型格式做专门图结构提取。",
            "-",
            "-",
            ["fallback"],
            [detail("available_keys", sorted(config.keys()))],
            [section("建议", [detail("next step", "可为该模型家族补充专用解析器")])],
            "output",
        )
    ]
    sources: list[str] = []
    append_source_file(model_dir, sources, "config.json")
    append_source_file(model_dir, sources, "configuration.json")
    append_source_file(model_dir, sources, "params.json")
    return base_model_payload(
        model_id,
        "unknown",
        architecture,
        "当前模型格式未完全归类，页面展示原始配置摘要。",
        summary,
        [],
        {},
        build_graph(lanes, nodes, []),
        ["当前模型格式没有专用图提取器，因此仅展示摘要信息。"],
        sources,
        "raw_config",
    )


def build_model_payload(model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    model_dir = resolve_model_dir(model_id)
    model_type = classify_model_dir(model_dir)
    if is_hy_world_bundle(model_dir):
        return build_hy_world_payload(model_dir, model_id, query)
    if is_nemotron_streaming_asr(model_dir):
        return build_nemotron_streaming_asr_payload(model_dir, model_id, query)
    if model_type == "llm":
        return build_llm_payload(model_dir, model_id, query)
    if model_type == "multimodal":
        config = primary_config(model_dir)
        yaml_config = supplemental_yaml_config(model_dir)
        if is_asr_model_config(config, yaml_config):
            return build_asr_payload(model_dir, model_id, query)
        if is_sam_video_config(config):
            return build_sam_video_payload(model_dir, model_id, query)
        if is_tts_model_config(config):
            return build_tts_payload(model_dir, model_id, query)
        return build_multimodal_payload(model_dir, model_id, query)
    if model_type == "diffusers":
        return build_diffusers_payload(model_dir, model_id, query)
    return build_fallback_payload(model_dir, model_id, query)


class ModelArchRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        segments = [unquote(segment) for segment in parsed.path.split("/") if segment]

        if segments[:2] == ["api", "models"]:
            if len(segments) == 2:
                self.respond_json({"models": build_model_catalog()})
                return

            if len(segments) == 3:
                try:
                    payload = build_model_payload(segments[2], parse_qs(parsed.query))
                except FileNotFoundError:
                    self.respond_json({"error": "Model not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self.respond_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self.respond_json(payload)
                return

            self.respond_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        if parsed.path == "/":
            self.path = "/index.html"

        return super().do_GET()

    def respond_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local model architecture viewer")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ModelArchRequestHandler)
    print(f"[INFO] Serving model architecture viewer at http://{args.host}:{args.port}")
    print(f"[INFO] Reading model configs from {MODEL_CONFIGS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

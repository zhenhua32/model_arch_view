#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Serve a local model architecture viewer for model_configs."""

from __future__ import annotations

import argparse
import json
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


def primary_config(model_dir: Path) -> dict[str, Any]:
    for name in ("config.json", "configuration.json"):
        path = model_dir / name
        if path.exists():
            data = read_json_file(path)
            if data:
                return data
    return {}


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


def classify_model_dir(model_dir: Path) -> str:
    if (model_dir / "model_index.json").exists():
        return "diffusers"

    config = primary_config(model_dir)
    params = read_json_file(model_dir / "params.json")
    processor = read_json_file(model_dir / "processor_config.json")

    if config.get("vision_config") or config.get("text_config") or config.get("language_config"):
        return "multimodal"

    if processor or params.get("vision_encoder"):
        return "multimodal"

    if config or params:
        return "llm"

    return "unknown"


def infer_architecture_name(model_dir: Path, model_type: str) -> str:
    if model_type == "diffusers":
        model_index = read_json_file(model_dir / "model_index.json")
        return str(model_index.get("_class_name") or "DiffusersPipeline")

    config = primary_config(model_dir)
    if config.get("architectures"):
        return str(config["architectures"][0])

    params = read_json_file(model_dir / "params.json")
    if params.get("vision_encoder"):
        return "VisionLanguageModel"
    if params:
        return "TransformerModel"

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
    linear_attention = sum(1 for item in layer_types if "linear" in str(item))
    return f"linear={linear_attention}, full={full_attention}"


def build_llm_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    params_config = read_json_file(model_dir / "params.json")
    architecture = infer_architecture_name(model_dir, "llm")

    hidden_size = int(first_defined(config.get("hidden_size"), params_config.get("dim"), 0) or 0)
    num_layers = int(first_defined(config.get("num_hidden_layers"), params_config.get("n_layers"), 0) or 0)
    num_heads = int(first_defined(config.get("num_attention_heads"), params_config.get("n_heads"), 0) or 0)
    num_kv_heads = int(first_defined(config.get("num_key_value_heads"), params_config.get("n_kv_heads"), num_heads) or num_heads or 0)
    head_dim = int(first_defined(config.get("head_dim"), params_config.get("head_dim"), hidden_size // num_heads if hidden_size and num_heads else 0) or 0)
    ffn_hidden = int(first_defined(config.get("intermediate_size"), params_config.get("hidden_dim"), config.get("moe_intermediate_size"), 0) or 0)
    vocab_size = int(first_defined(config.get("vocab_size"), params_config.get("vocab_size"), 0) or 0)
    max_position = int(first_defined(config.get("max_position_embeddings"), params_config.get("max_position_embeddings"), 4096) or 4096)
    sliding_window = int(first_defined(config.get("sliding_window"), config.get("sliding_window_size"), params_config.get("sliding_window_size"), 0) or 0)

    moe_config = params_config.get("moe") if isinstance(params_config.get("moe"), dict) else {}
    num_experts = int(first_defined(config.get("n_routed_experts"), config.get("num_experts"), moe_config.get("num_experts"), 0) or 0)
    experts_per_tok = int(first_defined(config.get("num_experts_per_tok"), moe_config.get("num_experts_per_tok"), 0) or 0)
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

    batch = clamp_int(query.get("batch", [1])[0], 1)
    seq_len = clamp_int(query.get("seq_len", [min(max_position, 2048)])[0], min(max_position, 2048), maximum=max_position)

    hidden_shape = shape(batch, seq_len, hidden_size or "hidden")
    logits_shape = shape(batch, seq_len, vocab_size or "vocab")
    attention_shape = shape(batch, seq_len, num_heads or "heads", head_dim or "head_dim")
    kv_shape = shape(batch, seq_len, num_kv_heads or "kv_heads", head_dim or "head_dim")
    score_shape = shape(batch, num_heads or "heads", seq_len, seq_len)
    rope_q_shape = f"Q_rope {attention_shape}; K_rope {kv_shape}"

    warnings = [
        "Shape 基于配置文件和当前输入参数推导，不是逐算子运行时真实张量。",
    ]
    if not hidden_size or not num_layers:
        warnings.append("该模型的层数或隐藏维度信息不完整，图中会保留摘要级展示。")

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
    if num_experts:
        summary.append(detail("MoE", f"{num_experts} experts / top-{experts_per_tok}"))

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 16, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "Token 长度", "type": "number", "value": seq_len, "min": 1, "max": max_position, "step": 1, "help": "输入 token 数"},
    ]

    headline = f"Decoder-only 语言模型，隐藏维度 {hidden_size or '?'}，共 {num_layers or '?'} 层。"
    return base_model_payload(
        model_id,
        "llm",
        architecture,
        headline,
        summary,
        controls,
        {"batch": batch, "seq_len": seq_len},
        build_graph(lanes, nodes, edges),
        warnings,
        sources,
        "decoder_stack",
    )


def infer_default_image_size(config: dict[str, Any], vision_config: dict[str, Any], image_processor: dict[str, Any]) -> tuple[int, int]:
    candidate_resolutions = config.get("candidate_resolutions")
    if isinstance(candidate_resolutions, list) and candidate_resolutions:
        pair = candidate_resolutions[0]
        if isinstance(pair, list) and len(pair) >= 2:
            return clamp_int(pair[0], 1024), clamp_int(pair[1], 1024)

    image_size = vision_config.get("image_size")
    if isinstance(image_size, (int, float)):
        size = clamp_int(image_size, 1024)
        return size, size

    size_config = image_processor.get("size") if isinstance(image_processor, dict) else None
    if isinstance(size_config, dict):
        shortest = first_defined(size_config.get("shortest_edge"), size_config.get("height"), size_config.get("width"))
        if isinstance(shortest, (int, float)):
            size = clamp_int(shortest, 1024)
            return size, size

    return 1024, 1024


def build_multimodal_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    config = primary_config(model_dir)
    params_config = read_json_file(model_dir / "params.json")
    processor_config = read_json_file(model_dir / "processor_config.json")
    video_processor_config = read_json_file(model_dir / "video_preprocessor_config.json")
    architecture = infer_architecture_name(model_dir, "multimodal")

    text_config = first_defined(config.get("text_config"), config.get("language_config"), config, params_config) or {}
    vision_config = first_defined(config.get("vision_config"), params_config.get("vision_encoder"), {}) or {}
    projector_config = config.get("projector_config") if isinstance(config.get("projector_config"), dict) else {}
    image_processor = processor_config.get("image_processor") if isinstance(processor_config.get("image_processor"), dict) else processor_config
    video_processor = processor_config.get("video_processor") if isinstance(processor_config.get("video_processor"), dict) else video_processor_config

    language_hidden = int(first_defined(text_config.get("hidden_size"), config.get("hidden_size"), params_config.get("dim"), projector_config.get("n_embed"), 2048) or 2048)
    num_layers = int(first_defined(text_config.get("num_hidden_layers"), config.get("num_hidden_layers"), params_config.get("n_layers"), 0) or 0)
    num_heads = int(first_defined(text_config.get("num_attention_heads"), config.get("num_attention_heads"), params_config.get("n_heads"), 0) or 0)
    head_dim = int(first_defined(text_config.get("head_dim"), params_config.get("head_dim"), language_hidden // num_heads if language_hidden and num_heads else 0) or 0)
    vocab_size = int(first_defined(text_config.get("vocab_size"), config.get("vocab_size"), params_config.get("vocab_size"), 0) or 0)
    max_position = int(first_defined(text_config.get("max_position_embeddings"), config.get("max_position_embeddings"), params_config.get("max_position_embeddings"), 8192) or 8192)

    patch_size = int(first_defined(image_processor.get("patch_size"), vision_config.get("patch_size"), 16) or 16)
    merge_size = int(first_defined(image_processor.get("merge_size"), vision_config.get("spatial_merge_size"), processor_config.get("downsample_ratio"), 1) or 1)
    temporal_patch = int(first_defined(video_processor.get("temporal_patch_size"), image_processor.get("temporal_patch_size"), vision_config.get("temporal_patch_size"), 1) or 1)
    vision_hidden = int(
        first_defined(
            vision_config.get("out_hidden_size"),
            projector_config.get("input_dim"),
            vision_config.get("hidden_size"),
            max_nested_numeric(vision_config.get("width"), "width") if isinstance(vision_config.get("width"), dict) else None,
            language_hidden,
        )
        or language_hidden
    )
    projector_hidden = int(first_defined(projector_config.get("n_embed"), language_hidden, vision_config.get("out_hidden_size"), language_hidden) or language_hidden)
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
    has_video = bool(config.get("video_token_id") or video_processor or video_processor_config or vision_config.get("temporal_patch_size"))

    batch = clamp_int(query.get("batch", [1])[0], 1)
    seq_len = clamp_int(query.get("seq_len", [min(max_position, 1024)])[0], min(max_position, 1024), maximum=max_position)
    image_height = clamp_int(query.get("image_height", [default_height])[0], default_height)
    image_width = clamp_int(query.get("image_width", [default_width])[0], default_width)
    frames = clamp_int(query.get("frames", [max(temporal_patch, 8)])[0], max(temporal_patch, 8))

    image_patch_count = ceil_div(image_height, patch_size) * ceil_div(image_width, patch_size)
    image_token_count = ceil_div(image_patch_count, max(merge_size, 1) ** 2)
    video_frame_groups = ceil_div(frames, temporal_patch)
    video_patch_count = video_frame_groups * ceil_div(image_height, patch_size) * ceil_div(image_width, patch_size)
    video_token_count = ceil_div(video_patch_count, max(merge_size, 1) ** 2)
    total_tokens = seq_len + image_token_count + (video_token_count if has_video else 0)
    image_patch_width = 3 * patch_size * patch_size
    video_patch_width = image_patch_width * temporal_patch

    warnings = [
        "视觉 token 数根据 patch_size、merge_size 和当前分辨率估算，实际实现可能包含额外裁剪或动态采样。"
    ]
    layer_pattern = summarize_layer_pattern(text_config.get("layer_types"))
    if layer_pattern:
        warnings.append(f"文本主干层型摘要: {layer_pattern}。")

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
            [detail("raw_patch_count", image_patch_count), detail("patch_feature_width", image_patch_width), detail("merged_token_count", image_token_count)],
            [
                section("预处理参数", [detail("merge_size", merge_size), detail("patch_size", patch_size)]),
                section(
                    "推导公式",
                    [
                        detail("raw patches", f"ceil({image_height}/{patch_size}) * ceil({image_width}/{patch_size}) = {image_patch_count}"),
                        detail("patch width", f"3 * {patch_size} * {patch_size} = {image_patch_width}"),
                        detail("merged tokens", f"ceil({image_patch_count} / {max(merge_size, 1) ** 2}) = {image_token_count}"),
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
            shape(batch, image_token_count, vision_hidden),
            [f"vision {vision_hidden}", f"tokens {image_token_count}"],
            [detail("vision_hidden", vision_hidden), detail("vision_depth", vision_depth), detail("merged_tokens", image_token_count)],
            [section("视觉流", [detail("encoder output", shape(batch, image_token_count, vision_hidden))])],
            "vision",
        ),
        build_node(
            "image_projector",
            "fusion",
            1,
            "多模态投影",
            "vision -> language hidden",
            "把视觉特征映射到语言主干维度。",
            shape(batch, image_token_count, vision_hidden),
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
            [detail("text_tokens", seq_len), detail("image_tokens", image_token_count), detail("video_tokens", video_token_count if has_video else 0), detail("total_tokens", total_tokens)],
            [
                section("融合后序列", [detail("combined hidden", shape(batch, total_tokens, language_hidden))]),
                section(
                    "推导公式",
                    [
                        detail("token budget", f"text {seq_len} + image {image_token_count} + video {video_token_count if has_video else 0} = {total_tokens}"),
                        detail("fusion output", f"[B, total_tokens, H] = {shape(batch, total_tokens, language_hidden)}"),
                    ],
                ),
            ],
            "fusion",
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
                    [detail("frame_groups", video_frame_groups), detail("raw_patch_count", video_patch_count), detail("merged_token_count", video_token_count)],
                    [
                        section("时空 patch", [detail("patch width", video_patch_width), detail("temporal_patch", temporal_patch)]),
                        section(
                            "推导公式",
                            [
                                detail("frame groups", f"ceil({frames}/{temporal_patch}) = {video_frame_groups}"),
                                detail("raw patches", f"{video_frame_groups} * ceil({image_height}/{patch_size}) * ceil({image_width}/{patch_size}) = {video_patch_count}"),
                                detail("merged tokens", f"ceil({video_patch_count} / {max(merge_size, 1) ** 2}) = {video_token_count}"),
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
                    shape(batch, video_token_count, vision_hidden),
                    [f"vision {vision_hidden}", f"tokens {video_token_count}"],
                    [detail("vision_hidden", vision_hidden), detail("video_tokens", video_token_count)],
                    [section("输出张量", [detail("video encoder output", shape(batch, video_token_count, vision_hidden))])],
                    "vision",
                ),
                build_node(
                    "video_projector",
                    "fusion",
                    0,
                    "视频投影",
                    "video -> language hidden",
                    "将视频特征映射到语言主干维度。",
                    shape(batch, video_token_count, vision_hidden),
                    shape(batch, video_token_count, projector_hidden),
                    [f"to {projector_hidden}", "projector"],
                    [detail("projector_out", projector_hidden), detail("video_tokens", video_token_count)],
                    [section("投影结果", [detail("projected video tokens", shape(batch, video_token_count, projector_hidden))])],
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

    edges = [
        {"source": "text_input", "target": "text_embedding", "label": "token ids"},
        {"source": "text_embedding", "target": "fusion_context", "label": "text hidden"},
        {"source": "image_input", "target": "image_processor", "label": "image tensor"},
        {"source": "image_processor", "target": "image_encoder", "label": "patches"},
        {"source": "image_encoder", "target": "image_projector", "label": "vision hidden"},
        {"source": "image_projector", "target": "fusion_context", "label": "image tokens"},
        {"source": "fusion_context", "target": "language_backbone", "label": "conditioned sequence"},
        {"source": "language_backbone", "target": "lm_head", "label": "hidden stream"},
        {"source": "lm_head", "target": "logits", "label": "vocab projection"},
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

    sources: list[str] = []
    append_source_file(model_dir, sources, "config.json")
    append_source_file(model_dir, sources, "configuration.json")
    append_source_file(model_dir, sources, "processor_config.json")
    append_source_file(model_dir, sources, "video_preprocessor_config.json")
    append_source_file(model_dir, sources, "params.json")

    summary = [
        detail("类型", "多模态"),
        detail("语言隐藏维度", language_hidden),
        detail("视觉隐藏维度", vision_hidden),
        detail("主干层数", num_layers),
        detail("patch size", patch_size),
        detail("当前总 tokens", total_tokens),
    ]
    if has_video:
        summary.append(detail("视频", f"temporal patch {temporal_patch}"))

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 8, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "文本 token", "type": "number", "value": seq_len, "min": 1, "max": max_position, "step": 1, "help": "文本序列长度"},
        {"name": "image_height", "label": "图像高", "type": "number", "value": image_height, "min": patch_size, "max": 8192, "step": patch_size, "help": "输入图像高度"},
        {"name": "image_width", "label": "图像宽", "type": "number", "value": image_width, "min": patch_size, "max": 8192, "step": patch_size, "help": "输入图像宽度"},
    ]
    if has_video:
        controls.append({"name": "frames", "label": "视频帧数", "type": "number", "value": frames, "min": temporal_patch, "max": 1024, "step": temporal_patch, "help": "采样到模型的帧数"})

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
            "seq_len": seq_len,
            "image_height": image_height,
            "image_width": image_width,
            "frames": frames if has_video else 0,
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
            2048,
        )
        or 2048
    )


def infer_text_layers(text_component_config: dict[str, Any], transformer_config: dict[str, Any]) -> int:
    nested = text_component_config.get("text_config") if isinstance(text_component_config.get("text_config"), dict) else {}
    return int(first_defined(nested.get("num_hidden_layers"), text_component_config.get("num_hidden_layers"), transformer_config.get("num_text_layers"), 0) or 0)


def infer_transformer_width(transformer_config: dict[str, Any]) -> int:
    hidden_size = transformer_config.get("hidden_size")
    if isinstance(hidden_size, (int, float)):
        return int(hidden_size)

    num_heads = first_defined(transformer_config.get("num_attention_heads"), transformer_config.get("text_num_attention_heads"))
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


def build_diffusers_payload(model_dir: Path, model_id: str, query: dict[str, list[str]]) -> dict[str, Any]:
    model_index = read_json_file(model_dir / "model_index.json")
    architecture = infer_architecture_name(model_dir, "diffusers")
    component_keys = list(model_index.keys())

    text_component_name = "text_encoder" if "text_encoder" in component_keys else "mllm" if "mllm" in component_keys else None
    processor_component_name = "processor" if "processor" in component_keys else "tokenizer" if "tokenizer" in component_keys else None
    scheduler_component_name = "scheduler" if "scheduler" in component_keys else None
    transformer_component_name = "transformer" if "transformer" in component_keys else None
    vae_component_name = "vae" if "vae" in component_keys else None

    text_component_config, text_source = load_component_config(model_dir, text_component_name)
    processor_component_config, processor_source = load_component_config(model_dir, processor_component_name)
    scheduler_config, scheduler_source = load_component_config(model_dir, scheduler_component_name)
    transformer_config, transformer_source = load_component_config(model_dir, transformer_component_name)
    vae_config, vae_source = load_component_config(model_dir, vae_component_name)

    text_hidden = infer_text_hidden(text_component_config, transformer_config)
    text_layers = infer_text_layers(text_component_config, transformer_config)
    text_nested = text_component_config.get("text_config") if isinstance(text_component_config.get("text_config"), dict) else {}
    max_position = int(first_defined(text_nested.get("max_position_embeddings"), text_component_config.get("max_position_embeddings"), 4096) or 4096)
    prompt_tokens = clamp_int(query.get("seq_len", [min(max_position, 256)])[0], min(max_position, 256), maximum=max_position)
    batch = clamp_int(query.get("batch", [1])[0], 1)

    default_image = int(first_defined(vae_config.get("sample_size"), 1024) or 1024)
    image_height = clamp_int(query.get("image_height", [default_image])[0], default_image)
    image_width = clamp_int(query.get("image_width", [default_image])[0], default_image)
    steps = clamp_int(query.get("steps", [28 if model_index.get("is_distilled") else 40])[0], 28 if model_index.get("is_distilled") else 40)

    vae_scale = derive_vae_scale(vae_config)
    latent_channels = int(first_defined(vae_config.get("latent_channels"), vae_config.get("z_dim"), transformer_config.get("in_channels"), 4) or 4)
    latent_height = ceil_div(image_height, vae_scale)
    latent_width = ceil_div(image_width, vae_scale)
    transformer_patch = int(first_defined(transformer_config.get("patch_size"), model_index.get("patch_size"), 1) or 1)
    latent_tokens = ceil_div(latent_height, transformer_patch) * ceil_div(latent_width, transformer_patch)
    transformer_width = infer_transformer_width(transformer_config)
    transformer_layers = int(first_defined(transformer_config.get("num_layers"), transformer_config.get("num_double_stream_layers"), 0) or 0)
    transformer_heads = int(first_defined(transformer_config.get("num_attention_heads"), 0) or 0)
    transformer_head_dim = int(first_defined(transformer_config.get("attention_head_dim"), transformer_config.get("head_dim"), 0) or 0)

    has_conditioning_image = bool(processor_component_name == "processor" or text_component_name == "mllm")
    conditioning_patch = int(first_defined(processor_component_config.get("patch_size"), text_component_config.get("vision_config", {}).get("patch_size") if isinstance(text_component_config.get("vision_config"), dict) else None, 16) or 16)
    conditioning_merge = int(first_defined(processor_component_config.get("merge_size"), text_component_config.get("vision_config", {}).get("spatial_merge_size") if isinstance(text_component_config.get("vision_config"), dict) else None, 1) or 1)
    conditioning_raw_patches = ceil_div(image_height, conditioning_patch) * ceil_div(image_width, conditioning_patch)
    conditioning_tokens = ceil_div(conditioning_raw_patches, max(conditioning_merge, 1) ** 2)
    scheduler_seq_len = int(first_defined(scheduler_config.get("base_image_seq_len"), scheduler_config.get("seq_len"), latent_tokens) or latent_tokens)

    warnings = [
        "Diffusers 图中的 latent token 数由 VAE 下采样倍率和 transformer patch_size 估算。",
        "Scheduler 节点展示的是步数和序列长度摘要，不是单个运行时张量。",
    ]
    if has_conditioning_image:
        warnings.append("该 pipeline 含图像条件分支，是否必须输入图像取决于具体调用方式。")

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

    latents_input_node = "latent_init"
    latent_input_label = "Latent 初始化"
    latent_input_description = "以高斯噪声 latent 开始迭代去噪。"
    latent_input_badges = [f"scale 1/{vae_scale}", f"channels {latent_channels}"]
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
                f"latent {latent_channels} x {latent_height} x {latent_width}",
                latent_input_description,
                shape(batch, 3, image_height, image_width) if has_conditioning_image else shape(batch, latent_channels, latent_height, latent_width),
                shape(batch, latent_channels, latent_height, latent_width),
                latent_input_badges,
                [detail("vae_scale", vae_scale), detail("latent_height", latent_height), detail("latent_width", latent_width), detail("latent_channels", latent_channels)],
                [
                    section("latent 形状", [detail("latent tensor", shape(batch, latent_channels, latent_height, latent_width)), detail("latent tokens", latent_tokens)]),
                    section(
                        "推导公式",
                        [
                            detail("latent height", f"ceil({image_height}/{vae_scale}) = {latent_height}"),
                            detail("latent width", f"ceil({image_width}/{vae_scale}) = {latent_width}"),
                            detail("latent tokens", f"ceil({latent_height}/{transformer_patch}) * ceil({latent_width}/{transformer_patch}) = {latent_tokens}"),
                        ],
                    ),
                ],
                "latent",
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
                            detail("latent grid", shape(batch, latent_channels, latent_height, latent_width)),
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
                "latent -> image",
                "把最终 latent 还原到像素空间。",
                shape(batch, latent_channels, latent_height, latent_width),
                shape(batch, 3, image_height, image_width),
                [f"scale x{vae_scale}", vae_config.get("_class_name", "vae")],
                [detail("latent_channels", latent_channels), detail("vae_scale", vae_scale), detail("output_size", shape(batch, 3, image_height, image_width))],
                [
                    section("解码结果", [detail("image tensor", shape(batch, 3, image_height, image_width))]),
                    section("推导公式", [detail("decode", f"[B, C, h, w] -> [B, 3, H, W] = {shape(batch, 3, image_height, image_width)}")]),
                ],
                "decode",
            ),
            build_node(
                "image_output",
                "output",
                0,
                "图像输出",
                "像素空间结果",
                "生成或编辑后的图像。",
                shape(batch, 3, image_height, image_width),
                shape(batch, 3, image_height, image_width),
                ["output"],
                [detail("image", shape(batch, 3, image_height, image_width))],
                [section("说明", [detail("postprocess", "可进一步保存、显示或后处理")])],
                "output",
            ),
        ]
    )

    edges = [
        {"source": "prompt_input", "target": "text_condition", "label": "prompt"},
        {"source": "text_condition", "target": "transformer", "label": "text hidden"},
        {"source": "scheduler", "target": "transformer", "label": "timesteps"},
        {"source": latents_input_node, "target": "transformer", "label": "latent tokens"},
        {"source": "transformer", "target": "vae_decode", "label": "denoised latent"},
        {"source": "vae_decode", "target": "image_output", "label": "pixels"},
    ]

    if has_conditioning_image:
        edges.extend(
            [
                {"source": "image_condition_input", "target": "image_condition_processor", "label": "conditioning image"},
                {"source": "image_condition_processor", "target": "vae_encode", "label": "prepared image"},
            ]
        )

    sources: list[str] = []
    append_source_file(model_dir, sources, "model_index.json")
    if text_source:
        append_source_file(model_dir, sources, text_source)
    if processor_source:
        append_source_file(model_dir, sources, processor_source)
    if scheduler_source:
        append_source_file(model_dir, sources, scheduler_source)
    if transformer_source:
        append_source_file(model_dir, sources, transformer_source)
    if vae_source:
        append_source_file(model_dir, sources, vae_source)

    summary = [
        detail("类型", "Diffusers"),
        detail("Transformer 宽度", transformer_width),
        detail("去噪层数", transformer_layers),
        detail("latent channels", latent_channels),
        detail("latent tokens", latent_tokens),
        detail("scheduler", scheduler_config.get("_class_name", "-")),
    ]

    controls = [
        {"name": "batch", "label": "Batch", "type": "number", "value": batch, "min": 1, "max": 8, "step": 1, "help": "并行样本数"},
        {"name": "seq_len", "label": "Prompt token", "type": "number", "value": prompt_tokens, "min": 1, "max": max_position, "step": 1, "help": "文本条件 token 数"},
        {"name": "image_height", "label": "图像高", "type": "number", "value": image_height, "min": 64, "max": 4096, "step": 8, "help": "输出图像高度"},
        {"name": "image_width", "label": "图像宽", "type": "number", "value": image_width, "min": 64, "max": 4096, "step": 8, "help": "输出图像宽度"},
        {"name": "steps", "label": "推理步数", "type": "number", "value": steps, "min": 1, "max": 200, "step": 1, "help": "scheduler 迭代步数"},
    ]

    headline = "Diffusion pipeline，文本条件与 latent 空间共同驱动 transformer 去噪，再通过 VAE 解码为图像。"
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
            "steps": steps,
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
    if model_type == "llm":
        return build_llm_payload(model_dir, model_id, query)
    if model_type == "multimodal":
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
                self.respond_json(payload)
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
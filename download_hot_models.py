#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 ModelScope 下载热门模型的配置文件（不含权重）。

功能：
  1. 通过 ModelScope 官方 dolphin 搜索 API 获取热门模型列表（综合排序）。
  2. 对每个模型，使用 snapshot_download + allow_patterns 只下载配置类文件，
     自动跳过权重文件（.bin / .safetensors / .gguf / .pt 等）。
  3. 支持按任务类型过滤、自定义下载数量、是否包含模型代码(.py)等。
  4. 单个模型失败不影响其他模型，最终输出汇总报告。

依赖：
  pip install modelscope requests

用法示例：
  # 下载前 10 个热门模型的配置文件到 ./configs
  python download_hot_models.py

  # 下载前 20 个，指定输出目录
  python download_hot_models.py --limit 20 --output-dir ./model_configs

  # 只要文本生成类模型，并包含 .py 代码
  python download_hot_models.py --task text-generation --with-code

  # 指定 modelscope token（下载需要鉴权的模型时）
  python download_hot_models.py --token ms-xxxxxxxx
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import requests

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:  # 兼容旧版导入路径
    from modelscope import snapshot_download


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/dolphin/models"

# 默认只下载的配置类文件模式（白名单）
DEFAULT_ALLOW_PATTERNS = [
    "*.json",        # config.json / generation_config.json / tokenizer_config.json / vocab.json ...
    "*.yaml",        # yaml 配置
    "*.yml",
    "*.txt",         # tokenizer 词表 / special_tokens_map 等
    "*.md",          # README 等说明文档
    "*.model",       # sentencepiece 模型配置
    "*.cfg",
    "*.tiktoken",    # tiktoken BPE 词表
    "tokenizer*",    # 通配 tokenizer 相关文件
    "vocab*",        # 词表
    "merges*",       # BPE merges
    "special_tokens*",
    "added_tokens*",
    "generation_config*",
    "chat_template*",  # 对话模板(jinja/json)
]

# 模型代码文件（默认不下载，按需开启）
CODE_PATTERNS = ["*.py"]

# 权重文件模式（仅作参考展示，本脚本用白名单方式天然排除）
WEIGHT_PATTERNS = [
    "*.bin", "*.safetensors", "*.gguf", "*.pt", "*.pth", "*.onnx",
    "*.ckpt", "*.h5", "*.msgpack", "*.npz", "*.npy", "*.zip",
    "*.tar", "*.gz", "*.tflite", "*.pb", "*.caffemodel", "*.mar",
    "*.lora", "*.aram", "*.gguf.*",
]

# modelscope 支持的任务类型（映射到 API 的 Task 字段）
TASK_MAP = {
    "text-generation": "text-generation",
    "text-to-image": "text-to-image-synthesis",
    "image-to-image": "image-to-image",
    "image-classification": "image-classification",
    "speech-recognition": "speech-recognition",
    "text-to-speech": "text-to-speech",
    "ocr": "ocr-recognition",
    "feature-extraction": "feature-extraction",
}


# ---------------------------------------------------------------------------
# 获取热门模型列表
# ---------------------------------------------------------------------------

def get_hot_models(
    limit: int = 10,
    task: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 30,
) -> List[Dict]:
    """通过 ModelScope dolphin API 获取热门模型列表。

    Args:
        limit: 获取模型数量
        task: 任务类型过滤（如 text-generation）
        token: modelscope access token（可选）
        timeout: 请求超时秒数

    Returns:
        模型信息列表，每项包含 Path/Name/Downloads/ChineseName 等字段
    """
    if limit <= 0:
        return []

    session = requests.Session()
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "User-Agent": "modelscope-config-downloader/1.0 (requests)",
        "x-modelscope-accept-language": "zh_CN",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)

    # 构造过滤条件
    criterion = []
    if task:
        api_task = TASK_MAP.get(task, task)
        criterion.append({"category": "tasks", "predicate": "contain", "value": [api_task]})

    page_size = min(limit, 50)  # API 单页上限 50
    payload = {
        "Name": "",
        "Criterion": criterion,
        "SingleCriterion": [],
        "SortBy": "Default",  # modelscope 综合排序（热门）
        "PageNumber": 1,
        "PageSize": page_size,
    }

    print(f"[INFO] 正在从 ModelScope 获取热门模型（SortBy=Default, limit={limit}）...")
    resp = session.put(MODELSCOPE_API, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("Success"):
        raise RuntimeError(f"ModelScope API 返回失败: {data.get('Message', data)}")

    model_data = data.get("Data", {}).get("Model", {})
    models = model_data.get("Models", [])
    total = model_data.get("TotalCount", "?")

    # 如果 limit 大于单页，继续翻页
    all_models = list(models)
    page = 2
    while len(all_models) < limit and len(models) > 0:
        payload["PageNumber"] = page
        payload["PageSize"] = min(limit - len(all_models), 50)
        resp = session.put(MODELSCOPE_API, json=payload, timeout=timeout)
        resp.raise_for_status()
        d = resp.json().get("Data", {}).get("Model", {})
        models = d.get("Models", [])
        if not models:
            break
        all_models.extend(models)
        page += 1

    all_models = all_models[:limit]
    print(f"[INFO] 模型库总数: {total}, 已获取 {len(all_models)} 个热门模型")
    return all_models


# ---------------------------------------------------------------------------
# 下载单个模型的配置文件
# ---------------------------------------------------------------------------

def download_model_config(
    model_id: str,
    local_dir: str,
    allow_patterns: List[str],
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict:
    """只下载模型的配置文件（不含权重）。

    Args:
        model_id: 模型 ID，如 "Qwen/Qwen2.5-7B-Instruct"
        local_dir: 本地下载目录
        allow_patterns: 允许下载的文件模式白名单
        token: modelscope access token
        revision: 模型版本（默认 master）

    Returns:
        下载结果字典 {model_id, status, path, files, error}
    """
    result = {"model_id": model_id, "status": "pending", "path": "", "files": [], "downloaded_files": [], "error": ""}
    t0 = time.time()
    try:
        before: Dict[str, tuple[int, int]] = {}
        if os.path.isdir(local_dir):
            for root, _dirs, files in os.walk(local_dir):
                for filename in files:
                    path = os.path.join(root, filename)
                    stat = os.stat(path)
                    before[os.path.relpath(path, local_dir)] = (stat.st_size, stat.st_mtime_ns)

        kwargs = dict(
            model_id=model_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            revision=revision or "master",
        )
        if token:
            kwargs["token"] = token

        model_dir = snapshot_download(**kwargs)
        # 收集实际下载的文件
        downloaded_files = []
        changed_files = []
        if model_dir and os.path.isdir(model_dir):
            for root, _dirs, files in os.walk(model_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, model_dir)
                    stat = os.stat(fp)
                    entry = {"file": rel, "size": stat.st_size}
                    downloaded_files.append(entry)
                    if before.get(rel) != (stat.st_size, stat.st_mtime_ns):
                        changed_files.append(entry)

        result["status"] = "success"
        result["path"] = model_dir or local_dir
        result["files"] = downloaded_files
        result["downloaded_files"] = changed_files
        result["elapsed"] = round(time.time() - t0, 1)
        result["total_size"] = sum(f["size"] for f in downloaded_files)
        result["downloaded_size"] = sum(f["size"] for f in changed_files)
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["elapsed"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def format_size(n: int) -> str:
    """把字节数格式化为人类可读字符串。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    parser = argparse.ArgumentParser(
        description="下载 ModelScope 热门模型的配置文件（不含权重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python download_hot_models.py --limit 10\n"
               "  python download_hot_models.py --task text-generation --with-code",
    )
    parser.add_argument("-n", "--limit", type=int, default=10,
                        help="获取的热门模型数量（默认 10）")
    parser.add_argument("-o", "--output-dir", default="./model_configs",
                        help="配置文件下载根目录（默认 ./model_configs）")
    parser.add_argument("-t", "--task", default=None,
                        choices=list(TASK_MAP.keys()),
                        help="按任务类型过滤（如 text-generation）")
    parser.add_argument("--with-code", action="store_true",
                        help="同时下载模型代码文件(*.py)，默认不下载")
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_API_TOKEN", ""),
                        help="ModelScope access token（也可设置环境变量 MODELSCOPE_API_TOKEN）")
    parser.add_argument("--list-only", action="store_true",
                        help="只列出热门模型，不下载")
    parser.add_argument("--models", nargs="*", default=None,
                        help="直接指定模型 ID 列表，跳过热门获取（如 Qwen/Qwen2.5-7B）")
    parser.add_argument("--revision", default="master",
                        help="模型版本/分支（默认 master）")
    args = parser.parse_args()

    token = args.token.strip() or None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 组装 allow_patterns ----
    allow_patterns = list(DEFAULT_ALLOW_PATTERNS)
    if args.with_code:
        allow_patterns += CODE_PATTERNS

    print("=" * 70)
    print("ModelScope 热门模型配置文件下载器")
    print("=" * 70)
    expected_count = len(args.models) if args.models else args.limit
    print(f"  输出目录   : {output_dir.resolve()}")
    print(f"  模型数量   : {expected_count}")
    print(f"  任务过滤   : {args.task or '无'}")
    print(f"  包含代码   : {'是' if args.with_code else '否'}")
    print(f"  允许的文件 : {allow_patterns}")
    print(f"  排除的权重 : {WEIGHT_PATTERNS} (通过白名单天然排除)")
    print("=" * 70)

    # ---- 获取模型列表 ----
    if args.models:
        models = [{"Path": m.split("/")[0], "Name": "/".join(m.split("/")[1:])}
                  for m in args.models]
        print(f"[INFO] 使用手动指定的 {len(models)} 个模型")
    else:
        models = get_hot_models(limit=args.limit, task=args.task, token=token)

    if not models:
        print("[WARN] 未获取到任何模型，退出。")
        return

    # ---- 打印模型列表 ----
    print(f"\n{'序号':>4}  {'下载量':>10}  {'收藏':>6}  模型ID")
    print("-" * 70)
    for i, m in enumerate(models, 1):
        path = m.get("Path", "")
        name = m.get("Name", "")
        model_id = f"{path}/{name}" if path else name
        dl = m.get("Downloads", 0) or 0
        star = m.get("Stars", 0) or 0
        cn = m.get("ChineseName", "")
        suffix = f"  ({cn})" if cn and cn != name else ""
        print(f"{i:>4}  {dl:>10,}  {star:>6}  {model_id}{suffix}")

    if args.list_only:
        print("\n[INFO] --list-only 模式，仅列出模型，不下载。")
        return

    # ---- 逐个下载配置文件 ----
    print("\n" + "=" * 70)
    print("开始下载配置文件...")
    print("=" * 70)

    results = []
    success_count = 0
    for i, m in enumerate(models, 1):
        path = m.get("Path", "")
        name = m.get("Name", "")
        model_id = f"{path}/{name}" if path else name
        dl = m.get("Downloads", 0) or 0

        # 每个模型一个子目录
        safe_name = model_id.replace("/", "__")
        model_local_dir = str(output_dir / safe_name)

        print(f"\n[{i}/{len(models)}] {model_id}  (下载量: {dl:,})")
        print(f"       -> {model_local_dir}")

        res = download_model_config(
            model_id=model_id,
            local_dir=model_local_dir,
            allow_patterns=allow_patterns,
            token=token,
            revision=args.revision,
        )
        results.append(res)

        if res["status"] == "success":
            success_count += 1
            n_files = len(res["files"])
            total_sz = format_size(res.get("total_size", 0))
            changed_count = len(res.get("downloaded_files", []))
            changed_size = format_size(res.get("downloaded_size", 0))
            print(f"       [成功] 目录共 {n_files} 个文件 / {total_sz}; 本次新增或更新 {changed_count} 个 / {changed_size}; 耗时 {res['elapsed']}s")
            for f in res["files"][:8]:
                print(f"          - {f['file']}  ({format_size(f['size'])})")
            if len(res["files"]) > 8:
                print(f"          ... 还有 {len(res['files']) - 8} 个文件")
        else:
            print(f"       [失败] {res['error']}")

    # ---- 汇总报告 ----
    print("\n" + "=" * 70)
    print("下载汇总")
    print("=" * 70)
    print(f"  总计模型 : {len(results)}")
    print(f"  成功     : {success_count}")
    print(f"  失败     : {len(results) - success_count}")

    if success_count > 0:
        total_files = sum(len(r["files"]) for r in results if r["status"] == "success")
        total_size = sum(r.get("total_size", 0) for r in results if r["status"] == "success")
        print(f"  总文件数 : {total_files}")
        print(f"  总大小   : {format_size(total_size)}")

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print("\n失败详情:")
        for r in failed:
            print(f"  - {r['model_id']}: {r['error']}")

    # ---- 保存结果 JSON ----
    report_path = output_dir / "download_report.json"
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir.resolve()),
        "allow_patterns": allow_patterns,
        "weight_patterns_excluded": WEIGHT_PATTERNS,
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "models": [],
    }
    for r in results:
        report["models"].append({
            "model_id": r["model_id"],
            "status": r["status"],
            "path": r.get("path", ""),
            "file_count": len(r.get("files", [])),
            "total_size": r.get("total_size", 0),
            "downloaded_file_count": len(r.get("downloaded_files", [])),
            "downloaded_size": r.get("downloaded_size", 0),
            "elapsed": r.get("elapsed", 0),
            "error": r.get("error", ""),
            "files": [{"file": f["file"], "size": f["size"]} for f in r.get("files", [])],
        })
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()

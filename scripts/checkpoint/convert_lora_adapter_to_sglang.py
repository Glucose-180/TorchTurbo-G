#!/usr/bin/env python
"""Convert a standard TorchTurbo HF LoRA adapter for the current SGLang path.

The canonical TorchTurbo export remains a standard HF/PEFT adapter.  This script
only creates an engine import variant: it filters targets not covered by the
current SGLang loader and performs the Qwen3.5 split-projection packing that
SGLang can execute at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."
LORA_SUFFIX = re.compile(r"\.(lora_A|lora_B)\.weight$")


@dataclass
class ConversionResult:
    profile: str
    source_tensor_count: int
    output_tensor_count: int
    retained: Counter[str]
    converted: Counter[str]
    dropped: Counter[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an SGLang-compatible variant of a TorchTurbo LoRA adapter."
    )
    parser.add_argument("--adapter-dir", required=True, help="Standard HF LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Converted adapter directory.")
    parser.add_argument(
        "--profile",
        choices=("auto", "qwen3_5", "bailing_v3"),
        default="auto",
        help="Model-specific conversion profile.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow targets unsupported by the current SGLang loader to be dropped. "
            "Required for current all-linear Qwen3.5/Bailing-V3 exports."
        ),
    )
    parser.add_argument("--report", default=None, help="Optional JSON conversion report.")
    return parser.parse_args(argv)


def _module_name(key: str) -> str:
    name = key.removeprefix(ADAPTER_PREFIX)
    return LORA_SUFFIX.sub("", name)


def _load_source(adapter_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(adapter_path, framework="pt") as source:
        for key in list(source.keys()):
            tensors[key] = source.get_tensor(key)
    return tensors


def _detect_profile(modules: set[str]) -> str:
    if any(".linear_attn.in_proj_qkv" in module for module in modules):
        return "qwen3_5"
    if any(re.search(r"\.mlp\.experts\.\d+\.", module) for module in modules) and any(
        ".attention." in module for module in modules
    ):
        return "bailing_v3"
    raise ValueError(
        "Could not infer the model profile; pass --profile qwen3_5 or bailing_v3."
    )


def _matrix(module: str, name: str) -> str:
    return f"{ADAPTER_PREFIX}{module}.{name}.weight"


def _convert_qwen3_5(
    tensors: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], ConversionResult]:
    modules = {_module_name(key) for key in tensors}
    output = {key: tensor for key, tensor in tensors.items()}
    retained: Counter[str] = Counter()
    converted: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    qkv_modules = sorted(
        module for module in modules if module.endswith(".linear_attn.in_proj_qkv")
    )
    for module in qkv_modules:
        prefix, _ = module.rsplit(".in_proj_qkv", 1)
        qkv_module = f"{prefix}.in_proj_qkv"
        z_module = f"{prefix}.in_proj_z"
        for matrix in ("lora_A", "lora_B"):
            qkv_key = _matrix(qkv_module, matrix)
            z_key = _matrix(z_module, matrix)
            if qkv_key not in tensors or z_key not in tensors:
                raise ValueError(f"Incomplete Qwen3.5 GDN LoRA pair: {qkv_key} / {z_key}")
            output[_matrix(f"{prefix}.in_proj_qkvz", matrix)] = torch.cat(
                (tensors[qkv_key], tensors[z_key]), dim=0
            ).contiguous()
            output.pop(qkv_key, None)
            output.pop(z_key, None)
            converted["in_proj_qkvz"] += 1

    for key in list(output):
        module = _module_name(key)
        leaf = module.rsplit(".", 1)[-1]
        if re.search(r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$", module):
            retained["routed_expert"] += 1
        elif module.endswith(".linear_attn.in_proj_qkvz"):
            retained["gdn_in_proj_qkvz"] += 1
        elif module.endswith(".linear_attn.out_proj"):
            retained["gdn_out_proj"] += 1
        elif module.endswith(
            (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
        ):
            retained["attention_qkv"] += 1
        elif module.endswith(".self_attn.o_proj"):
            retained["attention_o_proj"] += 1
        elif re.search(r"\.mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj)$", module):
            retained["shared_expert"] += 1
        elif re.search(r"\.mlp\.(?:gate_proj|up_proj|down_proj)$", module):
            retained["dense_mlp"] += 1
        elif ".linear_attn.in_proj_b" == module[-len(".linear_attn.in_proj_b") :]:
            dropped["gdn_in_proj_b"] += 1
            output.pop(key)
        elif ".linear_attn.in_proj_a" == module[-len(".linear_attn.in_proj_a") :]:
            dropped["gdn_in_proj_a"] += 1
            output.pop(key)
        elif module.endswith(".mlp.shared_expert_gate"):
            dropped["shared_expert_gate"] += 1
            output.pop(key)
        elif any(
            module.endswith(suffix)
            for suffix in (
                ".attn.proj",
                ".attn.qkv",
                ".mlp.linear_fc1",
                ".mlp.linear_fc2",
                ".merger.linear_fc1",
                ".merger.linear_fc2",
            )
        ):
            dropped["vision_or_connector"] += 1
            output.pop(key)
        else:
            dropped[f"unsupported:{leaf}"] += 1
            output.pop(key)

    return output, ConversionResult(
        "qwen3_5",
        len(tensors),
        len(output),
        retained,
        converted,
        dropped,
    )


def _convert_bailing_v3(
    tensors: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], ConversionResult]:
    output: dict[str, torch.Tensor] = {}
    retained: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    for key, tensor in tensors.items():
        module = _module_name(key)
        leaf = module.rsplit(".", 1)[-1]
        if re.search(r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$", module):
            output[key] = tensor
            retained["routed_expert"] += 1
        elif re.search(
            r"\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)$", module
        ):
            output[key] = tensor
            retained["shared_expert"] += 1
        elif re.search(r"\.mlp\.(?:gate_proj|up_proj|down_proj)$", module):
            output[key] = tensor
            retained["dense_mlp"] += 1
        else:
            dropped[f"unsupported:{leaf}"] += 1

    return output, ConversionResult(
        "bailing_v3",
        len(tensors),
        len(output),
        retained,
        Counter(),
        dropped,
    )


def _target_modules(profile: str, tensors: dict[str, torch.Tensor]) -> list[str]:
    modules = {_module_name(key) for key in tensors}
    targets: set[str] = set()
    if profile == "qwen3_5":
        if any(module.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")) for module in modules):
            targets.add("qkv_proj")
        if any(module.endswith(".self_attn.o_proj") for module in modules):
            targets.add("o_proj")
        if any(module.endswith(".linear_attn.out_proj") for module in modules):
            targets.add("out_proj")
        if any(module.endswith(".linear_attn.in_proj_qkvz") for module in modules):
            targets.add("in_proj_qkvz")
    if any(module.endswith((".gate_proj", ".up_proj")) for module in modules):
        targets.add("gate_up_proj")
    if any(module.endswith(".down_proj") for module in modules):
        targets.add("down_proj")
    return sorted(targets)


def _write_report(path: Path, result: ConversionResult) -> None:
    payload: dict[str, Any] = {
        "profile": result.profile,
        "source_tensor_count": result.source_tensor_count,
        "output_tensor_count": result.output_tensor_count,
        "dropped_tensor_count": sum(result.dropped.values()),
        "collapsed_tensor_count": (
            result.source_tensor_count
            - result.output_tensor_count
            - sum(result.dropped.values())
        ),
        "retained": dict(sorted(result.retained.items())),
        "converted": dict(sorted(result.converted.items())),
        "dropped": dict(sorted(result.dropped.items())),
        "partial": bool(result.dropped),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def convert_adapter(
    adapter_dir: Path, output_dir: Path, profile: str, *, allow_partial: bool
) -> ConversionResult:
    config_path = adapter_dir / "adapter_config.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    config = json.loads(config_path.read_text())
    tensors = _load_source(adapter_path)
    modules = {_module_name(key) for key in tensors}
    resolved_profile = _detect_profile(modules) if profile == "auto" else profile

    if resolved_profile == "qwen3_5":
        output, result = _convert_qwen3_5(tensors)
    else:
        output, result = _convert_bailing_v3(tensors)

    if result.dropped and not allow_partial:
        dropped = ", ".join(sorted(result.dropped))
        raise ValueError(
            "Conversion would drop tensors unsupported by the current SGLang path "
            f"({result.source_tensor_count - result.output_tensor_count} tensors: {dropped}). "
            "Pass --allow-partial after reviewing the semantic impact."
        )

    converted_config = dict(config)
    converted_config["target_modules"] = _target_modules(resolved_profile, output)
    if not converted_config["target_modules"]:
        raise ValueError("Conversion produced no SGLang-supported target modules.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        output,
        output_dir / "adapter_model.safetensors",
        metadata={"format": "pt"},
    )
    (output_dir / "adapter_config.json").write_text(
        json.dumps(converted_config, indent=2) + "\n"
    )
    added_tokens = adapter_dir / "added_tokens.json"
    if added_tokens.is_file():
        shutil.copy2(added_tokens, output_dir / added_tokens.name)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = convert_adapter(
        Path(args.adapter_dir),
        Path(args.output_dir),
        args.profile,
        allow_partial=args.allow_partial,
    )
    if args.report:
        _write_report(Path(args.report), result)
    print(
        f"Converted {result.source_tensor_count} -> {result.output_tensor_count} "
        f"tensors using profile {result.profile}."
    )


if __name__ == "__main__":
    main()

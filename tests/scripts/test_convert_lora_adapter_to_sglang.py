"""Tests for SGLang LoRA adapter conversion."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "checkpoint"
    / "convert_lora_adapter_to_sglang.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "convert_lora_adapter_to_sglang", _SCRIPT
)
converter = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = converter
_SPEC.loader.exec_module(converter)


def _write_adapter(
    root: Path,
    tensors: dict[str, torch.Tensor],
    target_modules: list[str],
) -> None:
    root.mkdir(parents=True)
    save_file(tensors, root / "adapter_model.safetensors")
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 2,
                "lora_alpha": 4,
                "use_rslora": False,
                "use_dora": False,
                "bias": "none",
                "target_modules": target_modules,
            }
        )
    )


def test_qwen3_5_conversion_packs_gdn_and_filters_unsupported(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    rank, hidden = 2, 6
    qkv_a = torch.randn(rank, hidden)
    z_a = torch.randn(rank, hidden)
    qkv_b = torch.randn(8, rank)
    z_b = torch.randn(4, rank)
    expert_a = torch.randn(rank, hidden)
    expert_b = torch.randn(5, rank)

    tensors = {
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": qkv_a,
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight": qkv_b,
        "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_A.weight": z_a,
        "base_model.model.model.layers.0.linear_attn.in_proj_z.lora_B.weight": z_b,
        "base_model.model.model.layers.0.linear_attn.in_proj_b.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.linear_attn.in_proj_b.lora_B.weight": torch.randn(3, rank),
        "base_model.model.model.layers.0.linear_attn.out_proj.lora_A.weight": torch.randn(rank, 12),
        "base_model.model.model.layers.0.linear_attn.out_proj.lora_B.weight": torch.randn(hidden, rank),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.mlp.experts.7.gate_proj.lora_A.weight": expert_a,
        "base_model.model.model.layers.0.mlp.experts.7.gate_proj.lora_B.weight": expert_b,
        "base_model.model.model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.model.layers.0.mlp.shared_expert_gate.lora_A.weight": torch.randn(rank, hidden),
        "base_model.model.visual.attn.qkv.lora_A.weight": torch.randn(rank, hidden),
    }
    _write_adapter(source, tensors, ["in_proj_qkv", "experts.7.gate_proj"])

    report = tmp_path / "report.json"
    converter.main(
        [
            "--adapter-dir",
            str(source),
            "--output-dir",
            str(output),
            "--profile",
            "qwen3_5",
            "--allow-partial",
            "--report",
            str(report),
        ]
    )

    converted = load_file(str(output / "adapter_model.safetensors"))
    qkvz_a = "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_A.weight"
    qkvz_b = "base_model.model.model.layers.0.linear_attn.in_proj_qkvz.lora_B.weight"
    assert torch.equal(converted[qkvz_a], torch.cat((qkv_a, z_a), dim=0))
    assert torch.equal(converted[qkvz_b], torch.cat((qkv_b, z_b), dim=0))
    assert torch.equal(
        converted["base_model.model.model.layers.0.mlp.experts.7.gate_proj.lora_A.weight"],
        expert_a,
    )
    assert not any("in_proj_b" in key for key in converted)
    assert not any("shared_expert_gate" in key for key in converted)
    assert not any("visual" in key for key in converted)

    config = json.loads((output / "adapter_config.json").read_text())
    assert config["target_modules"] == [
        "gate_up_proj",
        "in_proj_qkvz",
        "o_proj",
        "out_proj",
        "qkv_proj",
    ]
    result = json.loads(report.read_text())
    assert result["profile"] == "qwen3_5"
    assert result["partial"] is True


def test_bailing_v3_conversion_keeps_supported_mlp_targets(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    expert_a = torch.randn(2, 4)
    expert_b = torch.randn(3, 2)
    shared_a = torch.randn(2, 4)
    dense_a = torch.randn(2, 4)

    tensors = {
        "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_A.weight": expert_a,
        "base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_B.weight": expert_b,
        "base_model.model.model.layers.0.mlp.shared_experts.gate_proj.lora_A.weight": shared_a,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight": dense_a,
        "base_model.model.model.layers.0.attention.q_proj.lora_A.weight": torch.randn(2, 4),
        "base_model.model.model.layers.0.attention.b_proj.lora_A.weight": torch.randn(2, 4),
        "base_model.model.model.layers.0.attention.dense.lora_A.weight": torch.randn(2, 4),
    }
    _write_adapter(source, tensors, ["experts.3.gate_proj", "q_proj"])

    converter.main(
        [
            "--adapter-dir",
            str(source),
            "--output-dir",
            str(output),
            "--profile",
            "bailing_v3",
            "--allow-partial",
        ]
    )

    converted = load_file(str(output / "adapter_model.safetensors"))
    assert torch.equal(
        converted["base_model.model.model.layers.0.mlp.experts.3.gate_proj.lora_A.weight"],
        expert_a,
    )
    assert torch.equal(
        converted["base_model.model.model.layers.0.mlp.shared_experts.gate_proj.lora_A.weight"],
        shared_a,
    )
    assert torch.equal(
        converted["base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight"],
        dense_a,
    )
    assert not any(".attention." in key for key in converted)
    config = json.loads((output / "adapter_config.json").read_text())
    assert config["target_modules"] == ["gate_up_proj"]


def test_partial_conversion_requires_explicit_permission(tmp_path):
    source = tmp_path / "source"
    _write_adapter(
        source,
        {
            "base_model.model.model.layers.0.attention.q_proj.lora_A.weight": torch.randn(2, 4),
            "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.randn(2, 4),
        },
        ["q_proj", "experts.0.gate_proj"],
    )

    try:
        converter.convert_adapter(
            source,
            tmp_path / "output",
            "bailing_v3",
            allow_partial=False,
        )
    except ValueError as exc:
        assert "Pass --allow-partial" in str(exc)
    else:
        raise AssertionError("Expected a partial-conversion error")

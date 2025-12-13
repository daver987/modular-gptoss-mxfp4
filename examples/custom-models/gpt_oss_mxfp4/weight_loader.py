"""Weight adapters for GPT-OSS models."""

from __future__ import annotations

from collections import OrderedDict

from max.graph.weights import WeightData, Weights

# Ordered so that bias mappings happen before similarly-prefixed weights.
GPT_OSS_SAFETENSOR_MAP: OrderedDict[str, str] = OrderedDict([
    ("model.embed_tokens.", "language_model.embed_tokens."),
    ("model.norm.", "language_model.norm."),
    ("lm_head.", "language_model.lm_head."),
    ("model.layers.", "language_model.layers."),
    # MoE weight mappings
    (".mlp.router", ".mlp.gate.gate_score"),
    ("experts.gate_up_proj_bias", "_experts_gate_up_proj_bias"),
    ("experts.down_proj_bias", "_experts_down_proj_bias"),
    # The following weights must be listed after the bias weights.
    ("experts.gate_up_proj", "_experts_gate_up_proj_weight"),
    ("experts.down_proj", "_experts_down_proj_weight"),
])


def _apply_mapping(weight_name: str) -> str:
    mapped = weight_name
    for before, after in GPT_OSS_SAFETENSOR_MAP.items():
        mapped = mapped.replace(before, after)
    return mapped


def convert_safetensor_state_dict(state_dict: dict[str, Weights], **kwargs) -> dict[str, WeightData]:
    """Convert a safetensor state dict to the MAX naming convention.

    Unknown keys are rejected to avoid silently dropping or misnaming weights.
    """
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        mapped = _apply_mapping(weight_name)
        if mapped == weight_name and not any(
            weight_name.startswith(prefix) for prefix in ("language_model.", "model.")
        ):
            raise ValueError(f"Unknown weight name '{weight_name}' in safetensor map.")
        data = value.data() if hasattr(value, "data") else value
        new_state_dict[mapped] = data

    return new_state_dict


__all__ = ["GPT_OSS_SAFETENSOR_MAP", "convert_safetensor_state_dict"]

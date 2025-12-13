"""GPT-OSS configuration helpers.

This module keeps the Hugging Face config as the source of truth and derives
the MAX runtime configuration from it. Validation here prevents downstream
code from guessing at shapes or silently accepting unsupported settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    YarnScalingParams,
)


def _normalize_rope_scaling(rope_scaling: Any) -> YarnScalingParams:
    if rope_scaling is None:
        raise ValueError("RoPE scaling is required for GPT-OSS.")
    rope_type = (
        rope_scaling.get("rope_type") if isinstance(rope_scaling, dict) else getattr(rope_scaling, "rope_type", None)
    )
    if rope_type is None:
        raise ValueError("RoPE scaling is required for GPT-OSS.")
    if rope_type == "linear":
        raise ValueError("Linear RoPE scaling is not supported for GPT-OSS.")
    if rope_type != "yarn":
        raise ValueError(f"Unsupported RoPE scaling type: {rope_type}")

    params = rope_scaling if isinstance(rope_scaling, dict) else rope_scaling.__dict__
    return YarnScalingParams(
        rope_type=rope_type,
        factor=params["factor"],
        beta_fast=params["beta_fast"],
        beta_slow=params["beta_slow"],
        original_max_position_embeddings=params["original_max_position_embeddings"],
        truncate=params.get("truncate", False),
    )


def _layer_types_or_default(hf_config: Any) -> list[str]:
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types:
        return list(layer_types)
    # Fallback: alternate sliding/full attention.
    num_layers = hf_config.num_hidden_layers
    pattern = ["sliding_attention", "full_attention"]
    return [pattern[i % len(pattern)] for i in range(num_layers)]


@dataclass
class GptOssMXFP4Config(MAXModelConfig):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    sliding_window: int
    rope_scaling: YarnScalingParams
    hidden_activation: str
    tie_word_embeddings: bool
    layer_types: list[str]
    router_aux_loss_coef: float
    num_experts_per_tok: int
    num_local_experts: int
    swiglu_limit: float
    attention_bias: bool
    dtype: DType
    devices: list
    kv_params: KVCacheParams
    return_logits: ReturnLogits | None = ReturnLogits.LAST_TOKEN
    interleaved_rope_weights: bool = False
    query_pre_attn_scalar: float | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None

    @staticmethod
    def get_kv_params(
        hf_config: Any,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Derive KV cache params directly from the HF config."""
        ctor_kwargs = dict(
            dtype=cache_dtype,
            n_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            cache_strategy=getattr(kv_cache_config, "cache_strategy", None),
            n_devices=n_devices,
            enable_prefix_caching=getattr(kv_cache_config, "enable_prefix_caching", False),
            enable_kvcache_swapping_to_host=getattr(kv_cache_config, "enable_kvcache_swapping_to_host", False),
            host_kvcache_swap_space_gb=getattr(kv_cache_config, "host_kvcache_swap_space_gb", 0.0),
        )
        page_size = getattr(kv_cache_config, "kv_cache_page_size", None)
        num_layers = hf_config.num_hidden_layers
        try:
            params = KVCacheParams(
                **ctor_kwargs,
                page_size=page_size,
                num_layers=num_layers,
            )
        except TypeError:
            params = KVCacheParams(**ctor_kwargs)  # type: ignore[arg-type]
            params.page_size = page_size
            params.num_layers = num_layers
        return params

    @staticmethod
    def get_num_layers(hf_config: Any) -> int:
        return hf_config.num_hidden_layers

    @staticmethod
    def generate(
        pipeline_config: PipelineConfig,
        hf_config: Any,
        state_dict: dict[str, Any],
        return_logits: ReturnLogits | None,
        kv_cache_config: KVCacheConfig,
        devices: list,
        dtype: DType | None = None,
    ) -> GptOssMXFP4Config:
        rope_scaling = _normalize_rope_scaling(getattr(hf_config, "rope_scaling", None))

        hidden_activation = (
            "gelu_tanh" if getattr(hf_config, "hidden_act", None) == "gelu_pytorch_tanh" else hf_config.hidden_act
        )
        tie_word_embeddings = "language_model.lm_head.weight" not in state_dict

        layer_types = _layer_types_or_default(hf_config)

        cfg = GptOssConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            max_position_embeddings=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
            sliding_window=hf_config.sliding_window,
            rope_scaling=rope_scaling,
            hidden_activation=hidden_activation,
            tie_word_embeddings=tie_word_embeddings,
            layer_types=layer_types,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            num_local_experts=hf_config.num_local_experts,
            swiglu_limit=hf_config.swiglu_limit,
            attention_bias=getattr(hf_config, "attention_bias", True),
            dtype=dtype or getattr(hf_config, "dtype", DType.bfloat16),
            devices=devices,
            kv_params=GptOssConfig.get_kv_params(
                hf_config=hf_config,
                n_devices=len(devices) if devices else 1,
                kv_cache_config=kv_cache_config,
                cache_dtype=dtype or getattr(hf_config, "dtype", DType.bfloat16),
            ),
            return_logits=return_logits or ReturnLogits.LAST_TOKEN,
            interleaved_rope_weights=getattr(hf_config, "interleaved_rope_weights", False),
            query_pre_attn_scalar=getattr(hf_config, "query_pre_attn_scalar", None),
            final_logit_softcapping=getattr(hf_config, "final_logit_softcapping", None),
            attn_logit_softcapping=getattr(hf_config, "attn_logit_softcapping", None),
        )
        return cfg


__all__ = ["GptOssMXFP4Config"]

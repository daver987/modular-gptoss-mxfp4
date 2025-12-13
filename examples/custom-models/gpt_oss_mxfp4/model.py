"""Pipeline entrypoint for GPT-OSS."""

from __future__ import annotations

from typing import Any

from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn import ReturnLogits
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SupportedEncoding

from .gpt_oss import GptOss
from .model_config import GptOssConfig


class GptOssModelMXFP4:
    """Thin wrapper that mirrors the MAX pipeline model API."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: Any,
        encoding: SupportedEncoding,
        devices: list,
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits | None = None,
    ) -> None:
        self.config = GptOssConfig.generate(
            pipeline_config=pipeline_config,
            hf_config=huggingface_config,
            state_dict=getattr(weights, "state_dict", lambda: {})(),
            return_logits=return_logits,
            kv_cache_config=kv_cache_config,
            devices=devices,
            dtype=encoding,
        )
        # Keep HF config and runtime config aligned to avoid silent drift.
        assert self.config.num_hidden_layers == huggingface_config.num_hidden_layers
        assert self.config.num_attention_heads == huggingface_config.num_attention_heads
        assert self.config.hidden_size == huggingface_config.hidden_size
        assert self.config.head_dim == huggingface_config.head_dim
        assert self.config.max_position_embeddings == huggingface_config.max_position_embeddings
        self.session = session
        self.model = GptOss(self.config)

    # Placeholder hook for API completeness; actual graph building handled by MAX.
    def load(self) -> None:  # pragma: no cover - runtime hook
        return


__all__ = ["GptOssModelMXFP4"]

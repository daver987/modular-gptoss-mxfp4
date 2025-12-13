"""GPT-OSS architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.kv_cache import KVCacheStrategy
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from . import weight_adapters
from .model import GptOssModel

gpt_oss_arch = SupportedArchitecture(
    name="GptOssForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=GptOssModel,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)

__all__ = ["WeightsFormat", "gpt_oss_arch"]

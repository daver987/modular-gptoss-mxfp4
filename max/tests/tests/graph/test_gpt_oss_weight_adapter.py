# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import numpy as np
import pytest
from max.dtype import DType
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from max.graph import DeviceRef
from max.nn import ReturnLogits, YarnScalingParams
from max.nn.kv_cache.cache_params import KVCacheParams

from max.pipelines.architectures.gpt_oss import weight_adapters
from max.pipelines.architectures.gpt_oss.layers.moe import GptOssMoE
from max.pipelines.architectures.gpt_oss.model_config import GptOssConfig


class _StubWeight:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self._data = WeightData.from_numpy(array, name)

    def data(self) -> WeightData:  # noqa: D401
        """Return the underlying WeightData."""
        return self._data


def test_weight_adapter_keeps_mxfp4_pairs() -> None:
    blocks = np.arange(16, dtype=np.uint8).reshape(1, 1, 16)
    scales = np.arange(1, dtype=np.uint8).reshape(1, 1, 1)

    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj.blocks": _StubWeight(
            "gate.blocks", blocks
        ),
        "model.layers.0.mlp.experts.gate_up_proj.scales": _StubWeight(
            "gate.scales", scales
        ),
        "model.layers.0.mlp.experts.down_proj.blocks": _StubWeight(
            "down.blocks", blocks
        ),
        "model.layers.0.mlp.experts.down_proj.scales": _StubWeight(
            "down.scales", scales
        ),
        "model.layers.0.mlp.router.weight": _StubWeight(
            "router", np.zeros((1, 1), dtype=np.float32)
        ),
    }

    converted = weight_adapters.convert_safetensor_state_dict(state_dict)

    assert (
        "language_model.layers.0.mlp.experts.gate_up_proj.blocks" in converted
    )
    gate_blocks = converted[
        "language_model.layers.0.mlp.experts.gate_up_proj.blocks"
    ]
    assert gate_blocks.dtype == DType.uint8
    assert gate_blocks.quantization_encoding == QuantizationEncoding.MXFP4

    gate_scales = converted[
        "language_model.layers.0.mlp.experts.gate_up_proj.scales"
    ]
    assert gate_scales.quantization_encoding == QuantizationEncoding.MXFP4

    router_weight = converted[
        "language_model.layers.0.mlp.gate.gate_score.weight"
    ]
    assert router_weight.quantization_encoding is None
    assert router_weight.dtype == DType.float32


def test_weight_adapter_errors_on_incomplete_pair() -> None:
    state_dict = {
        "model.layers.0.mlp.experts.gate_up_proj.blocks": _StubWeight(
            "gate.blocks", np.zeros((1, 1, 16), dtype=np.uint8)
        ),
    }

    with pytest.raises(ValueError):
        weight_adapters.convert_safetensor_state_dict(state_dict)


def test_moe_allocates_quantized_weights() -> None:
    rope = YarnScalingParams(
        factor=1.0,
        beta_fast=1.0,
        beta_slow=1.0,
        original_max_position_embeddings=128,
        truncate=False,
    )
    kv_params = KVCacheParams(
        dtype=DType.float32,
        n_kv_heads=2,
        head_dim=8,
        page_size=128,
        n_devices=1,
    )
    config = GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        hidden_activation="gelu",
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_theta=1.0,
        attention_bias=False,
        sliding_window=16,
        rope_scaling=rope,
        num_local_experts=2,
        num_experts_per_tok=1,
        router_aux_loss_coef=0.0,
        layer_types=["full_attention", "full_attention"],
        attention_dropout=0.0,
        query_pre_attn_scalar=None,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        swiglu_limit=7.0,
        dtype=DType.float32,
        devices=[DeviceRef.CPU()],
        interleaved_rope_weights=False,
        return_logits=ReturnLogits.LAST_TOKEN,
        kv_params=kv_params,
        quantization="mxfp4",
    )

    moe = GptOssMoE(config)

    gate_weights = moe._mxfp4_gate_up_proj
    assert gate_weights is not None
    assert gate_weights.blocks.shape == [
        config.num_local_experts,
        2 * config.intermediate_size,
        config.hidden_size // 2,
    ]
    assert gate_weights.blocks.dtype == DType.uint8
    assert gate_weights.blocks.quantization_encoding == QuantizationEncoding.MXFP4

    down_weights = moe._mxfp4_down_proj
    assert down_weights is not None
    assert down_weights.blocks.shape == [
        config.num_local_experts,
        config.hidden_size,
        config.intermediate_size // 2,
    ]

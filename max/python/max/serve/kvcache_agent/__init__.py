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

# Ensure generated proto stubs are importable when running from a source checkout.
# Bazel writes kvcache_agent_service_v1_pb2*.py to bazel-bin/max/python/max/serve/kvcache_agent.
# When this package is imported directly from the repo, add that directory to __path__ if present.
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    candidate = parent / "bazel-bin" / "max" / "python" / "max" / "serve" / "kvcache_agent"
    if candidate.exists():
        __path__.append(str(candidate))
        break

from .dispatcher_v2 import DispatcherClientV2, DispatcherServerV2
from .kvcache_agent import start_kvcache_agent_service

__all__ = [
    "DispatcherClientV2",
    "DispatcherServerV2",
    "start_kvcache_agent_service",
]

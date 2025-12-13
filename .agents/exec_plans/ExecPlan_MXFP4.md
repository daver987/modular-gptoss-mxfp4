# MXFP4 GPT-OSS SM90 Kernel and Architecture

This ExecPlan is a living document. Maintain it in line with `.agents/PLANS.md` and `AGENTS.md`; keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current as work proceeds. Critical local rules: do not modify existing Modular kernel code under `max/`; add all new code under `examples/`. MXFP4 dequantization must stay fused inside the GEMM (decode in registers after staging packed data in shared), mirroring the Triton reference. Make sure to make use of the debugging tools and the Mojo test runner. These are critical to making sure everything is correct.

## Purpose / Big Picture

Enable GPT-OSS (20B/120B) to run natively in MXFP4 on H100 (SM90) with a custom Mojo kernel that decodes MXFP4 weights inside the GEMM and fuses the SwiGLU epilogue, wired into a MAX pipeline. After implementation, a user should be able to load MXFP4 checkpoints and execute generation via `max.entrypoints.pipelines` using the custom architecture, with MoE MLP1/MLP2 backed by the MXFP4 SM90 kernel rather than BF16 fallbacks.

## Feasibility, Preconditions, and Risks

Pre-flight checks refreshed 2025-12-10 02:26Z: `nvidia-smi` shows an H100 80GB (SM90); `mojo --version` is 0.26.1.0.dev2025120705; `python` imports `max`; `pixi --version` is 0.60.0. Current code under `examples/custom_ops/kernels/` includes MXFP4 decode utils plus a CPU reference and naive GPU fallback for `gpt_oss.mxfp4.matmul.sm90`; Python now has a wrapper and dummy MXFP4 pack/decode helpers plus a CPU smoke test. Architecture wiring is still absent; `weight_loader.py` holds only a reference decode, synthetic packer, and a static 20B weight map scaffold. Tests: one CPU smoke test exists but is not yet wired to pixi or run in this session. Assumptions: MXFP4 checkpoints (blocks + scales) will be available or synthetic weights will be used for correctness-only checks. Risks: correctness/perf of SM90 wgmma fragment mapping; mitigate by copying `max/kernels/src/linalg/matmul/gpu/sm90/*` ordering and Triton tiling (BLOCK_M/N=128, BLOCK_K≈64, FRAG_K≈16–32). Also track MAX Python API drift by keeping wrappers local in `examples/custom-models/gpt_oss_mxfp4/`. Make sure to use the debug tools which are supplied. Do not make guesses use the tools and refresh yourself by looking at the skills. Make sure to use the Mojo test runner as well and not pytest, pytest is for python Mojo has its own built in test runner.

## Progress

- [x] (2025-12-10 00:46Z) Read AGENTS/OVERVIEW, MXFP4 key docs, Triton reference; inspected existing MAX GPT-OSS code and confirmed GPU/tools availability.
- [x] (2025-12-10 01:41Z) Added MXFP4 decode helpers and registered `gpt_oss.mxfp4.matmul.sm90` CPU reference kernel skeleton in `examples/custom_ops/kernels/` (SM90 path still TODO).
- [x] (2025-12-10 02:00Z) Added naive GPU fallback kernel (per-element MXFP4 decode on device) to unblock graph execution; still need optimized SM90 wgmma path. Python wrapper + dummy MXFP4 pack/decode helpers added under `examples/custom-models/gpt_oss_mxfp4/`.
- [x] (2025-12-10 02:15Z) Added CPU correctness smoke test for the custom op using synthetic MXFP4 weights (`examples/custom-models/gpt_oss_mxfp4/tests/test_mxfp4_matmul.py`); SM90 perf tests still pending.
- [x] (2025-12-10 02:26Z) Re-oriented with PLANS/AGENTS, re-read SM90/Triton refs, revalidated H100+tooling state, confirmed custom op wrapper/test stubs exist; ready to replace naive GPU path with SM90 wgmma and build architecture.
- [x] (2025-12-10 02:36Z) Reviewed `.agents/skills/debugging-mojo` (LLDB/CUDA-GDB flows, `mojo debug --cuda-gdb --break-on-launch`) and `.agents/skills/testing-mojo` (TestSuite/discover_tests via `mojo run`); will apply these for kernel debugging and Mojo-side tests.
- [x] (2025-12-10 02:40Z) Added test dependency scaffold to `examples/custom-models/pixi.toml`.
- [x] (2025-12-10 02:42Z) Added per-byte MXFP4 decode helper (`mxfp4_decode_byte_to_f32_pair`) to support warp-level decode for the upcoming SM90 path.
- [x] (2025-12-10 02:44Z) Refactored `mxfp4_matmul_sm90.mojo` GPU dispatch: centralized shape validation, split naive GPU launch helper, and added a dedicated `target="sm90"` branch (currently shares the naive kernel until the wgmma path lands).
- [x] (2025-12-10 02:45Z) Defaulted Python wrapper target to `"sm90"` (still override to `"cpu"` in tests) and clarified target semantics.
- [x] (2025-12-10 02:46Z) Reused centralized shape validation in the CPU path to avoid divergence and prep for SM90 upgrades.
- [x] (2025-12-12 05:27Z) Reviewed MoE MXFP4 Mojo kernels (`mxfp4_decode.mojo`, `moe_mxfp4_kernels.mojo`, `moe_mxfp4_ops.mojo`) against the Triton reference; confirmed decode/layout/epilogue contracts match, with remaining gaps being perf-only internal details.
- [ ] Implement SM90 GPU path (warp-level decode + wgmma) to replace the naive fallback and land real performance.
- [ ] Implement GPT-OSS MXFP4 architecture wiring under `examples/custom-models/gpt_oss_mxfp4/` (attention/MoE using the custom op).
- [ ] Add tests/benchmarks and run smoke validations on H100.
- [ ] Update this plan with discoveries, decisions, and retrospective.

## Surprises & Discoveries

- H100 80GB is available locally; good for SM90 work without further setup.
- Custom op currently has a CPU path and a naive GPU fallback that decodes per-thread; no SM90 wgmma fast path yet, so wrapper defaults to `target="cpu"` to keep tests running.
- Python side now includes MXFP4 pack/decode helpers, wrapper, and a CPU smoke test; the actual GPT-OSS MXFP4 architecture is still missing, and `weight_loader.py` only contains reference helpers plus a 20B weight-map scaffold.
- Triton reference (`examples/custom-models/triton_example/moe.py`) already fuses matmul+SwiGLU with MXFP4 decode; it remains the behavioral template to mirror in Mojo.
- Debugging skill docs are available (`.agents/skills/debugging-mojo/SKILL.md`), prescribing `mojo debug --cuda-gdb --break-on-launch` for GPU kernels and LLDB for host paths; use them instead of ad-hoc debugging.
- Mojo testing guidance (`.agents/skills/testing-mojo/SKILL.md`) notes the `TestSuite.discover_tests[__functions_in_module()]().run()` pattern run via `mojo run <testfile.mojo>` since `mojo test` was removed; prefer this for kernel unit tests.
- MoE MXFP4 kernels in `examples/custom_ops/kernels/moe_mxfp4_kernels.mojo` already follow the Triton fused decode+GEMM+SwiGLU / gamma-scatter pattern; any upcoming rewrites (vectorized decode, K-stage pipelining, WGMMA/TMA) should preserve their public signatures.

## Decision Log

- Decision: Target SM90 tiling BLOCK_M=128, BLOCK_N=128, BLOCK_K≈64 with FRAG_K 16–32, decode MXFP4 in registers per warp and fuse SwiGLU epilogue (limit=7.0, alpha=1.702, interleaved gate/up layout).  
  Rationale: Matches `.agents/OVERVIEW.md` and Triton `matmul_ogs` pattern to keep the kernel compute-bound and minimize shared-memory footprint.  
  Date/Author: 2025-12-10 Codex
- Interim decision: expose custom op as `gpt_oss.mxfp4.matmul.sm90` with CPU reference path first and explicit `target` parameter; keep SM90 GPU path behind TODO to unblock Python integration and tests.  
  Date/Author: 2025-12-10 Codex
- Decision: Treat the MoE custom op interfaces (`mxfp4_moe_w1_swiglu`, `mxfp4_moe_w2_scatter`) and packed MXFP4 layouts as locked to match Triton; proceed with Python architecture wiring now and only do kernel perf work later behind the same contract.  
  Rationale: Prevents churn in the Python/architecture layer; the remaining delta to Triton is internal performance engineering.  
  Date/Author: 2025-12-12 Codex

## Outcomes & Retrospective

To be filled as milestones complete.

## Context and Orientation

Relevant docs: `.agents/OVERVIEW.md` (SM90 kernel shape, wgmma fragment handling, fused SwiGLU), `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md` (must keep decode fused, block size 32 values, power-of-two float8 scales, avoid extra memory traffic), `.agents/ref_docs/MXFP4.md` (format details), `.agents/ref_docs/GPT_OSS_OVERVIEW.md` (model architecture: 36 layers, MoE top-4, alternating dense/banded attention, SWIGLU with clamp).

Repository layout for this task: new Mojo kernels live in `examples/custom_ops/kernels/`; custom Python architecture in `examples/custom-models/gpt_oss_mxfp4/`; Triton reference in `examples/custom-models/triton_example/` shows the desired behavior. Existing MAX GPT-OSS (BF16) lives under `max/python/max/pipelines/architectures/gpt_oss/` and can be used for control flow and config patterns but must not be modified. Use `examples/custom-models/pixi.toml` to manage Python deps/tasks. Current custom code: `examples/custom_ops/kernels/mxfp4_utils.mojo` and `examples/custom_ops/kernels/mxfp4_decode.mojo` provide FP4 LUT/decode + SwiGLU helpers; `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo` registers `gpt_oss.mxfp4.matmul.sm90` with a CPU reference path and naive GPU fallback (no SM90 wgmma); `examples/custom_ops/kernels/moe_mxfp4_kernels.mojo` implements fused MoE W1 (gather+GEMM+SwiGLU) and W2 (GEMM+gamma+scatter-add) using MXFP4; `examples/custom_ops/kernels/moe_mxfp4_ops.mojo` registers `mxfp4_moe_w1_swiglu` and `mxfp4_moe_w2_scatter`; `examples/custom-models/gpt_oss_mxfp4/kernels.py` wraps the dense op; `examples/custom-models/gpt_oss_mxfp4/tests/test_mxfp4_matmul.py` provides a CPU smoke test; `weight_loader.py` exposes a reference decode, dummy MXFP4 packer, and a static 20B weight-map scaffold; `model_config.py` mirrors the 20B hyperparameters.

MXFP4 layout: groups of 32 FP4(E2M1) values packed into 16 bytes with one float8_e8m0fnu scale per group (power-of-two). Decoding: nibble→LUT→scale (ldexp)→FP16; scales stay packed alongside blocks. Kernels must stage packed blocks and scales in shared, decode per warp into register fragments, then issue wgmma; dequantizing to BF16 in shared is forbidden for perf. SwiGLU epilogue clamps gate/up to ±limit, computes gate*sigmoid(alpha*gate)*(up+1) with interleaved even/odd columns.

Debug/testing resources: use `.agents/skills/debugging-mojo/SKILL.md` for LLDB/CUDA-GDB workflows (`mojo debug --cuda-gdb --break-on-launch` for GPU kernels, `-O0 -g` builds when using binaries). For Mojo-side tests, the old `mojo test` is gone; write `test_*` functions and run via `TestSuite.discover_tests[__functions_in_module()]().run()` inside `main()`, executed with `mojo run <testfile.mojo>` (`--only/--skip` supported).

## Plan of Work

Upgrade the MXFP4 kernel in `examples/custom_ops/kernels/mxfp4_matmul_sm90.mojo`: keep the CPU reference path but add an SM90 fast path (BLOCK_M/N≈128, BLOCK_K≈64, FRAG_K 16–32, ~8 warps/CTA) that stages packed B bytes + scales and BF16 A tiles in shared, decodes B into register fragments in the wgmma order, issues `wgmma`, and applies the fused SwiGLU epilogue. Reuse SM90 fragment/tile helpers from `max/kernels/src/linalg/matmul/gpu/sm90/*` via imports only; do not touch upstream files. Dispatch on `target` (`cpu` vs `sm90`), validating shapes (K divisible by 32, gate/up interleave).

Extend helpers locally if needed (e.g., a warp-level B-fragment loader and scale indexing utilities) inside `examples/custom_ops/kernels/` rather than editing `max/`. Document the packed layout and fragment mapping inline so the SM90 path remains self-contained.

Tighten the Python wrapper in `examples/custom-models/gpt_oss_mxfp4/kernels.py`: keep shape checks, default to `target="sm90"`. Ensure scales are cast to `float8_e8m0fnu` before dispatch and that `custom_extensions` points at `examples/custom_ops/kernels`.

Finish weight loading in `examples/custom-models/gpt_oss_mxfp4/weight_loader.py`: reshape blocks/scales to `[K/32, N]` (or expert-aware shapes), and cast scales to float8. Keep the reference decode and dummy generator for tests. Update `model_config.py` if additional runtime knobs (dtype, swiglu_limit, experts_per_token) are needed.

Build the MXFP4 architecture under `examples/custom-models/gpt_oss_mxfp4/`: mirror the control flow of `max/pipelines/architectures/gpt_oss/` but redirect MoE MLP1 to the fused MoE op `mxfp4_moe_w1_swiglu` (routing gather + W1 MXFP4 GEMM + SwiGLU) and MLP2 to `mxfp4_moe_w2_scatter` (W2 MXFP4 GEMM + gamma + scatter-add). Keep `mxfp4_matmul_swiglu` for standalone dense tests/debug only. Implement attention layers (dense/banded alternating, RoPE, GQA) using existing MAX ops, register the architecture/pipeline entrypoints, and wire checkpoint loading to the packed weights.

Add validation and tooling: strengthen `tests/test_mxfp4_matmul.py` to compare against a torch dequant+matmul+SwiGLU reference on SM90, and add an optional benchmark script to time SM90 vs dequant matmul. Document an end-to-end generation command using synthetic weights or a real checkpoint.

Add Mojo-side unit tests using `testing.TestSuite.discover_tests[__functions_in_module()]().run()` run via `mojo run <testfile.mojo>`, when we add pure-Mojo decode/math checks; keep them under `examples/custom_ops/kernels/tests/`.

## Concrete Steps

Work in `/workspace/modular-gptoss-mxfp4`. Commands already run (2025-12-10 02:26Z):

    nvidia-smi  # H100 80GB (SM90) visible
    mojo --version
    python - <<'PY'\nimport sys\nimport max\nprint(sys.version)\nprint('max version', getattr(max,'__version__','unknown'))\nPY
    pixi --version

Upcoming steps during execution (update as completed):

    # Implement SM90 path in mxfp4_matmul_sm90.mojo and keep CPU/debug paths
    # Tighten Python wrapper/loader and add pixi deps/tasks under examples/custom-models
    pytest examples/custom-models/gpt_oss_mxfp4/tests/test_mxfp4_matmul.py -q  # once deps installed
    pixi run -p examples/custom-models mxfp4-matmul-test      # after task exists
    pixi run -p examples/custom-models gpt-oss-mxfp4-smoke    # synthetic smoke
    python -m max.entrypoints.pipelines generate --custom-architectures examples/custom-models/gpt_oss_mxfp4 --model <checkpoint> --prompt "Hello"  # real weights
    # For Mojo tests: mojo run examples/custom_ops/kernels/tests/test_*.mojo --only <name>  # uses TestSuite.discover_tests

## Validation and Acceptance

Acceptance requires: (1) MXFP4 matmul+SwiGLU custom op matches a Python/torch dequant+matmul reference within BF16 tolerance on representative shapes (including expert-partitioned layouts); CPU smoke test (`tests/test_mxfp4_matmul.py`) must pass. (2) SM90 path runs and outperforms the naive GPU fallback without expanding B to shared; code inspection plus an optional benchmark should show decode-in-register behavior. (3) End-to-end GPT-OSS MXFP4 graph runs a short prompt without errors using synthetic weights, and with real MXFP4 checkpoints. All pixi tasks added for this work must succeed.

## Artifacts and Notes

Evidence from pre-flight:

    nvidia-smi -> H100 80GB HBM3, CUDA 13.0, no active procs.
    mojo --version -> 0.26.1.0.dev2025120705
    pixi --version -> 0.60.0

Note that Mojo no longer needs numpy, torch, cuda or othere deps. as it is becoming more and more self contained. Use max to load weights if necessary. The weights are already downloaded in the /workspace/huggingface dir. Also note it is critical that we do not take any short cuts, MXFP4 will only show speedup gains when it is implemented like the existing Triton  example. See `MXFP4_KEY_TAKEAWAYS` for more details.

## Interfaces and Dependencies

Kernel interface (current/target): custom op name `gpt_oss.mxfp4.matmul.sm90` with `execute[target: StaticString]` taking `(output: OutputTensor[rank=2], a: InputTensor[bf16, rank=2], b_packed: InputTensor[uint8, rank=3], b_scales: InputTensor[float8_e8m0fnu, rank=2], bias: InputTensor[bf16, rank=1], alpha: Float32, limit: Float32, ctx: DeviceContextPtr)`; output shape `[M, N/2]` for fused gate/up columns. Packed layout: `[K/32, N, 16]` bytes (two FP4 per byte) with scales shaped `[K/32, N]`; K must be divisible by 32 and N even (interleaved gate/up). Python wrapper signature `mxfp4_matmul_swiglu(x, w_blocks, w_scales, bias, alpha=1.702, limit=7.0, target="sm90"| "cpu"| "debug") -> TensorValue`, passing `custom_extensions=[examples/custom_ops/kernels]`.

MoE kernel interfaces (locked): custom ops `mxfp4_moe_w1_swiglu` and `mxfp4_moe_w2_scatter` in `examples/custom_ops/kernels/moe_mxfp4_ops.mojo`. `mxfp4_moe_w1_swiglu` takes `x [T,D] bf16`, routing tensors (`token_expert_order [P] u32`, `expert_start_indices [num_active+1] u32`, `expert_ids [num_active] i32`, `max_tokens_per_expert [1] u32`), MXFP4 W1 weights (`blocks [num_experts,D,Nblocks,16] u8`, `scales [num_experts,D,Nblocks] float8_e8m0fnu`), and per-expert bias `[num_experts,2*I] f32`, and produces `h_sorted [P,I] bf16`. `mxfp4_moe_w2_scatter` takes `h_sorted [P,I] bf16`, the same routing tensors, `gate_weights [P] f16` in original pair order, MXFP4 W2 weights (`blocks [num_experts,I,Nblocks,16] u8`, `scales [num_experts,I,Nblocks] float8_e8m0fnu`), per-expert bias `[num_experts,D] f32`, and scatters into `y [T,D] bf16` (must be zeroed before launch). Future perf work must keep these signatures and layouts stable.

Dependencies to declare: Use MAX Graph ops (`ops.custom`, `ops.softmax`, `moe_create_indices`, etc.) and existing attention utilities from `max.nn`. Ensure pixi env under `examples/custom-models` includes new deps so tasks run reproducibly. You can use other custom arch. to learn the conventions for integration.

Update 2025-12-10 02:26Z: Refreshed feasibility with current environment checks, recorded existing CPU wrapper/test state, and focused upcoming work on the SM90 wgmma path plus environment/test wiring.
Update 2025-12-10 02:36Z: Incorporated `.agents/skills` debugging/testing guidance (mojo debug with CUDA-GDB, TestSuite-based Mojo tests) into the plan and noted usage for upcoming SM90 and test work.
Update 2025-12-12 05:27Z: Recorded MoE MXFP4 kernel/Triton alignment review, locked public contracts (MoE ops + packed layout), and updated the architecture plan to wire Python against these ops now; perf upgrades remain kernel-internal follow-ons.

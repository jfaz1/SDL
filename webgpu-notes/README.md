# SDL_GPU WebGPU Notes

These notes describe the implementation rationale, boundaries, and tradeoffs
of an SDL_GPU WebGPU backend and its associated offline shader tooling, as
found in this source snapshot. They are not an API reference, and they are not
an upstream SDL design document; `include/SDL3/SDL_gpu.h` and the backend
source remain the authoritative interface and behavior references.

## Shape Of The Current Design

The public surface for this backend is deliberately small:

- WGSL shaders are accepted through `SDL_GPU_SHADERFORMAT_WGSL` and must
  follow SDL's documented WebGPU binding convention and supported resource
  subset.
- Shader facts that WebGPU needs beyond plain resource counts are supplied as
  SDL-owned typed data through `SDL_CreateGPUShaderWithResourceLayout()` and
  `SDL_CreateGPUComputePipelineWithResourceLayout()`. These facts are intended
  to be generated offline by shader tooling, not recovered at runtime.
- The existing SDL_GPU full-resource bind and pass APIs, including the
  write-path mip/layer selectors already present in pass and binding structs,
  remain the public resource model.

Retained public texture/buffer view handles, view bind variants, pass
description overloads, compatible-format texture declarations, fragment-stage
writable storage, and public read-path subresource selectors are not part of
the current API. Each of those would add public surface and lifetime semantics
that current portability evidence does not justify. The backend models views
internally where the provider requires them.

Two rules shape most of the implementation:

- SDL core and the WebGPU backend never parse or reflect shader source at
  runtime, and never take a runtime dependency on shader producer tooling.
- Unsupported feature/provider combinations fail with clear SDL errors instead
  of silently degrading to different rendering behavior.

## Topics

- `backend-architecture.md`: renderer structure, asynchronous provider
  bridging, lifetime and fences, command encoding, bind groups, and error
  surfacing.
- `adaptation-boundaries.md`: SDL_GPU features that map directly to WebGPU,
  features that require narrowing, and features that fail explicitly.
- `shader-resource-bridge.md`: the WGSL binding convention, shader resource
  layout facts, and the division of responsibility between SDL, the backend,
  and offline shader producers.
- `validation-and-future-work.md`: validation strategy, provider caveats, and
  the evidence future growth would need.

## Evidence Scope

Support statements in these notes describe the source snapshot they accompany.
Browser evidence is concentrated on Chrome through Emscripten/Emdawn; other
browsers and native WebGPU runtimes currently have narrower or no evidence.
Native Metal, Vulkan, and D3D12 backends are parity references for shared
SDL_GPU semantics, not proof that every WebGPU provider has the same feature
coverage or numeric behavior.

# Validation And Future Work

Validation for a WebGPU backend needs two axes. Native Metal, Vulkan, and
D3D12 runs check that shared SDL_GPU semantics stay consistent where the
WebGPU work touches shared code or public API. Browser runs exercise what the
native backends cannot: the WebGPU adapter, Emscripten integration, WGSL
shaders and their layout facts, presentation behavior, and provider
limitations. Neither axis substitutes for the other.

## Validation Strategy

Focused validation should match the touched area:

- Public SDL_GPU API or shared validation changes need native backend
  coverage, because a WebGPU-motivated change must not shift shared behavior.
- WebGPU backend changes need browser coverage, including provider error-scope
  and runtime-error surfacing checks.
- Shader layout and producer changes need shader-tooling tests plus a real
  application build that consumes generated layout sidecars end to end.
- Format, copy, storage-texture, and view-shape changes need exact-format
  runtime cases. Coarse support-query passes are not sufficient: creation,
  binding, and submission validate facts that queries cannot see.

Negative testing is first-class rather than incidental. Because the design
contract is that unsupported combinations fail clearly, tests assert that
rejected paths produce SDL errors at the documented boundary without provider
crashes or partial state. App-style workloads such as asset upload, shadow and
post-processing passes, UI overlays, and readback complement unit cases by
exercising binding churn, transfer alignment, and presentation pacing in
combinations unit suites rarely reproduce.

Image readback and diffing are useful diagnostics, especially for catching
binding mistakes, format mismatches, and copy/readback errors. Cross-backend
image diffs should start as diagnostics rather than strict pass/fail gates:
screen-space precision, transcendental functions, atomics, derivatives, and
provider-specific rasterization create legitimate small differences, and a
strict gate on those teaches people to ignore red. Strict thresholds are worth
adopting only per-case, where a scene is known to be deterministic across the
compared targets.

## Provider Caveats

Chrome through Emscripten/Emdawn is a practical baseline, but it is one
provider on one platform family, not proof for every WebGPU implementation.
Other browsers and native WebGPU runtimes differ in available optional
features, limits, validation strictness, and shader translation behavior.
Support claims should name the evidence they come from, and broadening a claim
to a new provider needs a capability probe plus scoped validation for the
exact claim being made.

Packaging is part of the evidence scope. The supported browser shape is SDL
as a static library with Asyncify, because the synchronous fence and wait
contract is bridged through timed future waits that need a suspension
mechanism. Alternate suspension approaches (such as JSPI) and other packaging
modes need complete application-loop coverage before they can replace that,
not just isolated rendering passes.

Some limitations are platform maturity issues rather than SDL design issues:
optional texture features, translator behavior, and async packaging options
all continue to evolve. The backend's obligation is unchanged either way: fail
clearly when a provider cannot support a requested SDL_GPU feature.

## Evidence Needed For Future API Growth

The current public surface excludes several conveniences on purpose. Each has
a concrete evidence bar rather than a permanent prohibition:

- Retained public texture/buffer views and read-path subresource selectors
  would define SDL-wide lifetime, identity, and cache semantics. They need
  multi-provider behavioral evidence and application consumers whose needs the
  full-resource paths demonstrably cannot express. Validation-only usage is
  not enough to justify public API.
- Compatible-format reinterpretation needs its own design and exact-format
  cases per provider, since format aliasing rules differ across APIs.
- Fragment-stage writable storage needs portable semantics across native
  backends and WebGPU providers, plus real consumer pressure.
- Broader presentation claims (additional present modes, HDR compositions)
  need provider evidence and an SDL-visible presentation policy.

## Performance Follow-Ups

Renderer-global bind-group caching, resource-reference counters, allocation
pooling, and shared layout caches are compatible with the current
architecture and deliberately deferred. The backend's internal instrumentation
exists to make these decisions measurable; they should be adopted only if
representative workloads show enough churn to justify the added lifetime,
reset, eviction, and device-loss complexity. Reducing producer-side friction
for layout sidecars is a parallel track, with the fixed constraint that shader
reflection stays out of SDL runtime code.

## Non-Goals Worth Keeping Explicit

The current model intentionally avoids runtime shader parsing, arbitrary WGSL
group/binding layouts, shader-visible descriptor array emulation, texture
atomic emulation, multisampled storage textures, and hidden fallbacks for
unsupported provider features. These choices keep the runtime boundary clear,
keep SDL free of shader-toolchain dependencies, and make unsupported features
visible to applications instead of turning them into silent per-provider
rendering differences.

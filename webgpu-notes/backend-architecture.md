# Backend Architecture

The WebGPU backend follows SDL_GPU's normal backend shape. Shared SDL_GPU code
in `src/gpu/SDL_gpu.c` owns public parameter validation, debug-mode checks,
and dispatch through the backend function table declared in
`src/gpu/SDL_sysgpu.h`. The backend in `src/gpu/webgpu/SDL_gpu_webgpu.c`
lowers validated objects and commands to WebGPU, adding provider-specific
preflight on top of shared validation without changing public semantics on its
own.

The backend targets Emscripten builds with SDL as a static library and
consumes the standard `webgpu/webgpu.h` interface through the
Emscripten/Emdawn bindings. It does not talk to the browser directly; every
provider interaction goes through that C API.

## Device And Capabilities

A renderer instance owns the WebGPU instance, adapter, device, queue, and
recorded limits. Device creation queries optional provider features once:
storage-texture access language support, texture format tiers, 16-bit unorm
formats, `depth32float-stencil8`, filterable/blendable 32-bit float,
depth-clip control, and BC/ASTC compression families. It stores the results
as renderer facts. Later format, view, storage-texture, sample-count, and
limit checks read those facts instead of re-querying the provider, so support
answers are stable for the lifetime of the device.

## Asynchronous Provider Bridging

WebGPU is a callback-and-future API: adapter and device requests, buffer
mapping, error-scope resolution, queue completion, and device loss all
complete asynchronously. SDL_GPU's public contract is synchronous, so the
backend registers callbacks in wait-only mode and resolves them with bounded
`wgpuInstanceWaitAny` lists. On the browser this wait path needs an
Emscripten suspension mechanism so the browser event loop can make progress
while the C call stack is waiting; the current validated build uses that
shape rather than exposing asynchronous SDL_GPU calls.

The instance is created requesting timed waits with a bounded wait-list size;
the provider may clamp that request, and the negotiated cap governs how many
fences one wait-any call may take. Because these waits can yield to the
browser event loop, paths where yielding would be incorrect, notably while a
materialized swapchain texture is awaiting submission, check first and fail
with an explanatory error instead of suspending (see
`adaptation-boundaries.md`).

## Handles, Generations, And Internal Views

SDL objects wrap WebGPU objects behind the same opaque handles and common
header structs the native backends use. Textures and buffers additionally
carry generation counters that advance on cycling, so any cached binding or
bind group can detect that it refers to a stale internal allocation rather
than silently reusing a recycled resource.

Provider APIs need concrete texture views at bind, pass, copy, and
presentation boundaries. The backend derives those views from internal view
descriptors: parent identity and generation, format, dimension, aspect,
usage class, and mip/layer range, built from SDL_GPU resources and the
selector fields of the public structs. Texture wrappers keep default
full-resource views for the common cases; passes, binds, copies, and swapchain
presentation build additional views as needed. None of these views are public
SDL handles, which keeps view lifetime an implementation concern instead of an
API contract.

## Submissions, Fences, And Deferred Release

WebGPU queue work outlives the C call that submitted it, so ownership of
everything a command buffer referenced must outlive the command buffer.
Submission finishes the command encoder, hands the commands to the queue, and
moves all retained references: buffers, textures, internal views, samplers,
pipelines, bind groups, pending downloads, and any swapchain texture, from
the command buffer to a per-submission record. That record is released only
after the queue-work-done future reports completion.

Public fences are thin handles that point at a submission. Reference counts
distinguish fence-handle retention from presentation retention, so releasing a
fence, presenting a frame, and queue completion can happen in any order
without freeing a submission early. Waiting on all fences processes
submissions sequentially and has no list-size limit; waiting on any fence uses
one provider wait list and explicitly rejects counts above the negotiated cap
rather than silently truncating the list.

Texture and buffer downloads ride on submissions. Download commands copy into
provider staging buffers; when the submission completes, the staging buffers
are mapped, rows are repacked from the provider's padded row pitch to SDL's
logical pitch in the destination transfer memory, and the staging buffers are
released. If completion tracking cannot be established after a submit, the
backend records device loss rather than guessing at queue state; failing
closed is preferable to unreachable fences.

## Command Encoding

Each SDL command buffer owns one WebGPU command encoder. At any moment it can
have one active render pass, one active compute pass, or a logical copy
section, matching SDL_GPU's pass model. Pass-local state records the active
attachment formats, sample count, and extent, the bound pipeline, vertex and
index bindings, and staged resource bindings.

Pipeline binding validates against the active pass facts before anything is
encoded: mismatched color or depth-stencil formats, sample counts, and
unsupported attachment combinations are rejected as SDL errors instead of
becoming provider validation failures mid-pass. Draws and dispatches then
validate the staged bindings against the pipeline's declared layout facts.

`SDL_BlitGPUTexture()` is implemented as an internal render pass using
backend-owned WGSL shaders and a renderer-owned pipeline cache. The blit path
goes through the same binding convention and validation as user rendering,
which keeps it honest about the backend's own rules.

## Bind Groups

The backend uses a fixed SDL-owned binding convention instead of reflecting
WGSL at runtime. Graphics resources use group 0 for vertex resources and group
2 for fragment resources; graphics uniforms use groups 1 and 3. Compute uses
group 0 for read-only resources, group 1 for read-write resources, and group 2
for uniform data. `shader-resource-bridge.md` documents the per-binding
arithmetic.

Pipelines own the bind group layouts derived from typed SDL resource-layout
facts, plus pre-created empty bind groups: WebGPU pipeline layouts are dense,
so every group index below the highest used one must be populated even when a
stage declares no resources there.

Command buffers stage binding descriptions rather than provider bind groups.
A staged description copies the raw view/sampler/buffer handles and the
immutable facts needed for validation: generation, view dimension, format,
sample count, and usage class, without retaining SDL wrapper objects. Dirty
flags per group record what changed and why; bind groups are rebuilt lazily at
draw or dispatch time only for groups that are actually dirty.

Uniform data is staged into command-buffer-local 64 KiB pages and bound with
dynamic offsets, giving the four upstream uniform slots per stage cheap
per-draw updates. Uniform bind groups are kept separate from resource bind
groups so pushing uniform data never invalidates unrelated sampled or storage
bindings.

## Resource Bind Group Caching

Rebinding the same resources repeatedly is common, so the backend caches
resource bind groups within a command buffer. The cache key covers the group
kind, the bind group layout, and per-entry handle, generation, offset, size,
and usage facts: everything that makes a bind group unequal. The cache is
non-owning: created bind groups are tracked by the command buffer and move to
the submission for lifetime, and cache entries never outlive the command
buffer they were built in.

A renderer-global cache was considered and deliberately deferred. Global
caching requires eviction, reset, cross-thread, and device-loss policies, and
command-buffer-local caching already removes the redundant rebuild churn that
occurs within a frame. Internal counters (creation, hit/miss, and dirty-cause
tallies) exist to measure this; they are diagnostics, not public API, and a
global cache remains a future topic contingent on workloads that demonstrate
enough cross-command-buffer churn to justify the added lifetime complexity.

## Performance Posture

Several paths intentionally use simple lifetimes today: tracked-resource
tables are renderer-owned, command buffers and uniform pages are
command-buffer-local, and provider views and staging buffers are created
around concrete uses. These are implementation tradeoffs, not SDL_GPU API
requirements. Pooling, per-resource reference counters, and shared layout
caches are compatible future optimizations, but each adds reset, eviction, and
device-loss complexity, so they are deferred until representative workloads
show they pay for themselves.

## Errors And Validation

Backend validation is layered:

- Shared SDL_GPU code checks API-level parameters and, in debug mode, asserts
  on contract violations, identically across backends.
- Backend preflight checks WebGPU-specific facts: formats, view shapes,
  usage combinations, provider features, before commands reach the provider,
  reporting SDL errors in SDL vocabulary.
- Provider errors are captured through WebGPU error scopes where waiting is
  safe, and through uncaptured-error and device-lost callbacks otherwise.

Callback-reported errors are recorded on the renderer and drained into SDL
errors at stable entry points: command buffer acquisition, swapchain
acquisition and waits, submission, and fence wait/query paths. This keeps
asynchronous provider failures visible to applications at predictable
call sites instead of being lost in a callback. Device loss is fail-closed: it
is recorded once, losses initiated by teardown are filtered, and there is no
automatic recovery; subsequent GPU entry points report the loss.

Creation-time validation is the most reliable capability gate. Coarse support
queries such as `SDL_GPUTextureSupportsFormat()` and
`SDL_GPUTextureSupportsSampleCount()` are useful preflight tools, but exact
object creation and binding still validate the concrete formats, views, usage
flags, and provider features involved, and remain authoritative.

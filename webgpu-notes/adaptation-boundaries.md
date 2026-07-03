# Adaptation Boundaries

SDL_GPU and WebGPU have a large common subset, but they are not the same API.
This backend implements an SDL_GPU concept when WebGPU can represent it with
the intended semantics, and rejects unsupported combinations at creation,
binding, or encode time with SDL errors. It does not approximate a feature
with different rendering behavior, because a silent approximation on one
provider becomes an undebuggable difference on another.

## Presentation

Browser presentation differs from native swapchains in two load-bearing ways:
the current texture belongs to the frame the browser is composing, and the
runtime must yield to the browser event loop for that frame to appear. The
backend adapts `SDL_AcquireGPUSwapchainTexture()` semantics to that model
instead of exposing it.

An acquired swapchain texture is a command-buffer-owned proxy: a 2D,
single-sample, color-target texture with the configured surface format and
size, but no provider texture behind it yet. The browser's current texture is
materialized lazily when the acquiring command buffer first uses the proxy as
a write destination: render target, resolve target, copy destination, upload
destination, or blit destination. Applications may therefore acquire early,
create resources, and even submit unrelated work between acquire and first
use without holding the browser frame hostage.

Materialization revalidates the window state first. If the surface was
resized, hidden, or reconfigured since acquire, the proxy is invalidated with
an error telling the application to acquire a new swapchain texture; if the
provider reports the surface texture as outdated or lost, the backend
reconfigures and retries once before failing. Acquire itself follows the
upstream contract for degraded cases: on a hidden or zero-sized window, and on
a nonblocking acquire with no free frame slot, it succeeds and returns a NULL
texture rather than raising an error.

Once a browser texture is materialized, the runtime must not yield to the
event loop until the acquiring command buffer is submitted, or the browser
would compose an unfinished frame and invalidate the texture. Operations that
would have to suspend, including fence waits and similar wait paths, are
therefore rejected during this window with an error that says to submit first.
Some validation paths that normally wait on provider error scopes use
unscoped callback error capture instead, so the backend can keep encoding
rules without yielding during the active frame window. There is
no explicit present call: the browser presents after submission when control
returns to the event loop. Frame pacing honors
`SDL_SetGPUAllowedFramesInFlight()` by tracking per-window in-flight
submissions; blocking acquire waits on a slot, and presented submissions are
retained until both queue completion and slot release.

Swapchain textures are write-only destinations in this model. Sampling,
readback, copy-source and blit-source use, and mipmap generation from a
swapchain texture are rejected: the browser current texture's lifetime is
tied to the frame being composed, and read-side usage of surface textures is
not portably configurable across providers.

The current Chrome through Emscripten/Emdawn path supports the `SDR` and
`SDR_LINEAR` compositions with `VSYNC`, using an sRGB view over the surface
format for `SDR_LINEAR`. `IMMEDIATE` and `MAILBOX` present modes and the HDR
compositions fail clearly; broadening them needs provider evidence and an
SDL-visible presentation policy, not a per-backend guess.

## Formats, Internal Views, And Samples

Format support is table-driven and gated by the provider features recorded at
device creation (format tiers, 16-bit unorm formats, compressed families,
`depth32float-stencil8`, filterable 32-bit float). Support queries and
creation validation read the same tables, so an accepted texture and its later
binds agree about what the format can do.

Provider APIs need concrete texture views at bind, pass, copy, and
presentation boundaries, but those views are backend-internal objects derived
from SDL_GPU resources and the write-path selector fields of the public
structs. The public surface keeps SDL_GPU full-resource bind/pass paths;
compatible-format reinterpretation is not part of the current public surface
and would need a separate design and validation case before exposure.

Only sample counts WebGPU can represent are accepted: single-sample and 4x
MSAA. Color MSAA resolves are supported; resolving into a 3D destination uses
a temporary 2D resolve target plus a copy so neighboring depth planes are
preserved. Unsupported sample counts, multisampled storage textures,
depth/stencil resolves, and unsupported internal view shapes fail clearly.

## Depth And Stencil

Depth/stencil behavior is intentionally narrow where WebGPU or providers
expose less than native backends:

- D16 and D32 families are the broadly supported depth formats. `D32S8` is
  gated on the corresponding provider feature.
- D24 and D24S8 are accepted only for single-sample 2D or 2D-array
  depth-stencil target textures, optionally with sampler usage. D24 MSAA,
  cube/cube-array use, and transfer/copy/readback paths are unsupported.
  Upstream SDL already treats D24/D32 availability as backend- and
  device-dependent, so the portable WebGPU claim is kept small.
- Sampling depth requires explicit layout facts: a depth sample type for
  comparison or classic depth sampling, or an unfilterable-float sample type
  with a non-filtering sampler for ordinary floating-point shadow-map reads.
- Combined depth-stencil textures bind their depth aspect when sampled; there
  is no public stencil-aspect selector. Cube and cube-array depth sampling is
  limited to provider-supported D16/D32 full-resource paths.
- Stencil-only render targets are unsupported.

## Storage Textures

Read-only, write-only, and read-write storage textures are available for
defined format/dimension/access combinations, following WebGPU's storage
format vocabulary. Compute read-only slots accept the widest set: the
unorm/snorm, float, and integer storage formats enumerated in the public
header. Read-write access excludes the formats WebGPU cannot declare
read-write, and graphics read-only slots accept a narrower subset
(`rgba8unorm`, `r32uint`, `r32sint`, `r32float`).
WGSL that declares read-only or read-write storage access must include
`requires readonly_and_readwrite_storage_textures;`, and that language
feature is verified at device creation.

Write-path selectors follow the upstream compute pass contract: a 2D binding
selects a mip level, a 2D-array binding selects one layer viewed as 2D, and a
3D binding selects a whole mip level. Texture atomics are not supported and
not emulated: WGSL has no portable storage-texture atomic path, so shaders
using them fail at creation instead of producing an approximation whose
memory semantics SDL could not guarantee. Fragment-stage writable storage,
cube storage textures, multisampled storage textures, and broad
shader-visible descriptor arrays are likewise outside the supported set;
those are better rejected than partially simulated in application-specific
ways.

## Transfers And Compressed Data

WebGPU's transfer rules differ from the native backends in granularity:
buffer copies have 4-byte granularity, buffer-to-texture copies require
256-byte row pitch in staging memory, and buffer mapping is asynchronous. The
backend absorbs most of this:

- SDL transfer buffers are backed by CPU memory with generation tracking, so
  `SDL_MapGPUTransferBuffer()` stays synchronous and cycling stays cheap on a
  provider whose own buffer mapping is asynchronous.
- Uploads copy transfer data into provider staging buffers while encoding,
  padding texture rows to the required pitch internally. Buffer allocations
  carry small alignment slack, so a buffer upload whose size is unaligned but
  ends at the buffer's logical end is padded safely into that slack; other
  unaligned buffer offsets and sizes are rejected with clear errors rather
  than rounded.
- Downloads copy into staging buffers and are repacked to SDL's logical row
  pitch when the submission completes, so applications only ever see the
  sizes they asked for.
- Block-compressed edge mips are submitted with physical block-aligned
  extents while preserving SDL's logical transfer sizing.

The guiding rule is that logical SDL sizes, padded staging allocations, and
provider copy extents are kept as three separate quantities. That prevents
WebGPU alignment requirements from leaking into ordinary SDL transfer calls
while still encoding valid provider copies.

Compressed textures are accepted for sampled and copy use where the provider
advertises the relevant BC/ASTC features, including the sliced-3D variants
where available. Compressed render targets, storage use, MSAA, and generated
mipmaps are unsupported unless a provider gives a clear, portable path.

## Explicit Non-Goals

The backend does not add runtime shader parsing or reflection, arbitrary WGSL
group/binding layouts, renderer-global descriptor array emulation, texture
atomic emulation, or runtime shader-tool dependencies. It also does not add
public retained view handles or read-path subresource selectors on its own
authority: those would define SDL-wide lifetime and identity semantics from a
single backend's convenience, which is backwards. Such concerns belong either
in offline tooling or in a future cross-backend API decision with clear
portability evidence.

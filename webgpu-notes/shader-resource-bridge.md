# Shader Resource Bridge

The WebGPU backend consumes WGSL shaders through `SDL_GPU_SHADERFORMAT_WGSL`;
applications advertise WGSL capability at device creation with
`SDL_PROP_GPU_DEVICE_CREATE_SHADERS_WGSL_BOOLEAN`. SDL core and the backend do
not parse or reflect shader source at runtime. Shader reflection, language
conversion, and layout generation are producer-side offline steps.

## Why Layout Facts Exist

WebGPU requires complete bind group layouts before pipeline creation: each
binding's sampled-texture sample type, view dimension, multisample flag,
sampler binding type, and storage-texture format and access must be declared
up front. SDL_GPU's plain create-info structs only carry resource counts.
Native backends can bridge that gap from bytecode semantics or defer decisions
to bind time; for WGSL the information exists only in shader source.

Considered alternatives include parsing WGSL in SDL to recover the facts,
exposing raw WebGPU group/binding coordinates as public API, and accepting
arbitrary user-defined layouts. The implementation instead uses SDL-owned
typed layout facts and a fixed binding convention, because:

- Runtime parsing would pull a shader-compiler-grade dependency into SDL core,
  and its diagnostics and dialect coverage would become SDL's compatibility
  surface.
- Raw provider binding coordinates would leak a WebGPU mechanism into the
  portable API, where the other backends have no meaningful equivalent.
- Arbitrary layouts would force every backend to consume remapping metadata,
  turning a WebGPU adaptation detail into a cross-backend requirement.

Typed facts keep the runtime independent of shader compilers and reflection
libraries, and let layout validation report SDL concepts instead of raw WGSL
parser or provider diagnostics.

## Binding Convention

Every SDL_GPU shader format has a fixed resource-ordering convention (SPIR-V
descriptor sets, D3D register spaces, Metal argument indices); WGSL is no
different. WGSL shaders use an SDL-owned group and binding convention:

- Vertex stage: group 0 holds sampled texture/sampler pairs, then read-only
  storage textures, then read-only storage buffers; group 1 holds uniform
  buffers at binding N.
- Fragment stage: group 2 and group 3, with the same internal ordering.
- Compute: group 0 holds read-only resources in the same ordering, group 1
  holds read-write resources, and group 2 holds uniform buffers.

Within a resource group, sampled texture slot N uses bindings `2*N` (texture)
and `2*N + 1` (sampler). Read-only storage texture slot N follows at
`2*num_samplers + N`, and read-only storage buffer slot N at
`2*num_samplers + num_storage_textures + N`. In compute group 1, read-write
storage texture slot N uses binding `N` and read-write storage buffer slot N
uses binding `num_readwrite_storage_textures + N`.

Arbitrary WGSL group/binding layouts are intentionally out of scope. A
producer either follows the convention with default layouts, or follows the
convention and supplies explicit SDL layout facts for slots whose types differ
from the defaults. Layout facts describe slot properties under SDL's
convention; they are not raw WebGPU binding coordinates and cannot relocate
slots.

## Defaults And Explicit Facts

When a shader only uses the default resource shapes: 2D filterable-float
sampled textures with filtering samplers, and 2D `rgba8unorm` storage
textures (read-only for read-only slots, write-only for compute read-write
slots), plain `SDL_CreateGPUShader()` and `SDL_CreateGPUComputePipeline()`
work from resource counts alone, exactly like the other shader formats.

Anything else, such as depth or integer sampling, comparison or non-filtering
samplers, samplerless bindings, multisampled sampled textures, non-2D
dimensions, other storage formats, or read/read-write storage access, is a
layout fact and must be declared through
`SDL_CreateGPUShaderWithResourceLayout()` or
`SDL_CreateGPUComputePipelineWithResourceLayout()`.

## Layout Fact Data Model

Resource layouts are SDL-owned typed data. `SDL_GPUShaderResourceLayout`
carries the shader stage, the resource counts, and two optional per-slot
arrays; `SDL_GPUComputePipelineResourceLayout` carries the compute counts and
three optional arrays (sampled, read-only storage, read-write storage). A
NULL array requests defaults for the whole resource class; a non-NULL array
must describe every counted slot. SDL validates and copies the facts during
creation, so caller-owned memory only needs to stay valid until the create
call returns; there is no retained layout object to manage.

Sampled slots declare a texture type, a sample type (filterable float,
unfilterable float, depth, signed/unsigned integer, or the multisampled
unfilterable-float/depth variants), and a sampler type. Storage slots declare
a texture type, format, and access. Some facts imply binding-time contracts
that are enforced on every backend, not just WebGPU: samplerless slots must be
bound with a NULL sampler, non-filtering slots require an all-nearest
non-comparison sampler object, comparison slots require a comparison sampler,
and multisampled slots require samplerless 2D bindings whose backing texture
is actually multisampled.

Internally, shared SDL_GPU code normalizes both creation paths into one typed
facts structure, where plain create-info counts produce default facts, and passes
it to the backend create hook. The WebGPU backend consumes the facts to build
bind group layouts; native backends enforce explicitly supplied facts against
bound resources while leaving the count-only paths as lenient as upstream.

## Producer Workflow

SDL_shadercross provides the reference producer path. It can translate shaders
at runtime for native formats, but the WGSL-plus-layout path is deliberately
offline CLI output. A typical pipeline is HLSL to SPIR-V, then SPIR-V to WGSL
through Tint (Dawn's shader translator), with SDL_shadercross emitting a
generated C sidecar containing a `<prefix>_resource_layout` initializer for
the matching SDL layout struct. The application compiles the sidecar in and
passes the symbol to shader or compute pipeline creation; the runtime never
sees reflection data.

The relevant CLI options are `--resource-layout-c` and
`--resource-layout-symbol-prefix` for sidecar output,
`--resource-layout-sampled-slot` and `--storage-texture-slot` for per-slot
fact overrides where source reflection alone is ambiguous, and
`--suggest-resource-layout-policy` for guidance. Tint is located through
`--tint`, a bundled helper, the `SDL_SHADERCROSS_TINT` environment variable,
or `PATH`; it is an offline tool, not a runtime dependency of SDL or of the
SDL_shadercross library. The validated WGSL path requires the selected Tint to
pass a matrix-order conformance canary, so a translator with divergent matrix
conventions is rejected up front rather than producing transposed rendering.
Sidecars also emit reflected uniform/storage buffer size and member offset
constants as a CPU-side convenience; those constants never enter SDL.

## Source Portability Rules

WGSL is stricter than typical HLSL/GLSL source in ways producers must handle;
these are source and tooling concerns, not backend runtime problems:

- Host-visible layout rules differ: `vec3`-like members have 16-byte
  alignment, so buffer structs need explicit padding discipline.
- WGSL's uniformity analysis rejects implicit derivatives after non-uniform
  control flow; sampling with implicit gradients must happen before
  non-uniform `discard` paths, or use explicit-LOD/gradient sampling.
- Depth sampling and comparison sampling are distinct binding types, and
  ordinary float reads of depth textures require the unfilterable-float
  layout fact with a non-filtering sampler.
- Read-only or read-write storage texture access requires the
  `requires readonly_and_readwrite_storage_textures;` directive.

## Division Of Responsibility

The application or build system owns shader production and layout sidecars;
SDL_shadercross automates that, including Tint validation and layout
generation. SDL owns the typed input structs, validation, object creation,
and command encoding. The backend owns WebGPU lowering and provider error
handling. Each layer can be replaced independently: a project with its own
shader toolchain only needs to emit the same typed facts.

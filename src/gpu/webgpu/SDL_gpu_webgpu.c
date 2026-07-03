/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software in a
     product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#ifdef HAVE_GPU_WEBGPU

#include "../../events/SDL_windowevents_c.h"
#include "../SDL_sysgpu.h"

#include <webgpu/webgpu.h>

#define WINDOW_PROPERTY_DATA                        "SDL.internal.gpu.webgpu.data"
#define WEBGPU_VERTEX_RESOURCE_GROUP                0
#define WEBGPU_VERTEX_UNIFORM_GROUP                 1
#define WEBGPU_FRAGMENT_RESOURCE_GROUP              2
#define WEBGPU_FRAGMENT_UNIFORM_GROUP               3
#define WEBGPU_GRAPHICS_BIND_GROUP_COUNT            4
#define WEBGPU_COMPUTE_READONLY_GROUP               0
#define WEBGPU_COMPUTE_READWRITE_GROUP              1
#define WEBGPU_COMPUTE_UNIFORM_GROUP                2
#define WEBGPU_COMPUTE_BIND_GROUP_COUNT             3
#define WEBGPU_BUFFER_COPY_ALIGNMENT                4
#define WEBGPU_TEXTURE_COPY_BYTES_PER_ROW_ALIGNMENT 256
#define WEBGPU_UNIFORM_BUFFER_PAGE_SIZE             (64 * 1024)
#define WEBGPU_RESOURCE_GENERATION_INITIAL          1
#ifndef WEBGPU_WAIT_ANY_MAX_FENCES
#define WEBGPU_WAIT_ANY_MAX_FENCES                  128
#endif
#define WEBGPU_PROP_WAIT_ANY_MAX_COUNT              "SDL.internal.gpu.webgpu.wait_any.max_count"
#define WEBGPU_RESOURCE_BIND_GROUP_CACHE_INITIAL_CAPACITY 4
#define WEBGPU_RESOURCE_BIND_GROUP_CACHE_MAX_STORAGE_TEXTURES \
    (MAX_STORAGE_TEXTURES_PER_STAGE > MAX_COMPUTE_WRITE_TEXTURES ? MAX_STORAGE_TEXTURES_PER_STAGE : MAX_COMPUTE_WRITE_TEXTURES)
#define WEBGPU_RESOURCE_BIND_GROUP_CACHE_MAX_STORAGE_BUFFERS \
    (MAX_STORAGE_BUFFERS_PER_STAGE > MAX_COMPUTE_WRITE_BUFFERS ? MAX_STORAGE_BUFFERS_PER_STAGE : MAX_COMPUTE_WRITE_BUFFERS)

/*
 * WebGPU backend structure:
 * - SDL exposes the same command-buffer, resource, pass, and fence contracts as
 *   the native GPU backends. This file lowers those contracts to WebGPU without
 *   adding runtime shader parsing or a dependency on shader producer tooling.
 * - Shader resource layout facts are SDL-owned input provided at shader or
 *   compute-pipeline creation time. They drive bind group layouts and validation
 *   for the fixed SDL WGSL convention below.
 * - Swapchain textures are SDL proxy objects. A browser surface may be hidden or
 *   zero-sized at claim time, so actual WebGPU surface configuration and texture
 *   materialization happen lazily around acquire/submit.
 * - Public fences are thin handles. Submitted work, retained resources,
 *   downloads, and presentation slots are owned by per-submission records so
 *   release/cycle/wait semantics remain clear after command buffers are
 *   submitted.
 */

/*
 * Current SDL WebGPU WGSL convention:
 * - graphics resources use groups 0/2, uniforms use groups 1/3.
 * - compute read-only resources use group 0, read-write resources group 1,
 *   uniforms group 2.
 */
static const char WEBGPU_BlitVertexShaderSource[] =
    "struct VSOut {\n"
    "    @builtin(position) position: vec4<f32>,\n"
    "    @location(0) uv: vec2<f32>,\n"
    "};\n"
    "\n"
    "@vertex\n"
    "fn main(@builtin(vertex_index) vertex_index: u32) -> VSOut {\n"
    "    var positions = array<vec2<f32>, 3>(\n"
    "        vec2<f32>(-1.0, -1.0),\n"
    "        vec2<f32>(3.0, -1.0),\n"
    "        vec2<f32>(-1.0, 3.0));\n"
    "    var uvs = array<vec2<f32>, 3>(\n"
    "        vec2<f32>(0.0, 1.0),\n"
    "        vec2<f32>(2.0, 1.0),\n"
    "        vec2<f32>(0.0, -1.0));\n"
    "    var out: VSOut;\n"
    "    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);\n"
    "    out.uv = uvs[vertex_index];\n"
    "    return out;\n"
    "}\n";

static const char WEBGPU_BlitFrom2DShaderSource[] =
    "struct BlitUniforms {\n"
    "    left: f32,\n"
    "    top: f32,\n"
    "    width: f32,\n"
    "    height: f32,\n"
    "    mip_level: u32,\n"
    "    layer_or_depth: f32,\n"
    "};\n"
    "\n"
    "@group(2) @binding(0) var tex0: texture_2d<f32>;\n"
    "@group(2) @binding(1) var samp0: sampler;\n"
    "@group(3) @binding(0) var<uniform> uniforms: BlitUniforms;\n"
    "\n"
    "@fragment\n"
    "fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "    let coord = vec2<f32>(uniforms.left + uv.x * uniforms.width, uniforms.top + uv.y * uniforms.height);\n"
    "    return textureSampleLevel(tex0, samp0, coord, 0.0);\n"
    "}\n";

static const char WEBGPU_BlitFrom3DShaderSource[] =
    "struct BlitUniforms {\n"
    "    left: f32,\n"
    "    top: f32,\n"
    "    width: f32,\n"
    "    height: f32,\n"
    "    mip_level: u32,\n"
    "    layer_or_depth: f32,\n"
    "};\n"
    "\n"
    "@group(2) @binding(0) var tex0: texture_3d<f32>;\n"
    "@group(2) @binding(1) var samp0: sampler;\n"
    "@group(3) @binding(0) var<uniform> uniforms: BlitUniforms;\n"
    "\n"
    "@fragment\n"
    "fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {\n"
    "    let coord = vec3<f32>(uniforms.left + uv.x * uniforms.width, uniforms.top + uv.y * uniforms.height, uniforms.layer_or_depth);\n"
    "    return textureSampleLevel(tex0, samp0, coord, 0.0);\n"
    "}\n";

typedef struct WebGPURenderer WebGPURenderer;
typedef struct WebGPUWindowData WebGPUWindowData;
typedef struct WebGPUResourceBindGroupCacheEntry WebGPUResourceBindGroupCacheEntry;

typedef struct WebGPUAdapterRequest
{
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
} WebGPUAdapterRequest;

typedef struct WebGPUDeviceRequest
{
    WGPURequestDeviceStatus status;
    WGPUDevice device;
} WebGPUDeviceRequest;

typedef struct WebGPUDeviceLostCallbackState
{
    WebGPURenderer *renderer;
    bool renderer_owned;
} WebGPUDeviceLostCallbackState;

typedef struct WebGPUErrorScopeRequest
{
    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    char *message;
} WebGPUErrorScopeRequest;

typedef struct WebGPUBufferMapRequest
{
    WGPUMapAsyncStatus status;
    char *message;
} WebGPUBufferMapRequest;

typedef struct WebGPUTextureDownload WebGPUTextureDownload;
typedef struct WebGPUBufferDownload WebGPUBufferDownload;
typedef struct WebGPUTexture WebGPUTexture;
typedef struct WebGPUBuffer WebGPUBuffer;

static bool WEBGPU_WindowDataIsActive(WebGPUWindowData *window_data);
static void WEBGPU_ReleaseSwapchainTexture(WebGPUTexture *texture);

typedef enum WebGPUSampledTextureSampleKind
{
    WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT,
    WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT,
    WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH,
    WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_SINT,
    WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UINT
} WebGPUSampledTextureSampleKind;

typedef enum WebGPUSamplerBindingKind
{
    WEBGPU_SAMPLER_BINDINGKIND_FILTERING,
    WEBGPU_SAMPLER_BINDINGKIND_NONFILTERING,
    WEBGPU_SAMPLER_BINDINGKIND_COMPARISON,
    WEBGPU_SAMPLER_BINDINGKIND_NONE
} WebGPUSamplerBindingKind;

typedef enum WebGPUStorageTextureAccessKind
{
    WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY,
    WEBGPU_STORAGE_TEXTURE_ACCESSKIND_WRITE_ONLY,
    WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_WRITE
} WebGPUStorageTextureAccessKind;

typedef struct WebGPUSampledTextureLayoutFacts
{
    SDL_GPUTextureType texture_type;
    WebGPUSampledTextureSampleKind sample_type;
    WebGPUSamplerBindingKind sampler_type;
    bool multisampled;
} WebGPUSampledTextureLayoutFacts;

typedef struct WebGPUStorageTextureLayoutFacts
{
    WebGPUStorageTextureAccessKind access_kind;
    SDL_GPUTextureFormat format;
    SDL_GPUTextureType texture_type;
} WebGPUStorageTextureLayoutFacts;

typedef struct WebGPUShaderResourceLayoutFacts
{
    Uint32 sampler_count;
    Uint32 storage_texture_count;
    Uint32 storage_buffer_count;
    Uint32 uniform_buffer_count;
    WebGPUSampledTextureLayoutFacts samplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUStorageTextureLayoutFacts storage_textures[MAX_STORAGE_TEXTURES_PER_STAGE];
} WebGPUShaderResourceLayoutFacts;

typedef struct WebGPUComputeResourceLayoutFacts
{
    Uint32 sampler_count;
    Uint32 readonly_storage_texture_count;
    Uint32 readonly_storage_buffer_count;
    Uint32 readwrite_storage_texture_count;
    Uint32 readwrite_storage_buffer_count;
    Uint32 uniform_buffer_count;
    WebGPUSampledTextureLayoutFacts samplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUStorageTextureLayoutFacts readonly_storage_textures[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUStorageTextureLayoutFacts readwrite_storage_textures[MAX_COMPUTE_WRITE_TEXTURES];
} WebGPUComputeResourceLayoutFacts;

typedef struct WebGPUSampledTextureBindingLayout
{
    WGPUTextureSampleType sample_type;
    WGPUTextureViewDimension view_dimension;
    WGPUSamplerBindingType sampler_type;
    bool has_sampler;
    bool multisampled;
} WebGPUSampledTextureBindingLayout;

typedef struct WebGPUStorageTextureBindingLayout
{
    WGPUStorageTextureAccess access;
    WGPUTextureFormat format;
    WGPUTextureViewDimension view_dimension;
} WebGPUStorageTextureBindingLayout;

typedef struct WebGPUShaderResourceLayout
{
    Uint32 sampler_count;
    Uint32 storage_texture_count;
    Uint32 storage_buffer_count;
    Uint32 uniform_buffer_count;
    WebGPUSampledTextureBindingLayout samplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUStorageTextureBindingLayout storage_textures[MAX_STORAGE_TEXTURES_PER_STAGE];
} WebGPUShaderResourceLayout;

typedef struct WebGPUComputeResourceLayout
{
    Uint32 sampler_count;
    Uint32 readonly_storage_texture_count;
    Uint32 readonly_storage_buffer_count;
    Uint32 readwrite_storage_texture_count;
    Uint32 readwrite_storage_buffer_count;
    Uint32 uniform_buffer_count;
    WebGPUSampledTextureBindingLayout samplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUStorageTextureBindingLayout readonly_storage_textures[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUStorageTextureBindingLayout readwrite_storage_textures[MAX_COMPUTE_WRITE_TEXTURES];
} WebGPUComputeResourceLayout;

typedef enum WebGPUTextureViewUsage
{
    WEBGPU_TEXTURE_VIEW_USAGE_INVALID,
    WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED,
    WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ,
    WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READWRITE,
    WEBGPU_TEXTURE_VIEW_USAGE_COLOR_ATTACHMENT,
    WEBGPU_TEXTURE_VIEW_USAGE_DEPTH_STENCIL_ATTACHMENT,
    WEBGPU_TEXTURE_VIEW_USAGE_RESOLVE_ATTACHMENT,
    WEBGPU_TEXTURE_VIEW_USAGE_SWAPCHAIN_ATTACHMENT,
    WEBGPU_TEXTURE_VIEW_USAGE_BLIT_SOURCE
} WebGPUTextureViewUsage;

typedef enum WebGPUBufferBindingUsage
{
    WEBGPU_BUFFER_BINDING_USAGE_INVALID,
    WEBGPU_BUFFER_BINDING_USAGE_GRAPHICS_STORAGE_READ,
    WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ,
    WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ_WRITE
} WebGPUBufferBindingUsage;

typedef struct WebGPUBufferBindingDescription
{
    WGPUBuffer buffer;
    Uint64 generation;
    Uint64 offset;
    Uint64 size;
    WebGPUBufferBindingUsage usage;
} WebGPUBufferBindingDescription;

/* Queue-completion ownership object. Thin public fence handles point at a
 * submission; retained resources, downloads, and presentation references live
 * here until the submission retires after the WebGPU queue-work-done future
 * completes. */
typedef struct WebGPUSubmission
{
    WebGPURenderer *renderer;
    WGPUFuture future;
    WGPUQueueWorkDoneStatus status;
    WGPUBuffer *buffers;
    Uint32 buffer_count;
    WGPURenderPipeline *graphics_pipelines;
    Uint32 graphics_pipeline_count;
    WGPUComputePipeline *compute_pipelines;
    Uint32 compute_pipeline_count;
    WGPUTexture *textures;
    Uint32 texture_count;
    WGPUTextureView *texture_views;
    Uint32 texture_view_count;
    WebGPUTexture *swapchain_texture;
    WGPUSampler *samplers;
    Uint32 sampler_count;
    WGPUBindGroup *bind_groups;
    Uint32 bind_group_count;
    WebGPUTextureDownload **texture_downloads;
    Uint32 texture_download_count;
    WebGPUBufferDownload **buffer_downloads;
    Uint32 buffer_download_count;
    SDL_AtomicInt completed;
    Uint32 presentation_refcount;
    Uint32 refcount;
    bool handle_retained;
    bool submitted;
    bool completion_tracking_failed;
} WebGPUSubmission;

typedef struct WebGPUFence
{
    WebGPUSubmission *submission;
} WebGPUFence;

typedef struct WebGPUBufferReference
{
    WGPUBuffer buffer;
    Uint32 count;
} WebGPUBufferReference;

typedef struct WebGPUTextureReference
{
    WGPUTexture texture;
    Uint32 count;
} WebGPUTextureReference;

static void WEBGPU_ReleaseTrackedBufferReference(WebGPURenderer *renderer, WGPUBuffer buffer);
static void WEBGPU_ReleaseTrackedTextureReference(WebGPURenderer *renderer, WGPUTexture texture);

typedef struct WebGPUShader
{
    WGPUShaderModule module;
    char *entrypoint;
    SDL_GPUShaderStage stage;
    WebGPUShaderResourceLayout resources;
} WebGPUShader;

typedef struct WebGPUGraphicsPipeline
{
    GraphicsPipelineCommonHeader header;
    WGPURenderPipeline pipeline;
    WGPUBindGroupLayout bind_group_layouts[WEBGPU_GRAPHICS_BIND_GROUP_COUNT];
    WGPUBindGroup empty_bind_groups[WEBGPU_GRAPHICS_BIND_GROUP_COUNT];
    Uint32 bind_group_layout_count;
    Uint64 vertex_buffer_strides[MAX_VERTEX_BUFFERS];
    Uint64 vertex_buffer_last_strides[MAX_VERTEX_BUFFERS];
    Uint32 vertex_buffer_wgpu_slots[MAX_VERTEX_BUFFERS];
    Uint32 required_vertex_buffer_mask;
    Uint32 color_target_count;
    SDL_GPUTextureFormat color_target_formats[MAX_COLOR_TARGET_BINDINGS];
    SDL_GPUTextureFormat depth_stencil_format;
    SDL_GPUSampleCount sample_count;
    WebGPUShaderResourceLayout vertex_resources;
    WebGPUShaderResourceLayout fragment_resources;
    SDL_AtomicInt refcount;
    bool vertex_buffer_instance_step[MAX_VERTEX_BUFFERS];
    bool has_depth_stencil_target;
    bool released;
} WebGPUGraphicsPipeline;

typedef struct WebGPUComputePipeline
{
    ComputePipelineCommonHeader header;
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout bind_group_layouts[WEBGPU_COMPUTE_BIND_GROUP_COUNT];
    WGPUBindGroup empty_bind_groups[WEBGPU_COMPUTE_BIND_GROUP_COUNT];
    Uint32 bind_group_layout_count;
    WebGPUComputeResourceLayout resources;
    SDL_AtomicInt refcount;
    bool released;
} WebGPUComputePipeline;

struct WebGPUTexture
{
    TextureCommonHeader header;
    WGPUTexture texture;
    WGPUTextureView view;
    WGPUTextureView storage_view;
    Uint64 generation;
    Uint32 refcount;
    bool from_surface;
    bool released;
    WebGPUWindowData *window_data;
    Uint64 window_configuration_generation;
};

typedef struct WebGPUSwapchainTextureDestination
{
    WebGPUTexture *texture;
    WebGPUWindowData *window_data;
    bool is_swapchain;
} WebGPUSwapchainTextureDestination;

typedef struct WebGPU3DResolveCopy
{
    WGPUTexture source_2d_texture;
    WGPUTexture destination_3d_texture;
    Uint32 width;
    Uint32 height;
    Uint32 destination_mip_level;
    Uint32 destination_depth_plane;
} WebGPU3DResolveCopy;

typedef struct WebGPURenderAttachmentRegion
{
    WebGPUTexture *texture;
    Uint32 mip_level;
    Uint32 layer_or_depth_plane;
} WebGPURenderAttachmentRegion;

typedef struct WebGPURenderPassAttachmentViews
{
    WGPUTextureView color_views[MAX_COLOR_TARGET_BINDINGS];
    WGPUTextureView resolve_views[MAX_COLOR_TARGET_BINDINGS];
    WGPUTexture resolve_temp_textures[MAX_COLOR_TARGET_BINDINGS];
    WebGPU3DResolveCopy resolve_copies[MAX_COLOR_TARGET_BINDINGS];
    Uint32 resolve_copy_count;
    WGPUTextureView depth_view;
} WebGPURenderPassAttachmentViews;

typedef struct WebGPUSampler
{
    WGPUSampler sampler;
    WGPUSamplerBindingType binding_type;
} WebGPUSampler;

typedef struct WebGPUSampledTextureBindingDescription
{
    /* Binding descriptions copy the raw handles and immutable layout facts.
     * They intentionally do not retain SDL texture wrappers; render-attachment
     * alias checks happen while binding user resources. */
    WGPUTextureView view;
    WGPUSampler sampler;
    Uint64 generation;
    WGPUTextureViewDimension view_dimension;
    SDL_GPUTextureFormat format;
    SDL_GPUSampleCount sample_count;
    WebGPUTextureViewUsage view_usage;
    WGPUSamplerBindingType sampler_binding_type;
} WebGPUSampledTextureBindingDescription;

typedef struct WebGPUStorageTextureBindingDescription
{
    WGPUTextureView view;
    Uint64 generation;
    WGPUTextureViewDimension view_dimension;
    SDL_GPUTextureFormat format;
    SDL_GPUTextureUsageFlags usage;
    WebGPUTextureViewUsage view_usage;
} WebGPUStorageTextureBindingDescription;

struct WebGPUBuffer
{
    BufferCommonHeader header;
    WGPUBuffer buffer;
    Uint64 generation;
    Uint64 allocation_size;
    SDL_GPUBufferUsageFlags usage;
    Uint32 size;
    Uint32 refcount;
    bool released;
};

typedef struct WebGPUTransferBufferGeneration
{
    void *data;
    Uint32 pending_use_count;
} WebGPUTransferBufferGeneration;

typedef struct WebGPUTransferBuffer
{
    WebGPUTransferBufferGeneration **generations;
    char *debugName;
    Uint32 generation_count;
    Uint32 generation_capacity;
    Uint32 active_generation;
    Uint32 size;
    SDL_GPUTransferBufferUsage usage;
    Uint32 pending_use_count;
    bool released;
} WebGPUTransferBuffer;

struct WebGPUTextureDownload
{
    WGPUBuffer staging_buffer;
    WebGPUTransferBuffer *transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    Uint32 destination_offset;
    Uint32 row_count;
    Uint32 depth;
    Uint32 copy_bytes_per_row;
    Uint32 destination_bytes_per_row;
    Uint64 destination_bytes_per_layer;
    Uint32 staging_bytes_per_row;
    Uint64 staging_size;
    bool processed;
    bool succeeded;
};

typedef struct WebGPUBufferDownload
{
    WGPUBuffer staging_buffer;
    WebGPUTransferBuffer *transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    Uint32 destination_offset;
    Uint32 size;
    Uint32 source_offset;
    Uint64 staging_size;
    bool processed;
    bool succeeded;
} WebGPUBufferDownload;

typedef struct WebGPUCommandBuffer
{
    CommandBufferCommonHeader header;
    WebGPURenderer *renderer;
    WGPUCommandEncoder encoder;
    WGPURenderPassEncoder render_pass;
    WGPUComputePassEncoder compute_pass;
    bool copy_pass_active;
    WGPUBuffer *used_buffers;
    Uint32 used_buffer_count;
    Uint32 used_buffer_capacity;
    WGPURenderPipeline *used_graphics_pipelines;
    Uint32 used_graphics_pipeline_count;
    Uint32 used_graphics_pipeline_capacity;
    /* SDL pipeline wrappers stay alive while recording because draw/dispatch
     * still consult SDL-side layout facts and empty bind groups. Raw WGPU
     * pipeline handles move to WebGPUSubmission for queue lifetime. */
    WebGPUGraphicsPipeline **used_graphics_pipeline_wrappers;
    Uint32 used_graphics_pipeline_wrapper_count;
    Uint32 used_graphics_pipeline_wrapper_capacity;
    WGPUComputePipeline *used_compute_pipelines;
    Uint32 used_compute_pipeline_count;
    Uint32 used_compute_pipeline_capacity;
    WebGPUComputePipeline **used_compute_pipeline_wrappers;
    Uint32 used_compute_pipeline_wrapper_count;
    Uint32 used_compute_pipeline_wrapper_capacity;
    WGPUTexture *used_textures;
    Uint32 used_texture_count;
    Uint32 used_texture_capacity;
    WGPUTextureView *used_texture_views;
    Uint32 used_texture_view_count;
    Uint32 used_texture_view_capacity;
    WGPUSampler *used_samplers;
    Uint32 used_sampler_count;
    Uint32 used_sampler_capacity;
    WGPUBindGroup *used_bind_groups;
    Uint32 used_bind_group_count;
    Uint32 used_bind_group_capacity;
    WebGPUResourceBindGroupCacheEntry *resource_bind_group_cache_entries;
    Uint32 resource_bind_group_cache_count;
    Uint32 resource_bind_group_cache_capacity;
    WebGPUTextureDownload **texture_downloads;
    Uint32 texture_download_count;
    Uint32 texture_download_capacity;
    WebGPUBufferDownload **buffer_downloads;
    Uint32 buffer_download_count;
    Uint32 buffer_download_capacity;
    WebGPUGraphicsPipeline *current_graphics_pipeline;
    WebGPUComputePipeline *current_compute_pipeline;
    WebGPUSampledTextureBindingDescription vertex_sampler_bindings[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUSampledTextureBindingDescription fragment_sampler_bindings[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUSampledTextureBindingDescription compute_sampler_bindings[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUStorageTextureBindingDescription vertex_storage_texture_bindings[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUStorageTextureBindingDescription fragment_storage_texture_bindings[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUStorageTextureBindingDescription compute_readonly_storage_texture_bindings[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUBufferBindingDescription vertex_storage_buffer_bindings[MAX_STORAGE_BUFFERS_PER_STAGE];
    WebGPUBufferBindingDescription fragment_storage_buffer_bindings[MAX_STORAGE_BUFFERS_PER_STAGE];
    WGPUBuffer vertex_uniform_buffers[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 vertex_uniform_buffer_offsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 vertex_uniform_buffer_sizes[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroup vertex_uniform_bind_group;
    WGPUBuffer fragment_uniform_buffers[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 fragment_uniform_buffer_offsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 fragment_uniform_buffer_sizes[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroup fragment_uniform_bind_group;
    WGPUBuffer compute_uniform_buffers[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 compute_uniform_buffer_offsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    Uint64 compute_uniform_buffer_sizes[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroup compute_uniform_bind_group;
    WebGPUBufferBindingDescription compute_readonly_storage_buffer_bindings[MAX_STORAGE_BUFFERS_PER_STAGE];
    WebGPUStorageTextureBindingDescription compute_readwrite_storage_texture_bindings[MAX_COMPUTE_WRITE_TEXTURES];
    WebGPUBufferBindingDescription compute_readwrite_storage_buffer_bindings[MAX_COMPUTE_WRITE_BUFFERS];
    WGPUBuffer uniform_buffer_page;
    Uint64 uniform_buffer_page_offset;
    Uint64 uniform_buffer_page_size;
    Uint32 stencil_reference;
    WGPUBuffer vertex_buffers[MAX_VERTEX_BUFFERS];
    Uint64 vertex_buffer_offsets[MAX_VERTEX_BUFFERS];
    Uint64 vertex_buffer_sizes[MAX_VERTEX_BUFFERS];
    Uint64 index_buffer_size;
    Uint32 bound_vertex_buffer_mask;
    Uint32 index_element_size;
    Uint32 render_pass_color_target_count;
    WebGPUTexture *render_pass_color_targets[MAX_COLOR_TARGET_BINDINGS];
    WebGPUTexture *render_pass_resolve_targets[MAX_COLOR_TARGET_BINDINGS];
    WebGPUTexture *render_pass_depth_stencil_target;
    WebGPU3DResolveCopy render_pass_3d_resolve_copies[MAX_COLOR_TARGET_BINDINGS];
    Uint32 render_pass_3d_resolve_copy_count;
    SDL_GPUTextureFormat render_pass_color_target_formats[MAX_COLOR_TARGET_BINDINGS];
    SDL_GPUTextureFormat render_pass_depth_stencil_format;
    SDL_GPUSampleCount render_pass_sample_count;
    Uint32 render_pass_attachment_width;
    Uint32 render_pass_attachment_height;
    bool render_pass_has_depth_stencil_target;
    bool vertex_resource_bind_group_dirty;
    bool fragment_resource_bind_group_dirty;
    bool vertex_uniform_bind_group_dirty;
    bool fragment_uniform_bind_group_dirty;
    bool vertex_uniform_bind_group_offsets_dirty;
    bool fragment_uniform_bind_group_offsets_dirty;
    bool compute_readonly_bind_group_dirty;
    bool compute_readwrite_bind_group_dirty;
    bool compute_uniform_bind_group_dirty;
    bool compute_uniform_bind_group_offsets_dirty;
    bool index_buffer_bound;
    WebGPUTexture *swapchain_texture;
    bool swapchain_texture_pending_submit;
    bool failed;
} WebGPUCommandBuffer;

struct WebGPUWindowData
{
    WebGPURenderer *renderer;
    SDL_Window *window;
    char *canvas_selector;
    WGPUSurface surface;
    WGPUTextureFormat surface_format;
    WGPUTextureUsage configured_usage;
    SDL_GPUTextureFormat sdl_format;
    Uint64 configuration_generation;
    Uint32 width;
    Uint32 height;
    bool configured;
    bool needs_configure;
    Uint32 refcount;
    Uint32 swapchain_proxy_refcount;
    SDL_GPUPresentMode present_mode;
    SDL_GPUSwapchainComposition swapchain_composition;
    WebGPUSubmission *in_flight_submissions[MAX_FRAMES_IN_FLIGHT];
    Uint32 frame_counter;
};

typedef enum WebGPUBindGroupInstrumentationPath
{
    WEBGPU_BIND_GROUP_PATH_GRAPHICS_VERTEX_RESOURCE,
    WEBGPU_BIND_GROUP_PATH_GRAPHICS_VERTEX_UNIFORM,
    WEBGPU_BIND_GROUP_PATH_GRAPHICS_FRAGMENT_RESOURCE,
    WEBGPU_BIND_GROUP_PATH_GRAPHICS_FRAGMENT_UNIFORM,
    WEBGPU_BIND_GROUP_PATH_COMPUTE_READONLY,
    WEBGPU_BIND_GROUP_PATH_COMPUTE_READWRITE,
    WEBGPU_BIND_GROUP_PATH_COMPUTE_UNIFORM,
    WEBGPU_BIND_GROUP_PATH_GRAPHICS_EMPTY,
    WEBGPU_BIND_GROUP_PATH_COMPUTE_EMPTY,
    WEBGPU_BIND_GROUP_PATH_COUNT
} WebGPUBindGroupInstrumentationPath;

typedef enum WebGPUBindGroupDirtyCause
{
    WEBGPU_BIND_GROUP_DIRTY_PASS,
    WEBGPU_BIND_GROUP_DIRTY_PIPELINE,
    WEBGPU_BIND_GROUP_DIRTY_SAMPLED_RESOURCE,
    WEBGPU_BIND_GROUP_DIRTY_STORAGE_RESOURCE,
    WEBGPU_BIND_GROUP_DIRTY_UNIFORM,
    WEBGPU_BIND_GROUP_DIRTY_COUNT
} WebGPUBindGroupDirtyCause;

typedef struct WebGPUBindGroupInstrumentation
{
    SDL_AtomicU32 create_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 set_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 clean_apply_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 cache_hit_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 cache_miss_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 cache_insert_fail_count[WEBGPU_BIND_GROUP_PATH_COUNT];
    SDL_AtomicU32 dirty_count[WEBGPU_BIND_GROUP_DIRTY_COUNT];
    SDL_AtomicU32 create_total;
    SDL_AtomicU32 set_total;
    SDL_AtomicU32 clean_apply_total;
    SDL_AtomicU32 cache_hit_total;
    SDL_AtomicU32 cache_miss_total;
    SDL_AtomicU32 cache_insert_fail_total;
    SDL_AtomicU32 dirty_total;
    SDL_AtomicU32 peak_command_buffer_bind_groups;
    SDL_AtomicU32 peak_command_buffer_resource_cache_entries;
    SDL_AtomicU32 peak_command_buffer_resource_cache_capacity;
} WebGPUBindGroupInstrumentation;

typedef struct WebGPUResourceBindGroupSampledTextureKey
{
    WGPUTextureView view;
    WGPUSampler sampler;
    Uint64 generation;
    WebGPUTextureViewUsage view_usage;
} WebGPUResourceBindGroupSampledTextureKey;

typedef struct WebGPUResourceBindGroupStorageTextureKey
{
    WGPUTextureView view;
    Uint64 generation;
    WebGPUTextureViewUsage view_usage;
} WebGPUResourceBindGroupStorageTextureKey;

typedef struct WebGPUResourceBindGroupBufferKey
{
    WGPUBuffer buffer;
    Uint64 generation;
    Uint64 offset;
    Uint64 size;
    WebGPUBufferBindingUsage usage;
} WebGPUResourceBindGroupBufferKey;

typedef struct WebGPUResourceBindGroupCacheKey
{
    WebGPUBindGroupInstrumentationPath path;
    WGPUBindGroupLayout layout;
    Uint32 entry_count;
    Uint32 sampled_texture_count;
    Uint32 storage_texture_count;
    Uint32 storage_buffer_count;
    WebGPUResourceBindGroupSampledTextureKey sampled_textures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUResourceBindGroupStorageTextureKey storage_textures[WEBGPU_RESOURCE_BIND_GROUP_CACHE_MAX_STORAGE_TEXTURES];
    WebGPUResourceBindGroupBufferKey storage_buffers[WEBGPU_RESOURCE_BIND_GROUP_CACHE_MAX_STORAGE_BUFFERS];
} WebGPUResourceBindGroupCacheKey;

struct WebGPUResourceBindGroupCacheEntry
{
    WebGPUResourceBindGroupCacheKey key;
    WGPUBindGroup bind_group;
};

typedef struct WebGPUResourceGenerationInstrumentation
{
    SDL_AtomicU32 texture_cycle_count;
    SDL_AtomicU32 buffer_cycle_count;
} WebGPUResourceGenerationInstrumentation;

struct WebGPURenderer
{
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPULimits limits;
    SDL_PropertiesID props;
    SDL_Mutex *fence_lock;
    WebGPUSubmission **pending_submissions;
    Uint32 pending_submission_count;
    Uint32 pending_submission_capacity;
    SDL_Mutex *resource_lock;
    WebGPUBufferReference *buffer_references;
    Uint32 buffer_reference_count;
    Uint32 buffer_reference_capacity;
    WebGPUTextureReference *texture_references;
    Uint32 texture_reference_count;
    Uint32 texture_reference_capacity;
    SDL_Mutex *error_lock;
    WGPUErrorType last_uncaptured_error_type;
    char *last_uncaptured_error_message;
    Uint32 uncaptured_error_count;
    WGPUFuture device_lost_future;
    bool device_lost_future_pending;
    bool device_ready;
    bool device_shutting_down;
    bool device_lost;
    bool pending_device_loss;
    WGPUDeviceLostReason device_lost_reason;
    char *device_lost_message;
    WebGPUDeviceLostCallbackState *device_lost_callback_state;
    WebGPUWindowData **claimed_windows;
    Uint32 claimed_window_count;
    Uint32 claimed_window_capacity;
    SDL_AtomicU32 swapchain_textures_pending_submit;
    Uint32 allowed_frames_in_flight;
    SDL_GPUSampler *blit_linear_sampler;
    SDL_GPUSampler *blit_nearest_sampler;
    SDL_GPUShader *blit_vertex_shader;
    SDL_GPUShader *blit_from_2d_shader;
    SDL_GPUShader *blit_from_3d_shader;
    BlitPipelineCacheEntry *blit_pipelines;
    Uint32 blit_pipeline_count;
    Uint32 blit_pipeline_capacity;
    WebGPUBindGroupInstrumentation bind_group_instrumentation;
    WebGPUResourceGenerationInstrumentation resource_generation_instrumentation;
    size_t wait_any_max_count;
    bool supports_readonly_and_readwrite_storage_textures;
    bool supports_rg11b10ufloat_renderable;
    bool supports_texture_formats_tier1;
    bool supports_texture_formats_tier2;
    bool supports_unorm16_texture_formats;
    bool supports_core_features_and_limits;
    bool supports_depth32float_stencil8;
    bool supports_float32_filterable;
    bool supports_float32_blendable;
    bool supports_depth_clip_control;
    bool supports_texture_compression_bc;
    bool supports_texture_compression_astc;
    bool supports_texture_compression_bc_sliced_3d;
    bool supports_texture_compression_astc_sliced_3d;
};

static WGPUStringView WEBGPU_StringView(const char *string)
{
    WGPUStringView result = WGPU_STRING_VIEW_INIT;
    result.data = string;
    result.length = string ? SDL_strlen(string) : 0;
    return result;
}

static WGPUStringView WEBGPU_ByteView(const Uint8 *data, size_t length)
{
    WGPUStringView result = WGPU_STRING_VIEW_INIT;
    result.data = (const char *)data;
    result.length = length;
    return result;
}

static bool WEBGPU_HasSwapchainSubmitWindow(WebGPURenderer *renderer)
{
    return renderer && SDL_GetAtomicU32(&renderer->swapchain_textures_pending_submit) != 0;
}

/*
 * Some WebGPU error scopes and waits can yield through the browser runtime.
 * Once a browser swapchain texture is materialized, yielding before submit can
 * invalidate presentation work, so wait-dependent paths fail or use explicit
 * unscoped fallbacks until the submitting command buffer closes the window.
 */
static bool WEBGPU_CanWaitForErrorScope(WebGPURenderer *renderer, const char *context)
{
    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return SDL_SetError("Cannot perform WebGPU operation %s while a swapchain texture is awaiting submission; submit the command buffer that uses the swapchain texture first", context);
    }
    return true;
}

static void WEBGPU_BeginSwapchainSubmitWindow(WebGPUCommandBuffer *command_buffer)
{
    if (!command_buffer || command_buffer->swapchain_texture_pending_submit) {
        return;
    }

    command_buffer->swapchain_texture_pending_submit = true;
    SDL_AddAtomicU32(&command_buffer->renderer->swapchain_textures_pending_submit, 1);
}

static void WEBGPU_EndSwapchainSubmitWindow(WebGPUCommandBuffer *command_buffer)
{
    Uint32 previous;

    if (!command_buffer || !command_buffer->swapchain_texture_pending_submit) {
        return;
    }

    previous = SDL_AddAtomicU32(&command_buffer->renderer->swapchain_textures_pending_submit, -1);
    SDL_assert(previous > 0);
    command_buffer->swapchain_texture_pending_submit = false;
}

static const char *const WEBGPU_BIND_GROUP_CREATE_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.create.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.create.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.create.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.create.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.create.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.create.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.create.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.create.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.create.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_SET_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.set.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.set.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.set.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.set.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.set.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.set.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.set.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.set.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.set.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_CLEAN_APPLY_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.clean_apply.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_CACHE_HIT_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.cache_hit.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_CACHE_MISS_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.cache_miss.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_CACHE_INSERT_FAIL_PROPERTY_NAMES[WEBGPU_BIND_GROUP_PATH_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.graphics.vertex_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.graphics.vertex_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.graphics.fragment_resource",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.graphics.fragment_uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.compute.readonly",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.compute.readwrite",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.compute.uniform",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.graphics.empty",
    "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.compute.empty"
};

static const char *const WEBGPU_BIND_GROUP_DIRTY_PROPERTY_NAMES[WEBGPU_BIND_GROUP_DIRTY_COUNT] = {
    "SDL.internal.gpu.webgpu.bind_group.dirty.pass",
    "SDL.internal.gpu.webgpu.bind_group.dirty.pipeline",
    "SDL.internal.gpu.webgpu.bind_group.dirty.sampled_resource",
    "SDL.internal.gpu.webgpu.bind_group.dirty.storage_resource",
    "SDL.internal.gpu.webgpu.bind_group.dirty.uniform"
};

static void WEBGPU_SetCounterProperty(SDL_PropertiesID props, const char *name, Uint64 value)
{
    SDL_SetNumberProperty(props, name, (Sint64)value);
}

static void WEBGPU_IncrementInstrumentationCounter(SDL_AtomicU32 *counter)
{
    SDL_AddAtomicU32(counter, 1);
}

static Uint64 WEBGPU_GetInstrumentationCounter(SDL_AtomicU32 *counter)
{
    return (Uint64)SDL_GetAtomicU32(counter);
}

static Uint64 WEBGPU_NextResourceGeneration(Uint64 generation)
{
    if (generation == 0 || generation == SDL_MAX_UINT64) {
        return WEBGPU_RESOURCE_GENERATION_INITIAL;
    }
    return generation + 1;
}

static void WEBGPU_RecordBindGroupCreate(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->create_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->create_total);
}

static void WEBGPU_RecordBindGroupSet(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->set_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->set_total);
}

static void WEBGPU_RecordBindGroupCleanApply(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->clean_apply_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->clean_apply_total);
}

static void WEBGPU_RecordBindGroupCacheHit(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_hit_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_hit_total);
}

static void WEBGPU_RecordBindGroupCacheMiss(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_miss_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_miss_total);
}

static void WEBGPU_RecordBindGroupCacheInsertFail(WebGPURenderer *renderer, WebGPUBindGroupInstrumentationPath path)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_insert_fail_count[path]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->cache_insert_fail_total);
}

static void WEBGPU_RecordBindGroupDirty(WebGPURenderer *renderer, WebGPUBindGroupDirtyCause cause)
{
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    WEBGPU_IncrementInstrumentationCounter(&instrumentation->dirty_count[cause]);
    WEBGPU_IncrementInstrumentationCounter(&instrumentation->dirty_total);
}

static void WEBGPU_RecordTextureGenerationCycle(WebGPURenderer *renderer)
{
    WEBGPU_IncrementInstrumentationCounter(&renderer->resource_generation_instrumentation.texture_cycle_count);
}

static void WEBGPU_RecordBufferGenerationCycle(WebGPURenderer *renderer)
{
    WEBGPU_IncrementInstrumentationCounter(&renderer->resource_generation_instrumentation.buffer_cycle_count);
}

static void WEBGPU_RecordInstrumentationPeak(SDL_AtomicU32 *peak, Uint32 new_peak)
{
    Uint32 old_peak = SDL_GetAtomicU32(peak);

    while (new_peak > old_peak) {
        if (SDL_CompareAndSwapAtomicU32(peak, old_peak, new_peak)) {
            break;
        }
        old_peak = SDL_GetAtomicU32(peak);
    }
}

static void WEBGPU_RecordCommandBufferBindGroupPeak(WebGPUCommandBuffer *command_buffer)
{
    WEBGPU_RecordInstrumentationPeak(
        &command_buffer->renderer->bind_group_instrumentation.peak_command_buffer_bind_groups,
        command_buffer->used_bind_group_count);
}

static void WEBGPU_RecordResourceBindGroupCachePeak(WebGPUCommandBuffer *command_buffer)
{
    WebGPUBindGroupInstrumentation *instrumentation = &command_buffer->renderer->bind_group_instrumentation;

    WEBGPU_RecordInstrumentationPeak(
        &instrumentation->peak_command_buffer_resource_cache_entries,
        command_buffer->resource_bind_group_cache_count);
    WEBGPU_RecordInstrumentationPeak(
        &instrumentation->peak_command_buffer_resource_cache_capacity,
        command_buffer->resource_bind_group_cache_capacity);
}

static void WEBGPU_SyncBindGroupInstrumentationProperties(WebGPURenderer *renderer)
{
    SDL_PropertiesID props = renderer->props;
    WebGPUBindGroupInstrumentation *instrumentation = &renderer->bind_group_instrumentation;

    if (!props) {
        return;
    }

    for (Uint32 i = 0; i < WEBGPU_BIND_GROUP_PATH_COUNT; i += 1) {
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_CREATE_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->create_count[i]));
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_SET_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->set_count[i]));
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_CLEAN_APPLY_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->clean_apply_count[i]));
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_CACHE_HIT_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->cache_hit_count[i]));
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_CACHE_MISS_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->cache_miss_count[i]));
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_CACHE_INSERT_FAIL_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->cache_insert_fail_count[i]));
    }
    for (Uint32 i = 0; i < WEBGPU_BIND_GROUP_DIRTY_COUNT; i += 1) {
        WEBGPU_SetCounterProperty(
            props,
            WEBGPU_BIND_GROUP_DIRTY_PROPERTY_NAMES[i],
            WEBGPU_GetInstrumentationCounter(&instrumentation->dirty_count[i]));
    }

    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.create.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->create_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.set.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->set_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.clean_apply.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->clean_apply_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.cache_hit.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->cache_hit_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.cache_miss.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->cache_miss_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.cache_insert_fail.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->cache_insert_fail_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.dirty.total",
        WEBGPU_GetInstrumentationCounter(&instrumentation->dirty_total));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.peak.command_buffer_bind_groups",
        WEBGPU_GetInstrumentationCounter(&instrumentation->peak_command_buffer_bind_groups));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.peak.command_buffer_resource_cache_entries",
        WEBGPU_GetInstrumentationCounter(&instrumentation->peak_command_buffer_resource_cache_entries));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.bind_group.peak.command_buffer_resource_cache_capacity",
        WEBGPU_GetInstrumentationCounter(&instrumentation->peak_command_buffer_resource_cache_capacity));
}

static void WEBGPU_SyncResourceGenerationInstrumentationProperties(WebGPURenderer *renderer)
{
    SDL_PropertiesID props = renderer->props;
    WebGPUResourceGenerationInstrumentation *instrumentation = &renderer->resource_generation_instrumentation;

    if (!props) {
        return;
    }

    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.resource_generation.texture_cycle",
        WEBGPU_GetInstrumentationCounter(&instrumentation->texture_cycle_count));
    WEBGPU_SetCounterProperty(
        props,
        "SDL.internal.gpu.webgpu.resource_generation.buffer_cycle",
        WEBGPU_GetInstrumentationCounter(&instrumentation->buffer_cycle_count));
}

static void WEBGPU_InitResourceBindGroupCacheKey(
    WebGPUResourceBindGroupCacheKey *key,
    WebGPUBindGroupInstrumentationPath path,
    WGPUBindGroupLayout layout,
    Uint32 entry_count,
    Uint32 sampled_texture_count,
    Uint32 storage_texture_count,
    Uint32 storage_buffer_count)
{
    SDL_zero(*key);
    key->path = path;
    key->layout = layout;
    key->entry_count = entry_count;
    key->sampled_texture_count = sampled_texture_count;
    key->storage_texture_count = storage_texture_count;
    key->storage_buffer_count = storage_buffer_count;
}

static void WEBGPU_SetResourceBindGroupCacheSampledTexture(
    WebGPUResourceBindGroupCacheKey *key,
    Uint32 index,
    const WebGPUSampledTextureBindingDescription *binding)
{
    key->sampled_textures[index].view = binding->view;
    key->sampled_textures[index].sampler = binding->sampler;
    key->sampled_textures[index].generation = binding->generation;
    key->sampled_textures[index].view_usage = binding->view_usage;
}

static void WEBGPU_SetResourceBindGroupCacheStorageTexture(
    WebGPUResourceBindGroupCacheKey *key,
    Uint32 index,
    const WebGPUStorageTextureBindingDescription *binding)
{
    key->storage_textures[index].view = binding->view;
    key->storage_textures[index].generation = binding->generation;
    key->storage_textures[index].view_usage = binding->view_usage;
}

static void WEBGPU_SetResourceBindGroupCacheStorageBuffer(
    WebGPUResourceBindGroupCacheKey *key,
    Uint32 index,
    const WebGPUBufferBindingDescription *binding)
{
    key->storage_buffers[index].buffer = binding->buffer;
    key->storage_buffers[index].generation = binding->generation;
    key->storage_buffers[index].offset = binding->offset;
    key->storage_buffers[index].size = binding->size;
    key->storage_buffers[index].usage = binding->usage;
}

static void WEBGPU_InitResourceBindGroupCacheKeyFromBindings(
    WebGPUResourceBindGroupCacheKey *key,
    WebGPUBindGroupInstrumentationPath path,
    WGPUBindGroupLayout layout,
    Uint32 entry_count,
    const WebGPUSampledTextureBindingDescription *sampled_texture_bindings,
    Uint32 sampled_texture_count,
    const WebGPUStorageTextureBindingDescription *storage_texture_bindings,
    Uint32 storage_texture_count,
    const WebGPUBufferBindingDescription *storage_buffer_bindings,
    Uint32 storage_buffer_count)
{
    WEBGPU_InitResourceBindGroupCacheKey(
        key,
        path,
        layout,
        entry_count,
        sampled_texture_count,
        storage_texture_count,
        storage_buffer_count);

    for (Uint32 i = 0; i < sampled_texture_count; i += 1) {
        WEBGPU_SetResourceBindGroupCacheSampledTexture(key, i, &sampled_texture_bindings[i]);
    }
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        WEBGPU_SetResourceBindGroupCacheStorageTexture(key, i, &storage_texture_bindings[i]);
    }
    for (Uint32 i = 0; i < storage_buffer_count; i += 1) {
        WEBGPU_SetResourceBindGroupCacheStorageBuffer(key, i, &storage_buffer_bindings[i]);
    }
}

static bool WEBGPU_ResourceBindGroupSampledTextureKeysMatch(
    const WebGPUResourceBindGroupSampledTextureKey *a,
    const WebGPUResourceBindGroupSampledTextureKey *b)
{
    return a->view == b->view &&
        a->sampler == b->sampler &&
        a->generation == b->generation &&
        a->view_usage == b->view_usage;
}

static bool WEBGPU_ResourceBindGroupStorageTextureKeysMatch(
    const WebGPUResourceBindGroupStorageTextureKey *a,
    const WebGPUResourceBindGroupStorageTextureKey *b)
{
    return a->view == b->view &&
        a->generation == b->generation &&
        a->view_usage == b->view_usage;
}

static bool WEBGPU_ResourceBindGroupBufferKeysMatch(
    const WebGPUResourceBindGroupBufferKey *a,
    const WebGPUResourceBindGroupBufferKey *b)
{
    return a->buffer == b->buffer &&
        a->generation == b->generation &&
        a->offset == b->offset &&
        a->size == b->size &&
        a->usage == b->usage;
}

static bool WEBGPU_ResourceBindGroupCacheKeysMatch(
    const WebGPUResourceBindGroupCacheKey *a,
    const WebGPUResourceBindGroupCacheKey *b)
{
    if (a->path != b->path ||
        a->layout != b->layout ||
        a->entry_count != b->entry_count ||
        a->sampled_texture_count != b->sampled_texture_count ||
        a->storage_texture_count != b->storage_texture_count ||
        a->storage_buffer_count != b->storage_buffer_count) {
        return false;
    }

    for (Uint32 i = 0; i < a->sampled_texture_count; i += 1) {
        if (!WEBGPU_ResourceBindGroupSampledTextureKeysMatch(&a->sampled_textures[i], &b->sampled_textures[i])) {
            return false;
        }
    }
    for (Uint32 i = 0; i < a->storage_texture_count; i += 1) {
        if (!WEBGPU_ResourceBindGroupStorageTextureKeysMatch(&a->storage_textures[i], &b->storage_textures[i])) {
            return false;
        }
    }
    for (Uint32 i = 0; i < a->storage_buffer_count; i += 1) {
        if (!WEBGPU_ResourceBindGroupBufferKeysMatch(&a->storage_buffers[i], &b->storage_buffers[i])) {
            return false;
        }
    }
    return true;
}

static WGPUBindGroup WEBGPU_FindResourceBindGroupCacheEntry(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUResourceBindGroupCacheKey *key)
{
    for (Uint32 i = 0; i < command_buffer->resource_bind_group_cache_count; i += 1) {
        if (WEBGPU_ResourceBindGroupCacheKeysMatch(&command_buffer->resource_bind_group_cache_entries[i].key, key)) {
            WEBGPU_RecordBindGroupCacheHit(command_buffer->renderer, key->path);
            return command_buffer->resource_bind_group_cache_entries[i].bind_group;
        }
    }

    WEBGPU_RecordBindGroupCacheMiss(command_buffer->renderer, key->path);
    return NULL;
}

static void WEBGPU_InsertResourceBindGroupCacheEntry(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUResourceBindGroupCacheKey *key,
    WGPUBindGroup bind_group)
{
    WebGPUResourceBindGroupCacheEntry *entries;

    if (command_buffer->resource_bind_group_cache_count >= command_buffer->resource_bind_group_cache_capacity) {
        Uint32 new_capacity = command_buffer->resource_bind_group_cache_capacity
            ? command_buffer->resource_bind_group_cache_capacity * 2
            : WEBGPU_RESOURCE_BIND_GROUP_CACHE_INITIAL_CAPACITY;
        entries = (WebGPUResourceBindGroupCacheEntry *)SDL_realloc(
            command_buffer->resource_bind_group_cache_entries,
            new_capacity * sizeof(*entries));
        if (!entries) {
            WEBGPU_RecordBindGroupCacheInsertFail(command_buffer->renderer, key->path);
            return;
        }
        command_buffer->resource_bind_group_cache_entries = entries;
        command_buffer->resource_bind_group_cache_capacity = new_capacity;
    }

    command_buffer->resource_bind_group_cache_entries[command_buffer->resource_bind_group_cache_count].key = *key;
    command_buffer->resource_bind_group_cache_entries[command_buffer->resource_bind_group_cache_count].bind_group = bind_group;
    command_buffer->resource_bind_group_cache_count += 1;
    WEBGPU_RecordResourceBindGroupCachePeak(command_buffer);
}

static void WEBGPU_ClearResourceBindGroupCache(WebGPUCommandBuffer *command_buffer)
{
    SDL_free(command_buffer->resource_bind_group_cache_entries);
    command_buffer->resource_bind_group_cache_entries = NULL;
    command_buffer->resource_bind_group_cache_count = 0;
    command_buffer->resource_bind_group_cache_capacity = 0;
}

static void WEBGPU_SetStringError(const char *message)
{
    SDL_SetError("WebGPU backend: %s", message);
}

static void WEBGPU_FailCommandBuffer(WebGPUCommandBuffer *command_buffer, const char *message)
{
    WEBGPU_SetStringError(message);
    if (command_buffer) {
        command_buffer->failed = true;
    }
}

static bool WEBGPU_FailIfRenderOrComputePassActive(WebGPUCommandBuffer *command_buffer, const char *operation)
{
    if (command_buffer->render_pass) {
        SDL_SetError("WebGPU backend: %s cannot be called during a render pass", operation);
        command_buffer->failed = true;
        return true;
    }
    if (command_buffer->compute_pass) {
        SDL_SetError("WebGPU backend: %s cannot be called during a compute pass", operation);
        command_buffer->failed = true;
        return true;
    }
    return false;
}

static bool WEBGPU_FailIfAnyPassActive(WebGPUCommandBuffer *command_buffer, const char *operation)
{
    if (WEBGPU_FailIfRenderOrComputePassActive(command_buffer, operation)) {
        return true;
    }
    if (command_buffer->copy_pass_active) {
        SDL_SetError("WebGPU backend: %s cannot be called during a copy pass", operation);
        command_buffer->failed = true;
        return true;
    }
    return false;
}

static bool WEBGPU_FailIfCopyPassInactive(WebGPUCommandBuffer *command_buffer, const char *operation)
{
    if (!command_buffer->copy_pass_active) {
        SDL_SetError("WebGPU backend: %s requires an active copy pass", operation);
        command_buffer->failed = true;
        return true;
    }
    return false;
}

static bool WEBGPU_FailIfRenderPassInactive(WebGPUCommandBuffer *command_buffer, const char *operation)
{
    if (!command_buffer->render_pass) {
        SDL_SetError("WebGPU backend: %s requires an active render pass", operation);
        command_buffer->failed = true;
        return true;
    }
    return false;
}

static bool WEBGPU_FailIfComputePassInactive(WebGPUCommandBuffer *command_buffer, const char *operation)
{
    if (!command_buffer->compute_pass) {
        SDL_SetError("WebGPU backend: %s requires an active compute pass", operation);
        command_buffer->failed = true;
        return true;
    }
    return false;
}

static bool WEBGPU_ResolveSwapchainTextureDestination(
    WebGPUCommandBuffer *command_buffer,
    WebGPUTexture *texture,
    WGPUTextureUsage required_usage,
    const char *acquire_error,
    const char *usage_error,
    WebGPUSwapchainTextureDestination *destination)
{
    WebGPUWindowData *window_data;

    SDL_zero(*destination);
    destination->texture = texture;

    if (!texture || !texture->from_surface) {
        return true;
    }
    destination->is_swapchain = true;
    if (texture != command_buffer->swapchain_texture) {
        WEBGPU_FailCommandBuffer(command_buffer, acquire_error);
        return false;
    }

    window_data = texture->window_data;
    if (!WEBGPU_WindowDataIsActive(window_data) ||
        window_data->renderer != command_buffer->renderer) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain destination texture has no active WebGPU window");
        return false;
    }
    destination->window_data = window_data;
    if (required_usage != WGPUTextureUsage_None &&
        !(window_data->configured_usage & required_usage)) {
        WEBGPU_FailCommandBuffer(command_buffer, usage_error);
        return false;
    }
    return true;
}

static bool WEBGPU_ValidateSwapchainTextureDestination(
    WebGPUCommandBuffer *command_buffer,
    WebGPUTexture *texture,
    WGPUTextureUsage required_usage,
    const char *acquire_error,
    const char *usage_error)
{
    WebGPUSwapchainTextureDestination destination;

    return WEBGPU_ResolveSwapchainTextureDestination(
        command_buffer,
        texture,
        required_usage,
        acquire_error,
        usage_error,
        &destination);
}

static void *WEBGPU_UnsupportedPointer(const char *function_name)
{
    SDL_SetError("WebGPU backend does not support %s", function_name);
    return NULL;
}

static XrResult WEBGPU_UnsupportedXR(const char *function_name)
{
    SDL_SetError("WebGPU backend does not support %s", function_name);
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}

static void WEBGPU_LogStringView(const char *prefix, WGPUStringView message)
{
    if (!message.data) {
        return;
    }
    if (message.length == WGPU_STRLEN) {
        SDL_LogDebug(SDL_LOG_CATEGORY_GPU, "%s%s", prefix, message.data);
    } else if (message.length > 0) {
        SDL_LogDebug(SDL_LOG_CATEGORY_GPU, "%s%.*s", prefix, (int)message.length, message.data);
    }
}

static char *WEBGPU_StringViewDup(WGPUStringView string)
{
    if (!string.data || string.length == 0) {
        return NULL;
    }
    if (string.length == WGPU_STRLEN) {
        return SDL_strdup(string.data);
    }
    return SDL_strndup(string.data, string.length);
}

static bool WEBGPU_SetStringViewProperty(SDL_PropertiesID props, const char *name, WGPUStringView value)
{
    char *string = WEBGPU_StringViewDup(value);
    bool result;

    if (!string) {
        return false;
    }

    result = SDL_SetStringProperty(props, name, string);
    SDL_free(string);
    return result;
}

static void WEBGPU_RecordUncapturedError(WebGPURenderer *renderer, WGPUErrorType type, WGPUStringView message)
{
    char *message_copy;

    if (!renderer || !renderer->error_lock) {
        return;
    }

    message_copy = WEBGPU_StringViewDup(message);

    SDL_LockMutex(renderer->error_lock);
    SDL_free(renderer->last_uncaptured_error_message);
    renderer->last_uncaptured_error_type = type;
    renderer->last_uncaptured_error_message = message_copy;
    renderer->uncaptured_error_count += 1;
    SDL_UnlockMutex(renderer->error_lock);
}

static bool WEBGPU_DrainUncapturedError(WebGPURenderer *renderer, const char *context)
{
    WGPUErrorType type;
    char *message;
    Uint32 count;
    bool result;

    if (!renderer || !renderer->error_lock) {
        return true;
    }

    SDL_LockMutex(renderer->error_lock);
    count = renderer->uncaptured_error_count;
    if (count == 0) {
        SDL_UnlockMutex(renderer->error_lock);
        return true;
    }

    type = renderer->last_uncaptured_error_type;
    message = renderer->last_uncaptured_error_message;
    renderer->last_uncaptured_error_type = WGPUErrorType_NoError;
    renderer->last_uncaptured_error_message = NULL;
    renderer->uncaptured_error_count = 0;
    SDL_UnlockMutex(renderer->error_lock);

    if (count == 1) {
        result = SDL_SetError(
            "WebGPU backend: uncaptured WebGPU error surfaced at %s (type %d): %s",
            context,
            (int)type,
            message ? message : "<no message>");
    } else {
        result = SDL_SetError(
            "WebGPU backend: last of %u uncaptured WebGPU errors surfaced at %s (type %d): %s",
            count,
            context,
            (int)type,
            message ? message : "<no message>");
    }

    SDL_free(message);
    return result;
}

static const char *WEBGPU_DeviceLostReasonString(WGPUDeviceLostReason reason)
{
    switch (reason) {
    case WGPUDeviceLostReason_Unknown:
        return "Unknown";
    case WGPUDeviceLostReason_Destroyed:
        return "Destroyed";
    case WGPUDeviceLostReason_CallbackCancelled:
        return "CallbackCancelled";
    case WGPUDeviceLostReason_FailedCreation:
        return "FailedCreation";
    default:
        return "unrecognized";
    }
}

static void WEBGPU_LogDeviceLost(WGPUDeviceLostReason reason, WGPUStringView message)
{
    const char *reason_string = WEBGPU_DeviceLostReasonString(reason);

    if (reason == WGPUDeviceLostReason_Destroyed ||
        reason == WGPUDeviceLostReason_CallbackCancelled ||
        reason == WGPUDeviceLostReason_FailedCreation) {
        if (message.data) {
            if (message.length == WGPU_STRLEN) {
                SDL_LogDebug(SDL_LOG_CATEGORY_GPU, "WebGPU device loss filtered reason=%s: %s", reason_string, message.data);
            } else {
                SDL_LogDebug(SDL_LOG_CATEGORY_GPU, "WebGPU device loss filtered reason=%s: %.*s", reason_string, (int)message.length, message.data);
            }
        } else {
            SDL_LogDebug(SDL_LOG_CATEGORY_GPU, "WebGPU device loss filtered reason=%s", reason_string);
        }
        return;
    }

    if (message.data) {
        if (message.length == WGPU_STRLEN) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU device lost reason=%s: %s", reason_string, message.data);
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU device lost reason=%s: %.*s", reason_string, (int)message.length, message.data);
        }
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU device lost reason=%s", reason_string);
    }
}

static void WEBGPU_RecordDeviceLost(WebGPURenderer *renderer, WGPUDeviceLostReason reason, WGPUStringView message)
{
    char *message_copy;
    bool filter_loss;

    if (!renderer || !renderer->error_lock) {
        return;
    }

    WEBGPU_LogDeviceLost(reason, message);
    message_copy = WEBGPU_StringViewDup(message);

    SDL_LockMutex(renderer->error_lock);
    renderer->device_lost_future_pending = false;
    filter_loss = renderer->device_shutting_down ||
                  !renderer->device_ready ||
                  reason == WGPUDeviceLostReason_Destroyed ||
                  reason == WGPUDeviceLostReason_CallbackCancelled ||
                  reason == WGPUDeviceLostReason_FailedCreation;
    if (filter_loss || renderer->device_lost) {
        SDL_UnlockMutex(renderer->error_lock);
        SDL_free(message_copy);
        return;
    }

    SDL_free(renderer->device_lost_message);
    renderer->device_lost_reason = reason;
    renderer->device_lost_message = message_copy;
    renderer->pending_device_loss = true;
    SDL_UnlockMutex(renderer->error_lock);
}

static void WEBGPU_RecordBackendDeviceLost(WebGPURenderer *renderer, const char *message)
{
    char *message_copy;

    if (!renderer || !renderer->error_lock) {
        return;
    }

    WEBGPU_LogDeviceLost(WGPUDeviceLostReason_Unknown, WEBGPU_StringView(message));
    message_copy = SDL_strdup(message ? message : "");

    SDL_LockMutex(renderer->error_lock);
    if (renderer->device_lost || renderer->device_shutting_down || !renderer->device_ready) {
        SDL_UnlockMutex(renderer->error_lock);
        SDL_free(message_copy);
        return;
    }

    SDL_free(renderer->device_lost_message);
    renderer->device_lost_reason = WGPUDeviceLostReason_Unknown;
    renderer->device_lost_message = message_copy;
    renderer->pending_device_loss = false;
    renderer->device_lost = true;
    renderer->device_lost_future_pending = false;
    SDL_UnlockMutex(renderer->error_lock);
}

static bool WEBGPU_PollDeviceLossFuture(WebGPURenderer *renderer, const char *context, bool report_wait_error)
{
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    WGPUFuture future;
    WGPUWaitStatus status;
    bool should_poll;

    if (!renderer || !renderer->instance || !renderer->error_lock) {
        return true;
    }
    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return true;
    }

    SDL_LockMutex(renderer->error_lock);
    should_poll = renderer->device_lost_future_pending && renderer->device_lost_future.id != 0;
    future = renderer->device_lost_future;
    SDL_UnlockMutex(renderer->error_lock);

    if (!should_poll) {
        return true;
    }

    wait_info.future = future;
    status = wgpuInstanceWaitAny(renderer->instance, 1, &wait_info, 0);
    if (status == WGPUWaitStatus_TimedOut) {
        return true;
    }
    if (status != WGPUWaitStatus_Success) {
        if (report_wait_error) {
            return SDL_SetError("wgpuInstanceWaitAny failed while polling WebGPU device loss at %s with status %d", context, (int)status);
        }
        return true;
    }

    if (wait_info.completed) {
        SDL_LockMutex(renderer->error_lock);
        renderer->device_lost_future_pending = false;
        SDL_UnlockMutex(renderer->error_lock);
    }
    return true;
}

static bool WEBGPU_SetDeviceLostError(const char *context, WGPUDeviceLostReason reason, const char *message)
{
    return SDL_SetError(
        "WebGPU backend: device lost surfaced at %s (reason %s): %s",
        context,
        WEBGPU_DeviceLostReasonString(reason),
        message && *message ? message : "<no message>");
}

static bool WEBGPU_FailIfDeviceLost(WebGPURenderer *renderer, const char *context)
{
    WGPUDeviceLostReason reason;
    char *message;
    bool lost;

    if (!renderer || !renderer->error_lock) {
        return true;
    }

    SDL_LockMutex(renderer->error_lock);
    lost = renderer->device_lost;
    reason = renderer->device_lost_reason;
    message = lost ? SDL_strdup(renderer->device_lost_message ? renderer->device_lost_message : "") : NULL;
    SDL_UnlockMutex(renderer->error_lock);

    if (!lost) {
        return true;
    }

    lost = WEBGPU_SetDeviceLostError(context, reason, message);
    SDL_free(message);
    return lost;
}

static bool WEBGPU_DrainDeviceLoss(WebGPURenderer *renderer, const char *context)
{
    if (!renderer || !renderer->error_lock) {
        return true;
    }
    if (!WEBGPU_PollDeviceLossFuture(renderer, context, true)) {
        return false;
    }

    SDL_LockMutex(renderer->error_lock);
    if (renderer->pending_device_loss) {
        renderer->device_lost = true;
        renderer->pending_device_loss = false;
    }
    SDL_UnlockMutex(renderer->error_lock);

    return WEBGPU_FailIfDeviceLost(renderer, context);
}

static bool WEBGPU_DrainRuntimeErrors(WebGPURenderer *renderer, const char *context)
{
    return WEBGPU_DrainDeviceLoss(renderer, context) &&
           WEBGPU_DrainUncapturedError(renderer, context);
}

static const char *WEBGPU_BackendTypeString(WGPUBackendType backend_type)
{
    switch (backend_type) {
    case WGPUBackendType_Null:
        return "null";
    case WGPUBackendType_WebGPU:
        return "WebGPU";
    case WGPUBackendType_D3D11:
        return "D3D11";
    case WGPUBackendType_D3D12:
        return "D3D12";
    case WGPUBackendType_Metal:
        return "Metal";
    case WGPUBackendType_Vulkan:
        return "Vulkan";
    case WGPUBackendType_OpenGL:
        return "OpenGL";
    case WGPUBackendType_OpenGLES:
        return "OpenGLES";
    case WGPUBackendType_Undefined:
    default:
        return "unknown";
    }
}

static void WEBGPU_OnAdapter(
    WGPURequestAdapterStatus status,
    WGPUAdapter adapter,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUAdapterRequest *request = (WebGPUAdapterRequest *)userdata1;
    (void)userdata2;

    request->status = status;
    request->adapter = adapter;
    WEBGPU_LogStringView("RequestAdapter: ", message);
}

static void WEBGPU_OnDevice(
    WGPURequestDeviceStatus status,
    WGPUDevice device,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUDeviceRequest *request = (WebGPUDeviceRequest *)userdata1;
    (void)userdata2;

    request->status = status;
    request->device = device;
    WEBGPU_LogStringView("RequestDevice: ", message);
}

static void WEBGPU_OnUncapturedError(
    const WGPUDevice *device,
    WGPUErrorType type,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPURenderer *renderer = (WebGPURenderer *)userdata1;
    (void)device;
    (void)userdata2;

    if (message.data) {
        if (message.length == WGPU_STRLEN) {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU uncaptured error type=%d: %s", (int)type, message.data);
        } else {
            SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU uncaptured error type=%d: %.*s", (int)type, (int)message.length, message.data);
        }
    } else {
        SDL_LogError(SDL_LOG_CATEGORY_GPU, "WebGPU uncaptured error type=%d", (int)type);
    }
    WEBGPU_RecordUncapturedError(renderer, type, message);
}

static void WEBGPU_OnDeviceLost(
    const WGPUDevice *device,
    WGPUDeviceLostReason reason,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUDeviceLostCallbackState *callback_state = (WebGPUDeviceLostCallbackState *)userdata1;
    (void)device;
    (void)userdata2;

    if (!callback_state || !callback_state->renderer_owned) {
        return;
    }

    WEBGPU_RecordDeviceLost(callback_state->renderer, reason, message);
}

static void WEBGPU_OnPopErrorScope(
    WGPUPopErrorScopeStatus status,
    WGPUErrorType type,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUErrorScopeRequest *request = (WebGPUErrorScopeRequest *)userdata1;
    (void)userdata2;

    request->status = status;
    request->type = type;
    if (message.data) {
        if (message.length == WGPU_STRLEN) {
            request->message = SDL_strdup(message.data);
        } else if (message.length > 0) {
            request->message = SDL_strndup(message.data, message.length);
        }
    }
}

static void WEBGPU_OnBufferMap(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUBufferMapRequest *request = (WebGPUBufferMapRequest *)userdata1;
    (void)userdata2;

    request->status = status;
    if (message.data) {
        if (message.length == WGPU_STRLEN) {
            request->message = SDL_strdup(message.data);
        } else if (message.length > 0) {
            request->message = SDL_strndup(message.data, message.length);
        }
    }
}

static void WEBGPU_OnQueueWorkDone(
    WGPUQueueWorkDoneStatus status,
    WGPUStringView message,
    void *userdata1,
    void *userdata2)
{
    WebGPUSubmission *submission = (WebGPUSubmission *)userdata1;
    (void)userdata2;

    if (!submission) {
        return;
    }

    submission->status = status;
    WEBGPU_LogStringView("QueueOnSubmittedWorkDone: ", message);
    /* WaitAnyOnly callbacks run from wgpuInstanceWaitAny; publish status before completion. */
    SDL_SetAtomicInt(&submission->completed, 1);
}

static bool WEBGPU_WaitForFuture(WGPUInstance instance, WGPUFuture future)
{
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    WGPUWaitStatus status;

    wait_info.future = future;
    status = wgpuInstanceWaitAny(instance, 1, &wait_info, SDL_MAX_UINT64);
    return status == WGPUWaitStatus_Success && wait_info.completed;
}

static WebGPUTransferBufferGeneration *WEBGPU_CreateTransferBufferGeneration(Uint32 size)
{
    WebGPUTransferBufferGeneration *generation = (WebGPUTransferBufferGeneration *)SDL_calloc(1, sizeof(*generation));
    if (!generation) {
        return NULL;
    }

    generation->data = SDL_malloc(size);
    if (!generation->data) {
        SDL_free(generation);
        return NULL;
    }

    return generation;
}

static void WEBGPU_DestroyTransferBufferGeneration(WebGPUTransferBufferGeneration *generation)
{
    if (!generation) {
        return;
    }

    SDL_free(generation->data);
    SDL_free(generation);
}

static bool WEBGPU_AddTransferBufferGeneration(WebGPUTransferBuffer *transfer_buffer, WebGPUTransferBufferGeneration *generation)
{
    WebGPUTransferBufferGeneration **generations;

    if (transfer_buffer->generation_count >= transfer_buffer->generation_capacity) {
        Uint32 new_capacity = transfer_buffer->generation_capacity ? transfer_buffer->generation_capacity * 2 : 2;
        generations = (WebGPUTransferBufferGeneration **)SDL_realloc(transfer_buffer->generations, new_capacity * sizeof(*generations));
        if (!generations) {
            return false;
        }
        transfer_buffer->generations = generations;
        transfer_buffer->generation_capacity = new_capacity;
    }

    transfer_buffer->generations[transfer_buffer->generation_count] = generation;
    transfer_buffer->generation_count += 1;
    return true;
}

static WebGPUTransferBufferGeneration *WEBGPU_GetActiveTransferBufferGeneration(WebGPUTransferBuffer *transfer_buffer)
{
    if (!transfer_buffer || transfer_buffer->active_generation >= transfer_buffer->generation_count) {
        return NULL;
    }

    return transfer_buffer->generations[transfer_buffer->active_generation];
}

static bool WEBGPU_CycleTransferBuffer(WebGPUTransferBuffer *transfer_buffer)
{
    WebGPUTransferBufferGeneration *generation;

    if (!transfer_buffer) {
        return false;
    }

    for (Uint32 i = 0; i < transfer_buffer->generation_count; i += 1) {
        generation = transfer_buffer->generations[i];
        if (generation && generation->pending_use_count == 0) {
            transfer_buffer->active_generation = i;
            return true;
        }
    }

    generation = WEBGPU_CreateTransferBufferGeneration(transfer_buffer->size);
    if (!generation) {
        return false;
    }

    if (!WEBGPU_AddTransferBufferGeneration(transfer_buffer, generation)) {
        WEBGPU_DestroyTransferBufferGeneration(generation);
        return false;
    }

    transfer_buffer->active_generation = transfer_buffer->generation_count - 1;
    return true;
}

static void WEBGPU_DestroyTransferBuffer(WebGPUTransferBuffer *transfer_buffer)
{
    if (!transfer_buffer) {
        return;
    }

    for (Uint32 i = 0; i < transfer_buffer->generation_count; i += 1) {
        WEBGPU_DestroyTransferBufferGeneration(transfer_buffer->generations[i]);
    }
    SDL_free(transfer_buffer->debugName);
    SDL_free(transfer_buffer->generations);
    SDL_free(transfer_buffer);
}

static void WEBGPU_ReleaseTransferBufferPendingUse(
    WebGPUTransferBuffer *transfer_buffer,
    WebGPUTransferBufferGeneration *generation)
{
    if (!transfer_buffer || !generation) {
        return;
    }

    SDL_assert(generation->pending_use_count > 0);
    SDL_assert(transfer_buffer->pending_use_count > 0);
    generation->pending_use_count -= 1;
    transfer_buffer->pending_use_count -= 1;
    if (transfer_buffer->released && transfer_buffer->pending_use_count == 0) {
        WEBGPU_DestroyTransferBuffer(transfer_buffer);
    }
}

static void WEBGPU_FinishTextureDownload(WebGPUTextureDownload *download)
{
    if (!download) {
        return;
    }

    if (download->staging_buffer) {
        wgpuBufferRelease(download->staging_buffer);
        download->staging_buffer = NULL;
    }
    if (download->transfer_buffer && download->transfer_generation) {
        WebGPUTransferBuffer *transfer_buffer = download->transfer_buffer;
        WebGPUTransferBufferGeneration *generation = download->transfer_generation;
        download->transfer_buffer = NULL;
        download->transfer_generation = NULL;
        WEBGPU_ReleaseTransferBufferPendingUse(transfer_buffer, generation);
    }
}

static void WEBGPU_DestroyTextureDownload(WebGPUTextureDownload *download)
{
    if (!download) {
        return;
    }

    WEBGPU_FinishTextureDownload(download);
    SDL_free(download);
}

static void WEBGPU_FinishBufferDownload(WebGPUBufferDownload *download)
{
    if (!download) {
        return;
    }

    if (download->staging_buffer) {
        wgpuBufferRelease(download->staging_buffer);
        download->staging_buffer = NULL;
    }
    if (download->transfer_buffer && download->transfer_generation) {
        WebGPUTransferBuffer *transfer_buffer = download->transfer_buffer;
        WebGPUTransferBufferGeneration *generation = download->transfer_generation;
        download->transfer_buffer = NULL;
        download->transfer_generation = NULL;
        WEBGPU_ReleaseTransferBufferPendingUse(transfer_buffer, generation);
    }
}

static void WEBGPU_DestroyBufferDownload(WebGPUBufferDownload *download)
{
    if (!download) {
        return;
    }

    WEBGPU_FinishBufferDownload(download);
    SDL_free(download);
}

static bool WEBGPU_ProcessTextureDownload(WebGPURenderer *renderer, WebGPUTextureDownload *download)
{
    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    WebGPUBufferMapRequest request;
    WGPUFuture future;
    const Uint8 *mapped_data;
    Uint8 *destination_data;

    if (download->processed) {
        return download->succeeded ||
               SDL_SetError("previous WebGPU texture download failed");
    }

    SDL_zero(request);
    request.status = WGPUMapAsyncStatus_Error;

    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = WEBGPU_OnBufferMap;
    callback.userdata1 = &request;
    future = wgpuBufferMapAsync(download->staging_buffer, WGPUMapMode_Read, 0, (size_t)download->staging_size, callback);
    if (future.id == 0) {
        download->processed = true;
        WEBGPU_FinishTextureDownload(download);
        return SDL_SetError("wgpuBufferMapAsync failed for texture download");
    }
    if (!WEBGPU_WaitForFuture(renderer->instance, future)) {
        SDL_free(request.message);
        download->processed = true;
        WEBGPU_FinishTextureDownload(download);
        return SDL_SetError("wgpuBufferMapAsync wait failed for texture download");
    }
    if (request.status != WGPUMapAsyncStatus_Success) {
        const bool result = SDL_SetError("texture download map failed with status %d: %s", (int)request.status, request.message ? request.message : "<no message>");
        SDL_free(request.message);
        download->processed = true;
        WEBGPU_FinishTextureDownload(download);
        return result;
    }
    SDL_free(request.message);

    mapped_data = (const Uint8 *)wgpuBufferGetConstMappedRange(download->staging_buffer, 0, (size_t)download->staging_size);
    if (!mapped_data) {
        if (wgpuBufferGetMapState(download->staging_buffer) == WGPUBufferMapState_Mapped) {
            wgpuBufferUnmap(download->staging_buffer);
        }
        download->processed = true;
        WEBGPU_FinishTextureDownload(download);
        return SDL_SetError("wgpuBufferGetConstMappedRange failed for texture download");
    }

    destination_data = (Uint8 *)download->transfer_generation->data + download->destination_offset;
    for (Uint32 z = 0; z < download->depth; z += 1) {
        Uint8 *destination_layer = destination_data + (Uint64)z * download->destination_bytes_per_layer;
        const Uint8 *mapped_layer = mapped_data + (Uint64)z * download->staging_bytes_per_row * download->row_count;

        for (Uint32 y = 0; y < download->row_count; y += 1) {
            SDL_memcpy(
                destination_layer + (Uint64)y * download->destination_bytes_per_row,
                mapped_layer + (Uint64)y * download->staging_bytes_per_row,
                download->copy_bytes_per_row);
        }
    }

    wgpuBufferUnmap(download->staging_buffer);
    download->processed = true;
    download->succeeded = true;
    WEBGPU_FinishTextureDownload(download);
    return true;
}

static bool WEBGPU_ProcessBufferDownload(WebGPURenderer *renderer, WebGPUBufferDownload *download)
{
    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    WebGPUBufferMapRequest request;
    WGPUFuture future;
    const Uint8 *mapped_data;
    Uint8 *destination_data;

    if (download->processed) {
        return download->succeeded ||
               SDL_SetError("previous WebGPU buffer download failed");
    }

    SDL_zero(request);
    request.status = WGPUMapAsyncStatus_Error;

    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = WEBGPU_OnBufferMap;
    callback.userdata1 = &request;
    future = wgpuBufferMapAsync(download->staging_buffer, WGPUMapMode_Read, 0, (size_t)download->staging_size, callback);
    if (future.id == 0) {
        download->processed = true;
        WEBGPU_FinishBufferDownload(download);
        return SDL_SetError("wgpuBufferMapAsync failed for buffer download");
    }
    if (!WEBGPU_WaitForFuture(renderer->instance, future)) {
        SDL_free(request.message);
        download->processed = true;
        WEBGPU_FinishBufferDownload(download);
        return SDL_SetError("wgpuBufferMapAsync wait failed for buffer download");
    }
    if (request.status != WGPUMapAsyncStatus_Success) {
        const bool result = SDL_SetError("buffer download map failed with status %d: %s", (int)request.status, request.message ? request.message : "<no message>");
        SDL_free(request.message);
        download->processed = true;
        WEBGPU_FinishBufferDownload(download);
        return result;
    }
    SDL_free(request.message);

    mapped_data = (const Uint8 *)wgpuBufferGetConstMappedRange(download->staging_buffer, 0, (size_t)download->staging_size);
    if (!mapped_data) {
        if (wgpuBufferGetMapState(download->staging_buffer) == WGPUBufferMapState_Mapped) {
            wgpuBufferUnmap(download->staging_buffer);
        }
        download->processed = true;
        WEBGPU_FinishBufferDownload(download);
        return SDL_SetError("wgpuBufferGetConstMappedRange failed for buffer download");
    }

    destination_data = (Uint8 *)download->transfer_generation->data + download->destination_offset;
    SDL_memcpy(destination_data, mapped_data + download->source_offset, download->size);

    wgpuBufferUnmap(download->staging_buffer);
    download->processed = true;
    download->succeeded = true;
    WEBGPU_FinishBufferDownload(download);
    return true;
}

static bool WEBGPU_ProcessSubmissionDownloads(WebGPURenderer *renderer, WebGPUSubmission *submission)
{
    for (Uint32 i = 0; i < submission->texture_download_count; i += 1) {
        if (!WEBGPU_ProcessTextureDownload(renderer, submission->texture_downloads[i])) {
            return false;
        }
    }

    for (Uint32 i = 0; i < submission->buffer_download_count; i += 1) {
        if (!WEBGPU_ProcessBufferDownload(renderer, submission->buffer_downloads[i])) {
            return false;
        }
    }

    return true;
}

static bool WEBGPU_WaitForSubmission(WebGPURenderer *renderer, WebGPUSubmission *submission, Uint64 timeoutNS, bool *completed)
{
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    WGPUWaitStatus status;

    if (completed) {
        *completed = false;
    }

    if (!submission) {
        return SDL_InvalidParamError("submission");
    }
    if (submission->completion_tracking_failed) {
        return SDL_SetError("WebGPU submission completion tracking failed");
    }
    /* Processing completed downloads can yield, so keep it behind the submit-window guard. */
    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        if (timeoutNS == 0) {
            return true;
        }
        return SDL_SetError("Cannot wait for WebGPU submissions while a swapchain texture is awaiting submission");
    }

    if (SDL_GetAtomicInt(&submission->completed)) {
        if (completed) {
            *completed = true;
        }
        if (submission->status != WGPUQueueWorkDoneStatus_Success) {
            return SDL_SetError("WebGPU queue work completed with status %d", (int)submission->status);
        }
        return WEBGPU_ProcessSubmissionDownloads(renderer, submission);
    }

    if (submission->future.id == 0) {
        return SDL_SetError("WebGPU submission has no completion future");
    }

    wait_info.future = submission->future;
    status = wgpuInstanceWaitAny(renderer->instance, 1, &wait_info, timeoutNS);
    if (status == WGPUWaitStatus_TimedOut) {
        return true;
    }
    if (status != WGPUWaitStatus_Success) {
        return SDL_SetError("wgpuInstanceWaitAny failed with status %d", (int)status);
    }
    SDL_assert(!wait_info.completed || SDL_GetAtomicInt(&submission->completed));

    if (completed) {
        *completed = SDL_GetAtomicInt(&submission->completed) != 0;
    }

    if (SDL_GetAtomicInt(&submission->completed)) {
        if (submission->status != WGPUQueueWorkDoneStatus_Success) {
            return SDL_SetError("WebGPU queue work completed with status %d", (int)submission->status);
        }
        return WEBGPU_ProcessSubmissionDownloads(renderer, submission);
    }

    return true;
}

static bool WEBGPU_PopErrorScope(WebGPURenderer *renderer, const char *context)
{
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    WebGPUErrorScopeRequest request;
    WGPUFuture future;

    SDL_zero(request);
    request.status = WGPUPopErrorScopeStatus_Error;
    request.type = WGPUErrorType_Unknown;

    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = WEBGPU_OnPopErrorScope;
    callback.userdata1 = &request;
    future = wgpuDevicePopErrorScope(renderer->device, callback);
    if (future.id == 0) {
        return SDL_SetError("wgpuDevicePopErrorScope failed after %s", context);
    }
    if (!WEBGPU_WaitForFuture(renderer->instance, future)) {
        SDL_free(request.message);
        return SDL_SetError("wgpuDevicePopErrorScope wait failed after %s", context);
    }

    if (request.status != WGPUPopErrorScopeStatus_Success) {
        SDL_free(request.message);
        return SDL_SetError("wgpuDevicePopErrorScope completed with status %d after %s", (int)request.status, context);
    }
    if (request.type != WGPUErrorType_NoError) {
        const bool result = SDL_SetError("%s failed with WebGPU error type %d: %s", context, (int)request.type, request.message ? request.message : "<no message>");
        SDL_free(request.message);
        return result;
    }

    SDL_free(request.message);
    return true;
}

static WGPUBindGroup WEBGPU_CreateBindGroup(
    WebGPURenderer *renderer,
    const WGPUBindGroupDescriptor *bind_group_desc,
    const char *context)
{
    WGPUBindGroup bind_group;

    if (!WEBGPU_CanWaitForErrorScope(renderer, context)) {
        return NULL;
    }

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    bind_group = wgpuDeviceCreateBindGroup(renderer->device, bind_group_desc);
    if (!WEBGPU_PopErrorScope(renderer, context)) {
        if (bind_group) {
            wgpuBindGroupRelease(bind_group);
        }
        return NULL;
    }
    if (!bind_group) {
        SDL_SetError("WebGPU backend: %s failed", context);
        return NULL;
    }

    return bind_group;
}

static WGPUBindGroup WEBGPU_CreateEmptyBindGroup(
    WebGPURenderer *renderer,
    WGPUBindGroupLayout layout,
    WebGPUBindGroupInstrumentationPath path,
    const char *context)
{
    WGPUBindGroupDescriptor bind_group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    WGPUBindGroup bind_group;

    bind_group_desc.layout = layout;
    bind_group_desc.entryCount = 0;
    bind_group_desc.entries = NULL;
    bind_group = WEBGPU_CreateBindGroup(renderer, &bind_group_desc, context);
    if (bind_group) {
        WEBGPU_RecordBindGroupCreate(renderer, path);
    }
    return bind_group;
}

static bool WEBGPU_CreatePipelineEmptyBindGroups(
    WebGPURenderer *renderer,
    const WGPUBindGroupLayout *bind_group_layouts,
    const Uint32 *bind_group_layout_entry_counts,
    Uint32 bind_group_layout_count,
    WGPUBindGroup *empty_bind_groups,
    WebGPUBindGroupInstrumentationPath path,
    const char *context)
{
    for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
        if (bind_group_layout_entry_counts[i] == 0) {
            empty_bind_groups[i] = WEBGPU_CreateEmptyBindGroup(
                renderer,
                bind_group_layouts[i],
                path,
                context);
            if (!empty_bind_groups[i]) {
                return false;
            }
        }
    }

    return true;
}

static WGPUCommandBuffer WEBGPU_FinishCommandEncoder(WebGPURenderer *renderer, WGPUCommandEncoder encoder)
{
    WGPUCommandBuffer commands;

    if (!WEBGPU_CanWaitForErrorScope(renderer, "CommandEncoderFinish")) {
        return NULL;
    }

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    commands = wgpuCommandEncoderFinish(encoder, NULL);
    if (!WEBGPU_PopErrorScope(renderer, "CommandEncoderFinish")) {
        if (commands) {
            wgpuCommandBufferRelease(commands);
        }
        return NULL;
    }
    if (!commands) {
        WEBGPU_SetStringError("CommandEncoderFinish failed");
        return NULL;
    }

    return commands;
}

static bool WEBGPU_CommandBufferCanWaitForErrorScope(WebGPUCommandBuffer *command_buffer)
{
    return !WEBGPU_HasSwapchainSubmitWindow(command_buffer->renderer);
}

static void WEBGPU_DestroySubmission(WebGPUSubmission *submission, bool force)
{
    if (!submission) {
        return;
    }

    if (!force) {
        SDL_assert(!submission->handle_retained);
        SDL_assert(submission->presentation_refcount == 0);
        SDL_assert(!submission->submitted || submission->completion_tracking_failed || SDL_GetAtomicInt(&submission->completed));
    }

    for (Uint32 i = 0; i < submission->buffer_count; i += 1) {
        WEBGPU_ReleaseTrackedBufferReference(submission->renderer, submission->buffers[i]);
        wgpuBufferRelease(submission->buffers[i]);
    }
    SDL_free(submission->buffers);

    for (Uint32 i = 0; i < submission->graphics_pipeline_count; i += 1) {
        wgpuRenderPipelineRelease(submission->graphics_pipelines[i]);
    }
    SDL_free(submission->graphics_pipelines);

    for (Uint32 i = 0; i < submission->compute_pipeline_count; i += 1) {
        wgpuComputePipelineRelease(submission->compute_pipelines[i]);
    }
    SDL_free(submission->compute_pipelines);

    for (Uint32 i = 0; i < submission->bind_group_count; i += 1) {
        wgpuBindGroupRelease(submission->bind_groups[i]);
    }
    SDL_free(submission->bind_groups);

    for (Uint32 i = 0; i < submission->texture_download_count; i += 1) {
        WEBGPU_DestroyTextureDownload(submission->texture_downloads[i]);
    }
    SDL_free(submission->texture_downloads);

    for (Uint32 i = 0; i < submission->buffer_download_count; i += 1) {
        WEBGPU_DestroyBufferDownload(submission->buffer_downloads[i]);
    }
    SDL_free(submission->buffer_downloads);

    for (Uint32 i = 0; i < submission->sampler_count; i += 1) {
        wgpuSamplerRelease(submission->samplers[i]);
    }
    SDL_free(submission->samplers);

    for (Uint32 i = 0; i < submission->texture_view_count; i += 1) {
        wgpuTextureViewRelease(submission->texture_views[i]);
    }
    SDL_free(submission->texture_views);

    for (Uint32 i = 0; i < submission->texture_count; i += 1) {
        WEBGPU_ReleaseTrackedTextureReference(submission->renderer, submission->textures[i]);
        wgpuTextureRelease(submission->textures[i]);
    }
    SDL_free(submission->textures);

    WEBGPU_ReleaseSwapchainTexture(submission->swapchain_texture);

    SDL_free(submission);
}

static void WEBGPU_RetainSubmissionReferenceLocked(WebGPUSubmission *submission)
{
    SDL_assert(submission->refcount > 0);
    submission->refcount += 1;
}

static bool WEBGPU_ReleaseSubmissionReferenceLocked(WebGPUSubmission *submission)
{
    SDL_assert(submission->refcount > 0);
    submission->refcount -= 1;
    return submission->refcount == 0;
}

static void WEBGPU_ReleaseSubmissionReference(WebGPURenderer *renderer, WebGPUSubmission *submission)
{
    bool destroy;

    if (!submission) {
        return;
    }

    SDL_LockMutex(renderer->fence_lock);
    destroy = WEBGPU_ReleaseSubmissionReferenceLocked(submission);
    SDL_UnlockMutex(renderer->fence_lock);

    if (destroy) {
        WEBGPU_DestroySubmission(submission, false);
    }
}

static void WEBGPU_PollPendingSubmissions(WebGPURenderer *renderer)
{
    WGPUFutureWaitInfo wait_infos[WEBGPU_WAIT_ANY_MAX_FENCES];
    Uint32 batch_limit;
    Uint32 scan_index = 0;

    if (!renderer->instance || !renderer->fence_lock) {
        return;
    }
    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return;
    }

    batch_limit = renderer->wait_any_max_count > WEBGPU_WAIT_ANY_MAX_FENCES ? WEBGPU_WAIT_ANY_MAX_FENCES : (Uint32)renderer->wait_any_max_count;
    if (batch_limit == 0) {
        return;
    }

    while (true) {
        Uint32 wait_count = 0;
        Uint32 pending_count;

        if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
            return;
        }

        SDL_LockMutex(renderer->fence_lock);
        pending_count = renderer->pending_submission_count;
        if (scan_index >= pending_count) {
            SDL_UnlockMutex(renderer->fence_lock);
            break;
        }

        while (scan_index < pending_count && wait_count < batch_limit) {
            WebGPUSubmission *submission = renderer->pending_submissions[scan_index];
            scan_index += 1;
            if (submission->future.id != 0 && !SDL_GetAtomicInt(&submission->completed)) {
                wait_infos[wait_count] = (WGPUFutureWaitInfo)WGPU_FUTURE_WAIT_INFO_INIT;
                wait_infos[wait_count].future = submission->future;
                wait_count += 1;
            }
        }
        SDL_UnlockMutex(renderer->fence_lock);

        if (wait_count > 0) {
            (void)wgpuInstanceWaitAny(renderer->instance, wait_count, wait_infos, 0);
        }
    }
}

static bool WEBGPU_TrackPendingSubmission(WebGPURenderer *renderer, WebGPUSubmission *submission)
{
    WebGPUSubmission **pending_submissions;

    SDL_LockMutex(renderer->fence_lock);

    if (renderer->pending_submission_count >= renderer->pending_submission_capacity) {
        Uint32 new_capacity = renderer->pending_submission_capacity ? renderer->pending_submission_capacity * 2 : 4;
        pending_submissions = (WebGPUSubmission **)SDL_realloc(renderer->pending_submissions, new_capacity * sizeof(*pending_submissions));
        if (!pending_submissions) {
            SDL_UnlockMutex(renderer->fence_lock);
            return false;
        }
        renderer->pending_submissions = pending_submissions;
        renderer->pending_submission_capacity = new_capacity;
    }

    renderer->pending_submissions[renderer->pending_submission_count] = submission;
    renderer->pending_submission_count += 1;

    SDL_UnlockMutex(renderer->fence_lock);
    return true;
}

static bool WEBGPU_SubmissionCanRetire(WebGPUSubmission *submission)
{
    return !submission->handle_retained &&
           submission->presentation_refcount == 0 &&
           SDL_GetAtomicInt(&submission->completed);
}

static void WEBGPU_DrainRetiredSubmissions(WebGPURenderer *renderer)
{
    if (!renderer->fence_lock) {
        return;
    }

    WEBGPU_PollPendingSubmissions(renderer);

    while (true) {
        WebGPUSubmission *submission_to_destroy = NULL;
        bool destroy = false;

        SDL_LockMutex(renderer->fence_lock);
        for (Sint32 i = (Sint32)renderer->pending_submission_count - 1; i >= 0; i -= 1) {
            WebGPUSubmission *submission = renderer->pending_submissions[i];
            if (WEBGPU_SubmissionCanRetire(submission)) {
                renderer->pending_submissions[i] = renderer->pending_submissions[renderer->pending_submission_count - 1];
                renderer->pending_submission_count -= 1;
                submission_to_destroy = submission;
                break;
            }
        }
        SDL_UnlockMutex(renderer->fence_lock);

        if (!submission_to_destroy) {
            break;
        }
        if (submission_to_destroy->status == WGPUQueueWorkDoneStatus_Success) {
            (void)WEBGPU_ProcessSubmissionDownloads(renderer, submission_to_destroy);
        }

        SDL_LockMutex(renderer->fence_lock);
        destroy = WEBGPU_ReleaseSubmissionReferenceLocked(submission_to_destroy);
        SDL_UnlockMutex(renderer->fence_lock);

        if (destroy) {
            WEBGPU_DestroySubmission(submission_to_destroy, false);
        }
    }
}

static bool WEBGPU_WaitForAllPendingSubmissions(WebGPURenderer *renderer)
{
    Uint32 index = 0;
    bool result = true;
    bool tracking_failed = false;

    while (true) {
        WebGPUSubmission *submission;
        bool completed = false;
        bool submission_tracking_failed;

        SDL_LockMutex(renderer->fence_lock);
        if (index >= renderer->pending_submission_count) {
            SDL_UnlockMutex(renderer->fence_lock);
            break;
        }
        submission = renderer->pending_submissions[index];
        submission_tracking_failed = submission->completion_tracking_failed;
        WEBGPU_RetainSubmissionReferenceLocked(submission);
        SDL_UnlockMutex(renderer->fence_lock);

        if (submission_tracking_failed) {
            tracking_failed = true;
        } else if (!WEBGPU_WaitForSubmission(renderer, submission, SDL_MAX_UINT64, &completed)) {
            result = false;
        } else if (!completed) {
            SDL_SetError("WebGPU submission wait did not complete");
            result = false;
        }
        WEBGPU_ReleaseSubmissionReference(renderer, submission);
        index += 1;
    }

    if (!WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        WEBGPU_DrainRetiredSubmissions(renderer);
    }
    if (tracking_failed && result) {
        return SDL_SetError("WebGPU submission completion tracking failed");
    }
    return result && !tracking_failed;
}

static void WEBGPU_DestroyPendingSubmissions(WebGPURenderer *renderer)
{
    if (!renderer->fence_lock) {
        return;
    }

    (void)WEBGPU_WaitForAllPendingSubmissions(renderer);

    SDL_LockMutex(renderer->fence_lock);
    for (Uint32 i = 0; i < renderer->pending_submission_count; i += 1) {
        WEBGPU_DestroySubmission(renderer->pending_submissions[i], true);
    }
    renderer->pending_submission_count = 0;
    SDL_UnlockMutex(renderer->fence_lock);

    SDL_free(renderer->pending_submissions);
    renderer->pending_submissions = NULL;
    renderer->pending_submission_capacity = 0;
}

static WebGPUWindowData *WEBGPU_FetchWindowData(SDL_Window *window)
{
    SDL_PropertiesID properties = SDL_GetWindowProperties(window);
    return (WebGPUWindowData *)SDL_GetPointerProperty(properties, WINDOW_PROPERTY_DATA, NULL);
}

static bool WEBGPU_WindowDataIsActive(WebGPUWindowData *window_data)
{
    return window_data &&
           window_data->refcount > 0 &&
           window_data->window &&
           window_data->surface;
}

static WGPUTextureFormat WEBGPU_ToWGPUTextureFormat(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
        return WGPUTextureFormat_R8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
        return WGPUTextureFormat_R8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
        return WGPUTextureFormat_R8Uint;
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
        return WGPUTextureFormat_R8Sint;
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
        return WGPUTextureFormat_RG8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
        return WGPUTextureFormat_RG8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
        return WGPUTextureFormat_RG8Uint;
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
        return WGPUTextureFormat_RG8Sint;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
        return WGPUTextureFormat_RGBA8Unorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
        return WGPUTextureFormat_RGBA8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        return WGPUTextureFormat_RGBA8Snorm;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
        return WGPUTextureFormat_RGBA8Uint;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
        return WGPUTextureFormat_RGBA8Sint;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
        return WGPUTextureFormat_BGRA8Unorm;
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
        return WGPUTextureFormat_BGRA8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
        return WGPUTextureFormat_R16Unorm;
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
        return WGPUTextureFormat_R16Snorm;
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
        return WGPUTextureFormat_R16Uint;
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
        return WGPUTextureFormat_R16Sint;
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
        return WGPUTextureFormat_R16Float;
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
        return WGPUTextureFormat_RG16Unorm;
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
        return WGPUTextureFormat_RG16Snorm;
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
        return WGPUTextureFormat_RG16Uint;
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
        return WGPUTextureFormat_RG16Sint;
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
        return WGPUTextureFormat_RG16Float;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
        return WGPUTextureFormat_RGBA16Unorm;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
        return WGPUTextureFormat_RGBA16Snorm;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
        return WGPUTextureFormat_RGBA16Uint;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
        return WGPUTextureFormat_RGBA16Sint;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
        return WGPUTextureFormat_RGBA16Float;
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
        return WGPUTextureFormat_R32Float;
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
        return WGPUTextureFormat_R32Uint;
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
        return WGPUTextureFormat_R32Sint;
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
        return WGPUTextureFormat_RG32Float;
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
        return WGPUTextureFormat_RG32Uint;
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
        return WGPUTextureFormat_RG32Sint;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
        return WGPUTextureFormat_RGBA32Float;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
        return WGPUTextureFormat_RGBA32Uint;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
        return WGPUTextureFormat_RGBA32Sint;
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        return WGPUTextureFormat_RGB10A2Unorm;
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
        return WGPUTextureFormat_RG11B10Ufloat;
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
        return WGPUTextureFormat_BC1RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC1RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
        return WGPUTextureFormat_BC2RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC2RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
        return WGPUTextureFormat_BC3RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC3RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
        return WGPUTextureFormat_BC4RUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
        return WGPUTextureFormat_BC5RGUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
        return WGPUTextureFormat_BC6HRGBFloat;
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
        return WGPUTextureFormat_BC6HRGBUfloat;
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
        return WGPUTextureFormat_BC7RGBAUnorm;
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
        return WGPUTextureFormat_BC7RGBAUnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
        return WGPUTextureFormat_ASTC4x4Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
        return WGPUTextureFormat_ASTC4x4UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
        return WGPUTextureFormat_ASTC5x4Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
        return WGPUTextureFormat_ASTC5x4UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
        return WGPUTextureFormat_ASTC5x5Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC5x5UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
        return WGPUTextureFormat_ASTC6x5Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC6x5UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
        return WGPUTextureFormat_ASTC6x6Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC6x6UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
        return WGPUTextureFormat_ASTC8x5Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x5UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
        return WGPUTextureFormat_ASTC8x6Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x6UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
        return WGPUTextureFormat_ASTC8x8Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
        return WGPUTextureFormat_ASTC8x8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
        return WGPUTextureFormat_ASTC10x5Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x5UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
        return WGPUTextureFormat_ASTC10x6Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x6UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
        return WGPUTextureFormat_ASTC10x8Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x8UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
        return WGPUTextureFormat_ASTC10x10Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
        return WGPUTextureFormat_ASTC10x10UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
        return WGPUTextureFormat_ASTC12x10Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
        return WGPUTextureFormat_ASTC12x10UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
        return WGPUTextureFormat_ASTC12x12Unorm;
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
        return WGPUTextureFormat_ASTC12x12UnormSrgb;
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
        return WGPUTextureFormat_Depth16Unorm;
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
        return WGPUTextureFormat_Depth24Plus;
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
        return WGPUTextureFormat_Depth24PlusStencil8;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
        return WGPUTextureFormat_Depth32Float;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return WGPUTextureFormat_Depth32FloatStencil8;
    default:
        return WGPUTextureFormat_Undefined;
    }
}

static SDL_GPUTextureFormat WEBGPU_ToSDLTextureFormat(WGPUTextureFormat format)
{
    switch (format) {
    case WGPUTextureFormat_R8Unorm:
        return SDL_GPU_TEXTUREFORMAT_R8_UNORM;
    case WGPUTextureFormat_R8Snorm:
        return SDL_GPU_TEXTUREFORMAT_R8_SNORM;
    case WGPUTextureFormat_R8Uint:
        return SDL_GPU_TEXTUREFORMAT_R8_UINT;
    case WGPUTextureFormat_R8Sint:
        return SDL_GPU_TEXTUREFORMAT_R8_INT;
    case WGPUTextureFormat_RG8Unorm:
        return SDL_GPU_TEXTUREFORMAT_R8G8_UNORM;
    case WGPUTextureFormat_RG8Snorm:
        return SDL_GPU_TEXTUREFORMAT_R8G8_SNORM;
    case WGPUTextureFormat_RG8Uint:
        return SDL_GPU_TEXTUREFORMAT_R8G8_UINT;
    case WGPUTextureFormat_RG8Sint:
        return SDL_GPU_TEXTUREFORMAT_R8G8_INT;
    case WGPUTextureFormat_RGBA8Unorm:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    case WGPUTextureFormat_RGBA8UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB;
    case WGPUTextureFormat_RGBA8Snorm:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM;
    case WGPUTextureFormat_RGBA8Uint:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT;
    case WGPUTextureFormat_RGBA8Sint:
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT;
    case WGPUTextureFormat_BGRA8Unorm:
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM;
    case WGPUTextureFormat_BGRA8UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB;
    case WGPUTextureFormat_R16Unorm:
        return SDL_GPU_TEXTUREFORMAT_R16_UNORM;
    case WGPUTextureFormat_R16Snorm:
        return SDL_GPU_TEXTUREFORMAT_R16_SNORM;
    case WGPUTextureFormat_R16Uint:
        return SDL_GPU_TEXTUREFORMAT_R16_UINT;
    case WGPUTextureFormat_R16Sint:
        return SDL_GPU_TEXTUREFORMAT_R16_INT;
    case WGPUTextureFormat_R16Float:
        return SDL_GPU_TEXTUREFORMAT_R16_FLOAT;
    case WGPUTextureFormat_RG16Unorm:
        return SDL_GPU_TEXTUREFORMAT_R16G16_UNORM;
    case WGPUTextureFormat_RG16Snorm:
        return SDL_GPU_TEXTUREFORMAT_R16G16_SNORM;
    case WGPUTextureFormat_RG16Uint:
        return SDL_GPU_TEXTUREFORMAT_R16G16_UINT;
    case WGPUTextureFormat_RG16Sint:
        return SDL_GPU_TEXTUREFORMAT_R16G16_INT;
    case WGPUTextureFormat_RG16Float:
        return SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT;
    case WGPUTextureFormat_RGBA16Unorm:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM;
    case WGPUTextureFormat_RGBA16Snorm:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM;
    case WGPUTextureFormat_RGBA16Uint:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT;
    case WGPUTextureFormat_RGBA16Sint:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT;
    case WGPUTextureFormat_RGBA16Float:
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    case WGPUTextureFormat_R32Float:
        return SDL_GPU_TEXTUREFORMAT_R32_FLOAT;
    case WGPUTextureFormat_R32Uint:
        return SDL_GPU_TEXTUREFORMAT_R32_UINT;
    case WGPUTextureFormat_R32Sint:
        return SDL_GPU_TEXTUREFORMAT_R32_INT;
    case WGPUTextureFormat_RG32Float:
        return SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
    case WGPUTextureFormat_RG32Uint:
        return SDL_GPU_TEXTUREFORMAT_R32G32_UINT;
    case WGPUTextureFormat_RG32Sint:
        return SDL_GPU_TEXTUREFORMAT_R32G32_INT;
    case WGPUTextureFormat_RGBA32Float:
        return SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    case WGPUTextureFormat_RGBA32Uint:
        return SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT;
    case WGPUTextureFormat_RGBA32Sint:
        return SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT;
    case WGPUTextureFormat_RGB10A2Unorm:
        return SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM;
    case WGPUTextureFormat_RG11B10Ufloat:
        return SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT;
    case WGPUTextureFormat_BC1RGBAUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM;
    case WGPUTextureFormat_BC1RGBAUnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC2RGBAUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM;
    case WGPUTextureFormat_BC2RGBAUnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC3RGBAUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM;
    case WGPUTextureFormat_BC3RGBAUnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_BC4RUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM;
    case WGPUTextureFormat_BC5RGUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM;
    case WGPUTextureFormat_BC6HRGBFloat:
        return SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT;
    case WGPUTextureFormat_BC6HRGBUfloat:
        return SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT;
    case WGPUTextureFormat_BC7RGBAUnorm:
        return SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM;
    case WGPUTextureFormat_BC7RGBAUnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB;
    case WGPUTextureFormat_ASTC4x4Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM;
    case WGPUTextureFormat_ASTC4x4UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB;
    case WGPUTextureFormat_ASTC5x4Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM;
    case WGPUTextureFormat_ASTC5x4UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB;
    case WGPUTextureFormat_ASTC5x5Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM;
    case WGPUTextureFormat_ASTC5x5UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC6x5Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM;
    case WGPUTextureFormat_ASTC6x5UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC6x6Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM;
    case WGPUTextureFormat_ASTC6x6UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x5Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM;
    case WGPUTextureFormat_ASTC8x5UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x6Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM;
    case WGPUTextureFormat_ASTC8x6UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC8x8Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM;
    case WGPUTextureFormat_ASTC8x8UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x5Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM;
    case WGPUTextureFormat_ASTC10x5UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x6Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM;
    case WGPUTextureFormat_ASTC10x6UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x8Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM;
    case WGPUTextureFormat_ASTC10x8UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB;
    case WGPUTextureFormat_ASTC10x10Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM;
    case WGPUTextureFormat_ASTC10x10UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB;
    case WGPUTextureFormat_ASTC12x10Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM;
    case WGPUTextureFormat_ASTC12x10UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB;
    case WGPUTextureFormat_ASTC12x12Unorm:
        return SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM;
    case WGPUTextureFormat_ASTC12x12UnormSrgb:
        return SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB;
    case WGPUTextureFormat_Depth16Unorm:
        return SDL_GPU_TEXTUREFORMAT_D16_UNORM;
    case WGPUTextureFormat_Depth32Float:
        return SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    case WGPUTextureFormat_Depth32FloatStencil8:
        return SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT;
    default:
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
}

static WGPUTextureFormat WEBGPU_ToSwapchainViewFormat(WGPUTextureFormat surface_format, SDL_GPUSwapchainComposition composition)
{
    switch (composition) {
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR:
        return surface_format;
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR:
        switch (surface_format) {
        case WGPUTextureFormat_RGBA8Unorm:
            return WGPUTextureFormat_RGBA8UnormSrgb;
        case WGPUTextureFormat_BGRA8Unorm:
            return WGPUTextureFormat_BGRA8UnormSrgb;
        default:
            return WGPUTextureFormat_Undefined;
        }
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR:
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084:
        return WGPUTextureFormat_Undefined;
    }
    return WGPUTextureFormat_Undefined;
}

static SDL_GPUTextureFormat WEBGPU_ToSwapchainSDLFormat(WGPUTextureFormat surface_format, SDL_GPUSwapchainComposition composition)
{
    return WEBGPU_ToSDLTextureFormat(WEBGPU_ToSwapchainViewFormat(surface_format, composition));
}

static bool WEBGPU_IsDepthStencilFormat(SDL_GPUTextureFormat format)
{
    return format == SDL_GPU_TEXTUREFORMAT_D16_UNORM ||
           format == SDL_GPU_TEXTUREFORMAT_D24_UNORM ||
           format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT ||
           format == SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT ||
           format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT;
}

static bool WEBGPU_TextureFormatIsAcceptedDepthCube(SDL_GPUTextureFormat format)
{
    return format == SDL_GPU_TEXTUREFORMAT_D16_UNORM ||
           format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
}

static bool WEBGPU_IsStencilFormat(SDL_GPUTextureFormat format)
{
    return format == SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT ||
           format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT;
}

static bool WEBGPU_TextureFormatIsBCCompressedLDR(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatIsBCCompressed(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
        return true;
    default:
        return WEBGPU_TextureFormatIsBCCompressedLDR(format);
    }
}

static bool WEBGPU_TextureFormatIsASTCCompressedLDR(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
        return true;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatIsCompressed(SDL_GPUTextureFormat format)
{
    return WEBGPU_TextureFormatIsBCCompressed(format) ||
           WEBGPU_TextureFormatIsASTCCompressedLDR(format);
}

static Uint32 WEBGPU_TextureFormatBlockWidth(SDL_GPUTextureFormat format)
{
    return (Uint32)SDL_max(Texture_GetBlockWidth(format), 1);
}

static Uint32 WEBGPU_TextureFormatBlockHeight(SDL_GPUTextureFormat format)
{
    return (Uint32)SDL_max(Texture_GetBlockHeight(format), 1);
}

static bool WEBGPU_TextureFormatSupportedByDevice(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
        return renderer->supports_texture_compression_bc;
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
        return renderer->supports_texture_compression_astc;
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
        return renderer->supports_texture_formats_tier1 ||
               renderer->supports_unorm16_texture_formats;
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return renderer->supports_depth32float_stencil8;
    default:
        return WEBGPU_ToWGPUTextureFormat(format) != WGPUTextureFormat_Undefined;
    }
}

static bool WEBGPU_TextureFormatSupportsColorTarget(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        return true;
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
        return renderer->supports_rg11b10ufloat_renderable;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatSupportsMultisampleColorTarget(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        return true;
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
        return renderer->supports_rg11b10ufloat_renderable;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
        return renderer->supports_core_features_and_limits;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatSupportsBlend(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
        return true;
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
        return renderer->supports_texture_formats_tier1;
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
        return renderer->supports_rg11b10ufloat_renderable;
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
        return renderer->supports_float32_blendable;
    default:
        return false;
    }
}

static Uint32 WEBGPU_AlignUp(Uint32 value, Uint32 alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static Uint64 WEBGPU_AlignUp64(Uint64 value, Uint64 alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static Uint64 WEBGPU_BufferAllocationSize(Uint32 logical_size)
{
    return WEBGPU_AlignUp64(logical_size, WEBGPU_BUFFER_COPY_ALIGNMENT);
}

static bool WEBGPU_IsPowerOfTwo32(Uint32 value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static bool WEBGPU_FColorIsFinite(SDL_FColor color)
{
    return !SDL_isnanf(color.r) && !SDL_isinff(color.r) &&
           !SDL_isnanf(color.g) && !SDL_isinff(color.g) &&
           !SDL_isnanf(color.b) && !SDL_isinff(color.b) &&
           !SDL_isnanf(color.a) && !SDL_isinff(color.a);
}

static Uint32 WEBGPU_TextureBytesPerPixel(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
        return 1;
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
        return 2;
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_R10G10B10A2_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R11G11B10_UFLOAT:
        return 4;
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
        return 8;
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
        return 16;
    default:
        return 0;
    }
}

static bool WEBGPU_TextureFormatSupportedForType(
    WebGPURenderer *renderer,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type)
{
    if (type != SDL_GPU_TEXTURETYPE_3D) {
        return true;
    }

    if (WEBGPU_TextureFormatIsBCCompressed(format)) {
        return renderer->supports_texture_compression_bc_sliced_3d;
    }
    if (WEBGPU_TextureFormatIsASTCCompressedLDR(format)) {
        return renderer->supports_texture_compression_astc_sliced_3d;
    }
    return true;
}

static Uint32 WEBGPU_TextureMaxMipLevels(Uint32 width, Uint32 height, Uint32 depth)
{
    Uint32 dimension = SDL_max(SDL_max(width, height), depth);
    Uint32 levels = 1;

    while (dimension > 1) {
        dimension >>= 1;
        levels += 1;
    }

    return levels;
}

static bool WEBGPU_ValidateTextureDimensions(WebGPURenderer *renderer, const SDL_GPUTextureCreateInfo *createinfo)
{
    if (WEBGPU_TextureFormatIsCompressed(createinfo->format) &&
        ((createinfo->width % WEBGPU_TextureFormatBlockWidth(createinfo->format)) != 0 ||
         (createinfo->height % WEBGPU_TextureFormatBlockHeight(createinfo->format)) != 0)) {
        WEBGPU_SetStringError("compressed texture dimensions must be multiples of the format block size");
        return false;
    }

    if (createinfo->type == SDL_GPU_TEXTURETYPE_3D) {
        if (createinfo->width > renderer->limits.maxTextureDimension3D ||
            createinfo->height > renderer->limits.maxTextureDimension3D ||
            createinfo->layer_count_or_depth > renderer->limits.maxTextureDimension3D) {
            WEBGPU_SetStringError("texture dimensions exceed WebGPU limits");
            return false;
        }
    } else {
        if (createinfo->width > renderer->limits.maxTextureDimension2D ||
            createinfo->height > renderer->limits.maxTextureDimension2D) {
            WEBGPU_SetStringError("texture dimensions exceed WebGPU limits");
            return false;
        }
        if (createinfo->type != SDL_GPU_TEXTURETYPE_2D &&
            createinfo->layer_count_or_depth > renderer->limits.maxTextureArrayLayers) {
            WEBGPU_SetStringError("texture layer count exceeds WebGPU limit");
            return false;
        }
    }

    return true;
}

static Uint32 WEBGPU_TextureMipDimension(Uint32 dimension, Uint32 mip_level)
{
    dimension >>= mip_level;
    return dimension ? dimension : 1;
}

static Uint32 WEBGPU_TextureTransferBlockSize(SDL_GPUTextureFormat format)
{
    if (WEBGPU_TextureFormatIsCompressed(format)) {
        return SDL_GPUTextureFormatTexelBlockSize(format);
    }
    return WEBGPU_TextureBytesPerPixel(format);
}

static Uint64 WEBGPU_TextureBlockCount(Uint64 texel_count, Uint32 block_size)
{
    return (texel_count + block_size - 1) / block_size;
}

static WGPUTextureSampleType WEBGPU_ToTextureSampleType(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_B8G8R8A8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_R16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC4_R_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC5_RG_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC6H_RGB_UFLOAT:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM:
    case SDL_GPU_TEXTUREFORMAT_BC1_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC2_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC3_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_BC7_RGBA_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM:
    case SDL_GPU_TEXTUREFORMAT_ASTC_4x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x4_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_5x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_6x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_8x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x5_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x6_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x8_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_10x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x10_UNORM_SRGB:
    case SDL_GPU_TEXTUREFORMAT_ASTC_12x12_UNORM_SRGB:
        return WGPUTextureSampleType_Float;
    case SDL_GPU_TEXTUREFORMAT_R8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_UINT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_UINT:
        return WGPUTextureSampleType_Uint;
    case SDL_GPU_TEXTUREFORMAT_R8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8_INT:
    case SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT:
    case SDL_GPU_TEXTUREFORMAT_R16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16_INT:
    case SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT:
    case SDL_GPU_TEXTUREFORMAT_R32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_INT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_INT:
        return WGPUTextureSampleType_Sint;
    case SDL_GPU_TEXTUREFORMAT_D16_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT:
    case SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT:
        return WGPUTextureSampleType_Depth;
    default:
        return WGPUTextureSampleType_Undefined;
    }
}

static bool WEBGPU_TextureFormatRequiresFloat32FilterableFeature(SDL_GPUTextureFormat format)
{
    switch (format) {
    case SDL_GPU_TEXTUREFORMAT_R32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT:
    case SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT:
        return true;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatSupportsFilterableFloatSample(WebGPURenderer *renderer, SDL_GPUTextureFormat format)
{
    if (WEBGPU_TextureFormatRequiresFloat32FilterableFeature(format)) {
        return renderer->supports_float32_filterable;
    }
    return WEBGPU_ToTextureSampleType(format) == WGPUTextureSampleType_Float;
}

static bool WEBGPU_TextureFormatSupportsTextureBinding(SDL_GPUTextureFormat format)
{
    return WEBGPU_ToTextureSampleType(format) != WGPUTextureSampleType_Undefined ||
           SDL_GPUTextureFormatSupportsUnfilterableFloatSample(format);
}

static bool WEBGPU_TextureFormatSupportsTextureBindingForType(
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type)
{
    WGPUTextureSampleType sample_type = WEBGPU_ToTextureSampleType(format);

    if ((type == SDL_GPU_TEXTURETYPE_CUBE ||
         type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) &&
        (sample_type == WGPUTextureSampleType_Sint ||
         sample_type == WGPUTextureSampleType_Uint)) {
        return false;
    }

    return WEBGPU_TextureFormatSupportsTextureBinding(format);
}

static bool WEBGPU_TextureFormatMatchesSampleType(
    WebGPURenderer *renderer,
    SDL_GPUTextureFormat format,
    WGPUTextureSampleType sample_type)
{
    switch (sample_type) {
    case WGPUTextureSampleType_Float:
        return WEBGPU_TextureFormatSupportsFilterableFloatSample(renderer, format);
    case WGPUTextureSampleType_UnfilterableFloat:
        return SDL_GPUTextureFormatSupportsUnfilterableFloatSample(format);
    case WGPUTextureSampleType_Depth:
        return WEBGPU_ToTextureSampleType(format) == WGPUTextureSampleType_Depth;
    case WGPUTextureSampleType_Sint:
        return WEBGPU_ToTextureSampleType(format) == WGPUTextureSampleType_Sint;
    case WGPUTextureSampleType_Uint:
        return WEBGPU_ToTextureSampleType(format) == WGPUTextureSampleType_Uint;
    default:
        return false;
    }
}

static bool WEBGPU_TextureFormatMatchesSampledTextureLayout(
    WebGPURenderer *renderer,
    SDL_GPUTextureFormat format,
    WGPUTextureSampleType sample_type,
    bool multisampled)
{
    if (multisampled) {
        switch (sample_type) {
        case WGPUTextureSampleType_UnfilterableFloat:
            return SDL_GPUTextureFormatIsAcceptedMultisampledSampledColor(format);
        case WGPUTextureSampleType_Depth:
            return SDL_GPUTextureFormatIsAcceptedMultisampledSampledDepth(format);
        default:
            return false;
        }
    }

    return WEBGPU_TextureFormatMatchesSampleType(renderer, format, sample_type);
}

static bool WEBGPU_ValidateSampledTextureLayoutSupport(
    WebGPUSampledTextureSampleKind sample_type,
    WebGPUSamplerBindingKind sampler_type,
    bool multisampled)
{
    switch (sample_type) {
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT:
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT:
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH:
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_SINT:
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UINT:
        break;
    default:
        WEBGPU_SetStringError("invalid sampled texture sample type layout");
        return false;
    }

    if (multisampled) {
        if (sampler_type != WEBGPU_SAMPLER_BINDINGKIND_NONE) {
            WEBGPU_SetStringError("multisampled sampled texture layout requires samplerless slots");
            return false;
        }
        if (sample_type != WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT &&
            sample_type != WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH) {
            WEBGPU_SetStringError("unsupported multisampled sampled texture sample type layout");
            return false;
        }
        return true;
    }

    switch (sampler_type) {
    case WEBGPU_SAMPLER_BINDINGKIND_NONE:
        if (sample_type == WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH) {
            WEBGPU_SetStringError("samplerless sampled texture layout does not support depth textures");
            return false;
        }
        return true;
    case WEBGPU_SAMPLER_BINDINGKIND_FILTERING:
        if (sample_type != WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT) {
            WEBGPU_SetStringError("filtering sampler binding layout requires filterable-float sampled texture layout");
            return false;
        }
        return true;
    case WEBGPU_SAMPLER_BINDINGKIND_NONFILTERING:
        if (sample_type == WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_SINT ||
            sample_type == WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UINT) {
            WEBGPU_SetStringError("integer sampled texture layout requires samplerless slots");
            return false;
        }
        return true;
    case WEBGPU_SAMPLER_BINDINGKIND_COMPARISON:
        if (sample_type != WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH) {
            WEBGPU_SetStringError("comparison sampler binding layout requires depth sampled texture layout");
            return false;
        }
        return true;
    default:
        WEBGPU_SetStringError("invalid sampler binding type layout");
        return false;
    }
}

static bool WEBGPU_ToStorageTextureViewDimension(
    SDL_GPUTextureType texture_type,
    WGPUTextureViewDimension *view_dimension)
{
    switch (texture_type) {
    case SDL_GPU_TEXTURETYPE_2D:
        *view_dimension = WGPUTextureViewDimension_2D;
        return true;
    case SDL_GPU_TEXTURETYPE_2D_ARRAY:
        *view_dimension = WGPUTextureViewDimension_2DArray;
        return true;
    case SDL_GPU_TEXTURETYPE_3D:
        *view_dimension = WGPUTextureViewDimension_3D;
        return true;
    default:
        WEBGPU_SetStringError("unsupported storage texture type");
        return false;
    }
}

static SDL_GPUTextureType WEBGPU_StorageTextureTypeFromViewDimension(
    WGPUTextureViewDimension view_dimension)
{
    switch (view_dimension) {
    case WGPUTextureViewDimension_2D:
        return SDL_GPU_TEXTURETYPE_2D;
    case WGPUTextureViewDimension_2DArray:
        return SDL_GPU_TEXTURETYPE_2D_ARRAY;
    case WGPUTextureViewDimension_3D:
        return SDL_GPU_TEXTURETYPE_3D;
    default:
        return SDL_GPU_TEXTURETYPE_2D;
    }
}

static const char *WEBGPU_ComputeStorageTextureUsageNameForAccess(WGPUStorageTextureAccess access)
{
    switch (access) {
    case WGPUStorageTextureAccess_ReadOnly:
        return "COMPUTE_STORAGE_READ";
    case WGPUStorageTextureAccess_WriteOnly:
        return "COMPUTE_STORAGE_WRITE";
    case WGPUStorageTextureAccess_ReadWrite:
        return "COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE";
    default:
        return "supported compute storage texture";
    }
}

static bool WEBGPU_ComputeStorageTextureUsageMatchesAccess(
    SDL_GPUTextureUsageFlags usage,
    WGPUStorageTextureAccess access)
{
    switch (access) {
    case WGPUStorageTextureAccess_ReadOnly:
        return usage == SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ;
    case WGPUStorageTextureAccess_WriteOnly:
        return (usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) != 0;
    case WGPUStorageTextureAccess_ReadWrite:
        return usage == SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE;
    default:
        return false;
    }
}

static bool WEBGPU_StorageTextureReadWriteFormatRequiresTier2(
    SDL_GPUTextureFormat format)
{
    return format == SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM ||
           format == SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT ||
           format == SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT ||
           format == SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT ||
           format == SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT ||
           format == SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT ||
           format == SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT;
}

static bool WEBGPU_StorageTextureReadOnlyFormatRequiresCoreFeaturesAndLimits(
    SDL_GPUTextureFormat format)
{
    return format == SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT;
}

static bool WEBGPU_ValidateComputeStorageTextureViewBindingLayout(
    const WebGPUStorageTextureBindingLayout *layout,
    WGPUTextureViewDimension bound_view_dimension,
    SDL_GPUTextureFormat texture_format,
    SDL_GPUTextureUsageFlags texture_usage,
    const char *binding_context)
{
    WGPUTextureFormat bound_format;

    if (bound_view_dimension != layout->view_dimension) {
        SDL_SetError("%s storage texture type does not match shader resource layout", binding_context);
        return false;
    }
    bound_format = WEBGPU_ToWGPUTextureFormat(texture_format);
    if (bound_format != layout->format) {
        SDL_SetError("%s storage texture format does not match shader resource layout", binding_context);
        return false;
    }
    if (!WEBGPU_ComputeStorageTextureUsageMatchesAccess(texture_usage, layout->access)) {
        SDL_SetError(
            "%s storage texture binding requires %s usage",
            binding_context,
            WEBGPU_ComputeStorageTextureUsageNameForAccess(layout->access));
        return false;
    }

    return true;
}

static WGPUTextureUsage WEBGPU_ToTextureUsage(SDL_GPUTextureFormat format, SDL_GPUTextureUsageFlags usage)
{
    // SDL_GPU copy operations can use created textures as copy sources or
    // destinations without declaring WebGPU-style transfer usage upfront.
    // Zero-usage depth/stencil transfer-only textures are rejected before this.
    WGPUTextureUsage result = IsD24Format(format) ? 0 : (WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst);

    if (usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
        result |= WGPUTextureUsage_TextureBinding;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        result |= WGPUTextureUsage_RenderAttachment;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        result |= WGPUTextureUsage_RenderAttachment;
    }
    if (usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE |
                 SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        result |= WGPUTextureUsage_StorageBinding;
    }

    return result;
}

static WGPUTextureDimension WEBGPU_ToTextureDimension(SDL_GPUTextureType type)
{
    switch (type) {
    case SDL_GPU_TEXTURETYPE_2D:
    case SDL_GPU_TEXTURETYPE_2D_ARRAY:
    case SDL_GPU_TEXTURETYPE_CUBE:
    case SDL_GPU_TEXTURETYPE_CUBE_ARRAY:
        return WGPUTextureDimension_2D;
    case SDL_GPU_TEXTURETYPE_3D:
        return WGPUTextureDimension_3D;
    default:
        return WGPUTextureDimension_Undefined;
    }
}

static bool WEBGPU_TextureViewUsageIsRenderAttachment(WebGPUTextureViewUsage usage)
{
    return usage == WEBGPU_TEXTURE_VIEW_USAGE_COLOR_ATTACHMENT ||
           usage == WEBGPU_TEXTURE_VIEW_USAGE_DEPTH_STENCIL_ATTACHMENT ||
           usage == WEBGPU_TEXTURE_VIEW_USAGE_RESOLVE_ATTACHMENT ||
           usage == WEBGPU_TEXTURE_VIEW_USAGE_SWAPCHAIN_ATTACHMENT;
}

static WGPUTextureUsage WEBGPU_ToTextureViewWGPUUsage(WebGPUTextureViewUsage usage)
{
    switch (usage) {
    case WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED:
    case WEBGPU_TEXTURE_VIEW_USAGE_BLIT_SOURCE:
        return WGPUTextureUsage_TextureBinding;
    case WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ:
    case WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READWRITE:
        return WGPUTextureUsage_StorageBinding;
    case WEBGPU_TEXTURE_VIEW_USAGE_COLOR_ATTACHMENT:
    case WEBGPU_TEXTURE_VIEW_USAGE_DEPTH_STENCIL_ATTACHMENT:
    case WEBGPU_TEXTURE_VIEW_USAGE_RESOLVE_ATTACHMENT:
    case WEBGPU_TEXTURE_VIEW_USAGE_SWAPCHAIN_ATTACHMENT:
        return WGPUTextureUsage_RenderAttachment;
    default:
        return WGPUTextureUsage_None;
    }
}

static WGPUTextureViewDimension WEBGPU_ToTextureViewDimension(SDL_GPUTextureType type, WebGPUTextureViewUsage usage, Uint32 array_layer_count)
{
    switch (type) {
    case SDL_GPU_TEXTURETYPE_2D:
        return WGPUTextureViewDimension_2D;
    case SDL_GPU_TEXTURETYPE_2D_ARRAY:
        // Single-layer render attachment views over 2D arrays are 2D views.
        // Sampled layout facts still use WEBGPU_ToSampledTextureViewDimension.
        if (WEBGPU_TextureViewUsageIsRenderAttachment(usage) &&
            array_layer_count == 1) {
            return WGPUTextureViewDimension_2D;
        }
        return WGPUTextureViewDimension_2DArray;
    case SDL_GPU_TEXTURETYPE_CUBE:
        // Cube faces are single-layer 2D render attachment views.
        if (WEBGPU_TextureViewUsageIsRenderAttachment(usage) && array_layer_count == 1) {
            return WGPUTextureViewDimension_2D;
        }
        return WGPUTextureViewDimension_Cube;
    case SDL_GPU_TEXTURETYPE_CUBE_ARRAY:
        // Cube-array faces are single-layer 2D render attachment views.
        if (WEBGPU_TextureViewUsageIsRenderAttachment(usage) && array_layer_count == 1) {
            return WGPUTextureViewDimension_2D;
        }
        return WGPUTextureViewDimension_CubeArray;
    case SDL_GPU_TEXTURETYPE_3D:
        return WGPUTextureViewDimension_3D;
    default:
        return WGPUTextureViewDimension_Undefined;
    }
}

typedef struct WebGPUTextureViewDescription
{
    SDL_GPUTextureFormat format;
    WGPUTextureViewDimension dimension;
    WGPUTextureAspect aspect;
    WebGPUTextureViewUsage usage;
    Uint64 generation;
    Uint32 base_mip_level;
    Uint32 mip_level_count;
    Uint32 base_array_layer;
    Uint32 array_layer_count;
} WebGPUTextureViewDescription;

typedef struct WebGPURenderPassColorAttachmentInfo
{
    WebGPUTexture *texture;
    WebGPUTextureViewDescription view_description;
    WebGPUSwapchainTextureDestination swapchain_destination;
    SDL_FColor clear_color;
    SDL_GPULoadOp load_op;
    SDL_GPUStoreOp store_op;
    Uint32 mip_level;
    Uint32 layer_or_depth_plane;
    bool cycle;

    WebGPUTexture *resolve_texture;
    WebGPUTextureViewDescription resolve_view_description;
    WebGPUSwapchainTextureDestination resolve_swapchain_destination;
    Uint32 resolve_mip_level;
    Uint32 resolve_layer;
    bool cycle_resolve_texture;
} WebGPURenderPassColorAttachmentInfo;

typedef struct WebGPURenderPassDepthStencilAttachmentInfo
{
    WebGPUTexture *texture;
    WebGPUTextureViewDescription view_description;
    float clear_depth;
    SDL_GPULoadOp load_op;
    SDL_GPUStoreOp store_op;
    SDL_GPULoadOp stencil_load_op;
    SDL_GPUStoreOp stencil_store_op;
    Uint32 mip_level;
    Uint32 layer;
    bool cycle;
    Uint8 clear_stencil;
} WebGPURenderPassDepthStencilAttachmentInfo;

static WebGPUTextureViewDescription WEBGPU_TextureViewDescription(
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    WebGPUTextureViewUsage usage,
    Uint64 generation,
    Uint32 base_mip_level,
    Uint32 mip_level_count,
    Uint32 base_array_layer,
    Uint32 array_layer_count)
{
    WebGPUTextureViewDescription description;

    description.format = format;
    description.dimension = WEBGPU_ToTextureViewDimension(type, usage, array_layer_count);
    description.aspect = WGPUTextureAspect_All;
    description.usage = usage;
    description.generation = generation;
    description.base_mip_level = base_mip_level;
    description.mip_level_count = mip_level_count;
    description.base_array_layer = base_array_layer;
    description.array_layer_count = array_layer_count;

    return description;
}

static WebGPUTextureViewDescription WEBGPU_TextureViewDescriptionWithDimension(
    SDL_GPUTextureFormat format,
    WGPUTextureViewDimension view_dimension,
    WebGPUTextureViewUsage usage,
    Uint64 generation,
    Uint32 base_mip_level,
    Uint32 mip_level_count,
    Uint32 base_array_layer,
    Uint32 array_layer_count)
{
    WebGPUTextureViewDescription description;

    description.format = format;
    description.dimension = view_dimension;
    description.aspect = WGPUTextureAspect_All;
    description.usage = usage;
    description.generation = generation;
    description.base_mip_level = base_mip_level;
    description.mip_level_count = mip_level_count;
    description.base_array_layer = base_array_layer;
    description.array_layer_count = array_layer_count;

    return description;
}

static void WEBGPU_UseDepthOnlySampledTextureViewDescription(WebGPUTextureViewDescription *description)
{
    if (description->usage != WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED ||
        !WEBGPU_IsStencilFormat(description->format) ||
        description->aspect == WGPUTextureAspect_StencilOnly) {
        return;
    }

    description->aspect = WGPUTextureAspect_DepthOnly;
    if (description->format == SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT) {
        description->format = SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    } else if (description->format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT) {
        description->format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    }
}

static WGPUTextureFormat WEBGPU_TextureViewWGPUFormat(const WebGPUTextureViewDescription *description)
{
    if (description->usage == WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED &&
        description->aspect == WGPUTextureAspect_StencilOnly) {
        return WGPUTextureFormat_Stencil8;
    }
    return WEBGPU_ToWGPUTextureFormat(description->format);
}

static WGPUTextureView WEBGPU_CreateTextureViewFromDescription(
    WGPUTexture texture,
    const WebGPUTextureViewDescription *description)
{
    WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    WGPUTextureUsage wgpu_usage;

    if (description->generation == 0) {
        WEBGPU_SetStringError("invalid texture view generation");
        return NULL;
    }
    if (description->dimension == WGPUTextureViewDimension_Undefined) {
        WEBGPU_SetStringError("invalid texture view dimension");
        return NULL;
    }
    if (description->mip_level_count == 0 || description->array_layer_count == 0) {
        WEBGPU_SetStringError("invalid texture view range");
        return NULL;
    }
    wgpu_usage = WEBGPU_ToTextureViewWGPUUsage(description->usage);
    if (wgpu_usage == WGPUTextureUsage_None) {
        WEBGPU_SetStringError("invalid texture view usage");
        return NULL;
    }

    view_desc.format = WEBGPU_TextureViewWGPUFormat(description);
    view_desc.dimension = description->dimension;
    view_desc.baseMipLevel = description->base_mip_level;
    view_desc.mipLevelCount = description->mip_level_count;
    view_desc.baseArrayLayer = description->base_array_layer;
    view_desc.arrayLayerCount = description->array_layer_count;
    view_desc.aspect = description->aspect;
    view_desc.usage = wgpu_usage;

    return wgpuTextureCreateView(texture, &view_desc);
}

static WebGPUSampledTextureBindingDescription WEBGPU_SampledTextureBindingDescriptionFromTexture(
    WebGPUTexture *texture,
    WGPUTextureView view,
    WebGPUSampler *sampler,
    WGPUTextureViewDimension view_dimension,
    WebGPUTextureViewUsage view_usage)
{
    WebGPUSampledTextureBindingDescription description;

    description.view = view;
    description.sampler = sampler ? sampler->sampler : NULL;
    description.generation = texture->generation;
    description.view_dimension = view_dimension;
    description.format = texture->header.info.format;
    description.sample_count = texture->header.info.sample_count;
    description.view_usage = view_usage;
    description.sampler_binding_type = sampler ? sampler->binding_type : WGPUSamplerBindingType_Undefined;

    return description;
}

static bool WEBGPU_SampledTextureBindingDescriptionsMatch(
    const WebGPUSampledTextureBindingDescription *a,
    const WebGPUSampledTextureBindingDescription *b)
{
    return a->view == b->view &&
           a->sampler == b->sampler &&
           a->generation == b->generation &&
           a->view_dimension == b->view_dimension &&
           a->format == b->format &&
           a->sample_count == b->sample_count &&
           a->view_usage == b->view_usage &&
           a->sampler_binding_type == b->sampler_binding_type;
}

static WebGPUStorageTextureBindingDescription WEBGPU_StorageTextureBindingDescription(
    WebGPUTexture *texture,
    WGPUTextureView view,
    WGPUTextureViewDimension view_dimension,
    WebGPUTextureViewUsage view_usage)
{
    WebGPUStorageTextureBindingDescription description;

    description.view = view;
    description.generation = texture->generation;
    description.view_dimension = view_dimension;
    description.format = texture->header.info.format;
    description.usage = texture->header.info.usage;
    description.view_usage = view_usage;

    return description;
}

static bool WEBGPU_StorageTextureBindingDescriptionsMatch(
    const WebGPUStorageTextureBindingDescription *a,
    const WebGPUStorageTextureBindingDescription *b)
{
    return a->view == b->view &&
           a->generation == b->generation &&
           a->view_dimension == b->view_dimension &&
           a->format == b->format &&
           a->usage == b->usage &&
           a->view_usage == b->view_usage;
}

static bool WEBGPU_Is2DTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY;
}

static bool WEBGPU_IsTransferTextureType(SDL_GPUTextureType type)
{
    return WEBGPU_Is2DTextureType(type) ||
           type == SDL_GPU_TEXTURETYPE_3D ||
           type == SDL_GPU_TEXTURETYPE_CUBE ||
           type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY;
}

static bool WEBGPU_IsBlitTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
           type == SDL_GPU_TEXTURETYPE_3D ||
           type == SDL_GPU_TEXTURETYPE_CUBE ||
           type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY;
}

static bool WEBGPU_IsColorTargetTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
           type == SDL_GPU_TEXTURETYPE_3D ||
           type == SDL_GPU_TEXTURETYPE_CUBE ||
           type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY;
}

static bool WEBGPU_IsDepthStencilTargetTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
           type == SDL_GPU_TEXTURETYPE_CUBE ||
           type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY;
}

static bool WEBGPU_BlitTextureLayerInBounds(const WebGPUTexture *texture, Uint32 mip_level, Uint32 layer_or_depth_plane)
{
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_2D) {
        return layer_or_depth_plane == 0;
    }
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        return layer_or_depth_plane < WEBGPU_TextureMipDimension(texture->header.info.layer_count_or_depth, mip_level);
    }
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
        texture->header.info.type == SDL_GPU_TEXTURETYPE_CUBE ||
        texture->header.info.type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        return layer_or_depth_plane < texture->header.info.layer_count_or_depth;
    }
    return false;
}

static bool WEBGPU_TextureLayerRangeInBounds(const WebGPUTexture *texture, Uint32 layer, Uint32 depth)
{
    return layer <= texture->header.info.layer_count_or_depth &&
           depth <= texture->header.info.layer_count_or_depth - layer;
}

static Uint32 WEBGPU_TextureMipLayerCountOrDepth(const WebGPUTexture *texture, Uint32 mip_level)
{
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        return WEBGPU_TextureMipDimension(texture->header.info.layer_count_or_depth, mip_level);
    }
    return texture->header.info.layer_count_or_depth;
}

static bool WEBGPU_TextureRegionDepthInBounds(const WebGPUTexture *texture, Uint32 mip_level, Uint32 layer, Uint32 z, Uint32 depth)
{
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        const Uint32 mip_depth = WEBGPU_TextureMipLayerCountOrDepth(texture, mip_level);
        return layer == 0 &&
               z <= mip_depth &&
               depth <= mip_depth - z;
    }

    return z == 0 &&
           WEBGPU_TextureLayerRangeInBounds(texture, layer, depth);
}

static bool WEBGPU_TextureRenderLayerInBounds(const WebGPUTexture *texture, Uint32 mip_level, Uint32 layer_or_depth_plane)
{
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        return WEBGPU_TextureRegionDepthInBounds(texture, mip_level, 0, layer_or_depth_plane, 1);
    }
    return WEBGPU_TextureLayerRangeInBounds(texture, layer_or_depth_plane, 1);
}

static bool WEBGPU_RenderAttachmentRegionsOverlap(
    const WebGPURenderAttachmentRegion *a,
    const WebGPURenderAttachmentRegion *b)
{
    return a->texture == b->texture &&
           a->mip_level == b->mip_level &&
           a->layer_or_depth_plane == b->layer_or_depth_plane;
}

static bool WEBGPU_RenderAttachmentRegionIsAlreadyUsed(
    const WebGPURenderAttachmentRegion *regions,
    Uint32 region_count,
    const WebGPURenderAttachmentRegion *region)
{
    for (Uint32 i = 0; i < region_count; i += 1) {
        if (WEBGPU_RenderAttachmentRegionsOverlap(&regions[i], region)) {
            return true;
        }
    }
    return false;
}

static bool WEBGPU_RecordRenderPassAttachmentExtent(
    Uint32 *render_pass_attachment_width,
    Uint32 *render_pass_attachment_height,
    Uint32 attachment_width,
    Uint32 attachment_height)
{
    if (*render_pass_attachment_width == 0) {
        *render_pass_attachment_width = attachment_width;
        *render_pass_attachment_height = attachment_height;
        return true;
    }
    return *render_pass_attachment_width == attachment_width &&
           *render_pass_attachment_height == attachment_height;
}

static bool WEBGPU_ViewportFitsLimits(WebGPURenderer *renderer, const SDL_GPUViewport *viewport)
{
    const double max_texture_dimension_2d = (double)renderer->limits.maxTextureDimension2D;
    const double max_viewport_range = max_texture_dimension_2d * 2.0;
    const double x = (double)viewport->x;
    const double y = (double)viewport->y;
    const double w = (double)viewport->w;
    const double h = (double)viewport->h;

    return x >= -max_viewport_range &&
           y >= -max_viewport_range &&
           w <= max_texture_dimension_2d &&
           h <= max_texture_dimension_2d &&
           x + w <= max_viewport_range - 1.0 &&
           y + h <= max_viewport_range - 1.0;
}

static Uint32 WEBGPU_TextureCopyOriginZ(const WebGPUTexture *texture, Uint32 layer, Uint32 z)
{
    return texture->header.info.type == SDL_GPU_TEXTURETYPE_3D ? z : layer;
}

typedef struct WebGPUTextureCopyEndpoint
{
    WebGPUTexture *texture;
    Uint32 mip_level;
    Uint32 layer;
    Uint32 x;
    Uint32 y;
    Uint32 z;
    Uint32 mip_width;
    Uint32 mip_height;
    Uint32 origin_z;
} WebGPUTextureCopyEndpoint;

static void WEBGPU_InitTextureCopyEndpointFields(
    WebGPUTexture *texture,
    Uint32 mip_level,
    Uint32 layer,
    Uint32 x,
    Uint32 y,
    Uint32 z,
    WebGPUTextureCopyEndpoint *endpoint)
{
    endpoint->texture = texture;
    endpoint->mip_level = mip_level;
    endpoint->layer = layer;
    endpoint->x = x;
    endpoint->y = y;
    endpoint->z = z;
    endpoint->mip_width = 0;
    endpoint->mip_height = 0;
    endpoint->origin_z = 0;
}

static void WEBGPU_InitTextureCopyEndpointFromLocation(
    const SDL_GPUTextureLocation *location,
    WebGPUTextureCopyEndpoint *endpoint)
{
    WEBGPU_InitTextureCopyEndpointFields(
        (WebGPUTexture *)location->texture,
        location->mip_level,
        location->layer,
        location->x,
        location->y,
        location->z,
        endpoint);
}

static void WEBGPU_InitTextureCopyEndpointFromRegion(
    const SDL_GPUTextureRegion *region,
    WebGPUTextureCopyEndpoint *endpoint)
{
    WEBGPU_InitTextureCopyEndpointFields(
        (WebGPUTexture *)region->texture,
        region->mip_level,
        region->layer,
        region->x,
        region->y,
        region->z,
        endpoint);
}

static bool WEBGPU_ValidateTextureCopyEndpointDepth(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUTextureCopyEndpoint *endpoint,
    Uint32 depth,
    const char *error)
{
    if (!WEBGPU_TextureRegionDepthInBounds(endpoint->texture, endpoint->mip_level, endpoint->layer, endpoint->z, depth)) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    return true;
}

static bool WEBGPU_ResolveTextureCopyEndpointRegion(
    WebGPUCommandBuffer *command_buffer,
    WebGPUTextureCopyEndpoint *endpoint,
    Uint32 width,
    Uint32 height,
    const char *error)
{
    endpoint->mip_width = WEBGPU_TextureMipDimension(endpoint->texture->header.info.width, endpoint->mip_level);
    endpoint->mip_height = WEBGPU_TextureMipDimension(endpoint->texture->header.info.height, endpoint->mip_level);
    endpoint->origin_z = WEBGPU_TextureCopyOriginZ(endpoint->texture, endpoint->layer, endpoint->z);

    if (endpoint->x > endpoint->mip_width ||
        endpoint->y > endpoint->mip_height ||
        width > endpoint->mip_width - endpoint->x ||
        height > endpoint->mip_height - endpoint->y) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    return true;
}

static bool WEBGPU_InitTextureCopySize(
    WebGPUCommandBuffer *command_buffer,
    SDL_GPUTextureFormat format,
    Uint32 width,
    Uint32 height,
    Uint32 depth,
    const char *error,
    WGPUExtent3D *copy_size)
{
    Uint64 physical_width = width;
    Uint64 physical_height = height;

    if (WEBGPU_TextureFormatIsCompressed(format)) {
        Uint32 block_width = WEBGPU_TextureFormatBlockWidth(format);
        Uint32 block_height = WEBGPU_TextureFormatBlockHeight(format);

        physical_width = WEBGPU_TextureBlockCount(width, block_width) * block_width;
        physical_height = WEBGPU_TextureBlockCount(height, block_height) * block_height;
    }

    if (physical_width > SDL_MAX_UINT32 || physical_height > SDL_MAX_UINT32) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    copy_size->width = (Uint32)physical_width;
    copy_size->height = (Uint32)physical_height;
    copy_size->depthOrArrayLayers = depth;
    return true;
}

static bool WEBGPU_TextureCopyEndpointsDisjoint(
    const WebGPUTextureCopyEndpoint *source,
    const WebGPUTextureCopyEndpoint *destination,
    Uint32 depth)
{
    if (source->mip_level != destination->mip_level) {
        return true;
    }

    if (source->texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        return false;
    }

    return source->origin_z + depth <= destination->origin_z ||
           destination->origin_z + depth <= source->origin_z;
}

typedef struct WebGPUTextureBlitEndpoint
{
    WebGPUTexture *texture;
    Uint32 mip_level;
    Uint32 layer_or_depth_plane;
    Uint32 x;
    Uint32 y;
    Uint32 w;
    Uint32 h;
    Uint32 mip_width;
    Uint32 mip_height;
    Uint32 mip_depth;
    Uint32 view_base_array_layer;
    SDL_GPUTextureType view_type;
    bool is_3d;
} WebGPUTextureBlitEndpoint;

static void WEBGPU_InitTextureBlitEndpoint(
    const SDL_GPUBlitRegion *region,
    WebGPUTextureBlitEndpoint *endpoint)
{
    endpoint->texture = (WebGPUTexture *)region->texture;
    endpoint->mip_level = region->mip_level;
    endpoint->layer_or_depth_plane = region->layer_or_depth_plane;
    endpoint->x = region->x;
    endpoint->y = region->y;
    endpoint->w = region->w;
    endpoint->h = region->h;
    endpoint->mip_width = 0;
    endpoint->mip_height = 0;
    endpoint->mip_depth = 0;
    endpoint->view_base_array_layer = 0;
    endpoint->view_type = SDL_GPU_TEXTURETYPE_2D;
    endpoint->is_3d = false;
}

static bool WEBGPU_ValidateTextureBlitEndpointMip(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUTextureBlitEndpoint *endpoint,
    const char *error)
{
    if (endpoint->mip_level >= endpoint->texture->header.info.num_levels) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    return true;
}

static bool WEBGPU_ValidateTextureBlitEndpointLayer(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUTextureBlitEndpoint *endpoint,
    const char *error)
{
    if (!WEBGPU_BlitTextureLayerInBounds(endpoint->texture, endpoint->mip_level, endpoint->layer_or_depth_plane)) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    return true;
}

static bool WEBGPU_ResolveTextureBlitEndpointRegion(
    WebGPUCommandBuffer *command_buffer,
    WebGPUTextureBlitEndpoint *endpoint,
    const char *error)
{
    endpoint->is_3d = endpoint->texture->header.info.type == SDL_GPU_TEXTURETYPE_3D;
    endpoint->view_type = endpoint->is_3d ? SDL_GPU_TEXTURETYPE_3D : SDL_GPU_TEXTURETYPE_2D;
    endpoint->mip_width = WEBGPU_TextureMipDimension(endpoint->texture->header.info.width, endpoint->mip_level);
    endpoint->mip_height = WEBGPU_TextureMipDimension(endpoint->texture->header.info.height, endpoint->mip_level);
    endpoint->mip_depth = WEBGPU_TextureMipLayerCountOrDepth(endpoint->texture, endpoint->mip_level);
    endpoint->view_base_array_layer = endpoint->is_3d ? 0 : endpoint->layer_or_depth_plane;

    if (endpoint->x > endpoint->mip_width ||
        endpoint->y > endpoint->mip_height ||
        endpoint->w > endpoint->mip_width - endpoint->x ||
        endpoint->h > endpoint->mip_height - endpoint->y) {
        WEBGPU_FailCommandBuffer(command_buffer, error);
        return false;
    }

    return true;
}

static void WEBGPU_ReleaseTextureViews(WGPUTextureView *views, Uint32 count)
{
    for (Uint32 i = 0; i < count; i += 1) {
        if (views[i]) {
            wgpuTextureViewRelease(views[i]);
            views[i] = NULL;
        }
    }
}

static void WEBGPU_ReleaseTextures(WGPUTexture *textures, Uint32 count)
{
    for (Uint32 i = 0; i < count; i += 1) {
        if (textures[i]) {
            wgpuTextureRelease(textures[i]);
            textures[i] = NULL;
        }
    }
}

static WGPUAddressMode WEBGPU_ToAddressMode(SDL_GPUSamplerAddressMode address_mode)
{
    switch (address_mode) {
    case SDL_GPU_SAMPLERADDRESSMODE_REPEAT:
        return WGPUAddressMode_Repeat;
    case SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT:
        return WGPUAddressMode_MirrorRepeat;
    case SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE:
        return WGPUAddressMode_ClampToEdge;
    }
    return WGPUAddressMode_Undefined;
}

static WGPUFilterMode WEBGPU_ToFilterMode(SDL_GPUFilter filter)
{
    switch (filter) {
    case SDL_GPU_FILTER_NEAREST:
        return WGPUFilterMode_Nearest;
    case SDL_GPU_FILTER_LINEAR:
        return WGPUFilterMode_Linear;
    }
    return WGPUFilterMode_Undefined;
}

static WGPUMipmapFilterMode WEBGPU_ToMipmapFilterMode(SDL_GPUSamplerMipmapMode mipmap_mode)
{
    switch (mipmap_mode) {
    case SDL_GPU_SAMPLERMIPMAPMODE_NEAREST:
        return WGPUMipmapFilterMode_Nearest;
    case SDL_GPU_SAMPLERMIPMAPMODE_LINEAR:
        return WGPUMipmapFilterMode_Linear;
    }
    return WGPUMipmapFilterMode_Undefined;
}

static bool WEBGPU_ToPresentMode(SDL_GPUPresentMode present_mode, WGPUPresentMode *wgpu_present_mode)
{
    switch (present_mode) {
    case SDL_GPU_PRESENTMODE_VSYNC:
        *wgpu_present_mode = WGPUPresentMode_Fifo;
        return true;
    case SDL_GPU_PRESENTMODE_IMMEDIATE:
    case SDL_GPU_PRESENTMODE_MAILBOX:
        return false;
    }
    return false;
}

static bool WEBGPU_ToCompositeAlphaMode(SDL_GPUSwapchainComposition composition, WGPUCompositeAlphaMode *alpha_mode)
{
    switch (composition) {
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR:
    case SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR:
        *alpha_mode = WGPUCompositeAlphaMode_Auto;
        return true;
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR_EXTENDED_LINEAR:
    case SDL_GPU_SWAPCHAINCOMPOSITION_HDR10_ST2084:
        return false;
    }
    return false;
}

static WGPULoadOp WEBGPU_ToLoadOp(SDL_GPULoadOp load_op)
{
    switch (load_op) {
    case SDL_GPU_LOADOP_LOAD:
        return WGPULoadOp_Load;
    case SDL_GPU_LOADOP_CLEAR:
        return WGPULoadOp_Clear;
    case SDL_GPU_LOADOP_DONT_CARE:
        return WGPULoadOp_Clear;
    }
    return WGPULoadOp_Undefined;
}

static WGPUStoreOp WEBGPU_ToStoreOp(SDL_GPUStoreOp store_op)
{
    switch (store_op) {
    case SDL_GPU_STOREOP_STORE:
        return WGPUStoreOp_Store;
    case SDL_GPU_STOREOP_DONT_CARE:
        return WGPUStoreOp_Discard;
    case SDL_GPU_STOREOP_RESOLVE:
        return WGPUStoreOp_Discard;
    case SDL_GPU_STOREOP_RESOLVE_AND_STORE:
        return WGPUStoreOp_Store;
    }
    return WGPUStoreOp_Undefined;
}

static Uint32 WEBGPU_ToSampleCount(SDL_GPUSampleCount sample_count)
{
    switch (sample_count) {
    case SDL_GPU_SAMPLECOUNT_1:
        return 1;
    case SDL_GPU_SAMPLECOUNT_4:
        return 4;
    case SDL_GPU_SAMPLECOUNT_2:
    case SDL_GPU_SAMPLECOUNT_8:
    default:
        return 0;
    }
}

static WGPUPrimitiveTopology WEBGPU_ToPrimitiveTopology(SDL_GPUPrimitiveType primitive_type)
{
    switch (primitive_type) {
    case SDL_GPU_PRIMITIVETYPE_TRIANGLELIST:
        return WGPUPrimitiveTopology_TriangleList;
    case SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP:
        return WGPUPrimitiveTopology_TriangleStrip;
    case SDL_GPU_PRIMITIVETYPE_LINELIST:
        return WGPUPrimitiveTopology_LineList;
    case SDL_GPU_PRIMITIVETYPE_LINESTRIP:
        return WGPUPrimitiveTopology_LineStrip;
    case SDL_GPU_PRIMITIVETYPE_POINTLIST:
        return WGPUPrimitiveTopology_PointList;
    }
    return WGPUPrimitiveTopology_Undefined;
}

static bool WEBGPU_PrimitiveTypeSupportsDepthBias(SDL_GPUPrimitiveType primitive_type)
{
    return primitive_type == SDL_GPU_PRIMITIVETYPE_TRIANGLELIST ||
           primitive_type == SDL_GPU_PRIMITIVETYPE_TRIANGLESTRIP;
}

static bool WEBGPU_InitDepthBiasState(
    const WebGPURenderer *renderer,
    WGPUDepthStencilState *depth_stencil_state,
    const SDL_GPURasterizerState *rasterizer_state,
    SDL_GPUPrimitiveType primitive_type)
{
    const float constant_factor = rasterizer_state->depth_bias_constant_factor;
    const float slope_factor = rasterizer_state->depth_bias_slope_factor;
    const float clamp = rasterizer_state->depth_bias_clamp;

    if (!rasterizer_state->enable_depth_bias) {
        return true;
    }

    if (SDL_isnanf(constant_factor) || SDL_isinff(constant_factor) ||
        SDL_isnanf(slope_factor) || SDL_isinff(slope_factor) ||
        SDL_isnanf(clamp) || SDL_isinff(clamp)) {
        WEBGPU_SetStringError("invalid depth bias state");
        return false;
    }

    if ((double)constant_factor < (double)SDL_MIN_SINT32 ||
        (double)constant_factor > (double)SDL_MAX_SINT32) {
        WEBGPU_SetStringError("depth bias constant factor is outside WebGPU int32 range");
        return false;
    }

    if (!WEBGPU_PrimitiveTypeSupportsDepthBias(primitive_type) &&
        (constant_factor != 0.0f || slope_factor != 0.0f || clamp != 0.0f)) {
        WEBGPU_SetStringError("nonzero depth bias is only supported for triangle primitives");
        return false;
    }

    if (clamp != 0.0f && !renderer->supports_core_features_and_limits) {
        WEBGPU_SetStringError("depth bias clamp requires WebGPU core-features-and-limits");
        return false;
    }

    depth_stencil_state->depthBias = (int32_t)SDL_lroundf(constant_factor);
    depth_stencil_state->depthBiasSlopeScale = slope_factor;
    depth_stencil_state->depthBiasClamp = clamp;
    return true;
}

static WGPUFrontFace WEBGPU_ToFrontFace(SDL_GPUFrontFace front_face)
{
    switch (front_face) {
    case SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE:
        return WGPUFrontFace_CCW;
    case SDL_GPU_FRONTFACE_CLOCKWISE:
        return WGPUFrontFace_CW;
    default:
        return WGPUFrontFace_Undefined;
    }
}

static WGPUCullMode WEBGPU_ToCullMode(SDL_GPUCullMode cull_mode)
{
    switch (cull_mode) {
    case SDL_GPU_CULLMODE_NONE:
        return WGPUCullMode_None;
    case SDL_GPU_CULLMODE_FRONT:
        return WGPUCullMode_Front;
    case SDL_GPU_CULLMODE_BACK:
        return WGPUCullMode_Back;
    default:
        return WGPUCullMode_Undefined;
    }
}

static bool WEBGPU_IsValidFillMode(SDL_GPUFillMode fill_mode)
{
    switch (fill_mode) {
    case SDL_GPU_FILLMODE_FILL:
    case SDL_GPU_FILLMODE_LINE:
        return true;
    default:
        return false;
    }
}

static WGPUCompareFunction WEBGPU_ToCompareFunction(SDL_GPUCompareOp compare_op)
{
    switch (compare_op) {
    case SDL_GPU_COMPAREOP_NEVER:
        return WGPUCompareFunction_Never;
    case SDL_GPU_COMPAREOP_LESS:
        return WGPUCompareFunction_Less;
    case SDL_GPU_COMPAREOP_EQUAL:
        return WGPUCompareFunction_Equal;
    case SDL_GPU_COMPAREOP_LESS_OR_EQUAL:
        return WGPUCompareFunction_LessEqual;
    case SDL_GPU_COMPAREOP_GREATER:
        return WGPUCompareFunction_Greater;
    case SDL_GPU_COMPAREOP_NOT_EQUAL:
        return WGPUCompareFunction_NotEqual;
    case SDL_GPU_COMPAREOP_GREATER_OR_EQUAL:
        return WGPUCompareFunction_GreaterEqual;
    case SDL_GPU_COMPAREOP_ALWAYS:
        return WGPUCompareFunction_Always;
    case SDL_GPU_COMPAREOP_INVALID:
    default:
        return WGPUCompareFunction_Undefined;
    }
}

static WGPUStencilOperation WEBGPU_ToStencilOperation(SDL_GPUStencilOp stencil_op)
{
    switch (stencil_op) {
    case SDL_GPU_STENCILOP_KEEP:
        return WGPUStencilOperation_Keep;
    case SDL_GPU_STENCILOP_ZERO:
        return WGPUStencilOperation_Zero;
    case SDL_GPU_STENCILOP_REPLACE:
        return WGPUStencilOperation_Replace;
    case SDL_GPU_STENCILOP_INCREMENT_AND_CLAMP:
        return WGPUStencilOperation_IncrementClamp;
    case SDL_GPU_STENCILOP_DECREMENT_AND_CLAMP:
        return WGPUStencilOperation_DecrementClamp;
    case SDL_GPU_STENCILOP_INVERT:
        return WGPUStencilOperation_Invert;
    case SDL_GPU_STENCILOP_INCREMENT_AND_WRAP:
        return WGPUStencilOperation_IncrementWrap;
    case SDL_GPU_STENCILOP_DECREMENT_AND_WRAP:
        return WGPUStencilOperation_DecrementWrap;
    case SDL_GPU_STENCILOP_INVALID:
    default:
        return WGPUStencilOperation_Undefined;
    }
}

static bool WEBGPU_InitStencilFaceState(WGPUStencilFaceState *dst, const SDL_GPUStencilOpState *src)
{
    *dst = (WGPUStencilFaceState)WGPU_STENCIL_FACE_STATE_INIT;
    dst->compare = WEBGPU_ToCompareFunction(src->compare_op);
    dst->failOp = WEBGPU_ToStencilOperation(src->fail_op);
    dst->depthFailOp = WEBGPU_ToStencilOperation(src->depth_fail_op);
    dst->passOp = WEBGPU_ToStencilOperation(src->pass_op);

    if (dst->compare == WGPUCompareFunction_Undefined) {
        WEBGPU_SetStringError("invalid stencil compare operation");
        return false;
    }
    if (dst->failOp == WGPUStencilOperation_Undefined) {
        WEBGPU_SetStringError("invalid stencil fail operation");
        return false;
    }
    if (dst->depthFailOp == WGPUStencilOperation_Undefined) {
        WEBGPU_SetStringError("invalid stencil depth-fail operation");
        return false;
    }
    if (dst->passOp == WGPUStencilOperation_Undefined) {
        WEBGPU_SetStringError("invalid stencil pass operation");
        return false;
    }
    return true;
}

static WGPUBlendFactor WEBGPU_ToBlendFactor(SDL_GPUBlendFactor blend_factor)
{
    switch (blend_factor) {
    case SDL_GPU_BLENDFACTOR_ZERO:
        return WGPUBlendFactor_Zero;
    case SDL_GPU_BLENDFACTOR_ONE:
        return WGPUBlendFactor_One;
    case SDL_GPU_BLENDFACTOR_SRC_COLOR:
        return WGPUBlendFactor_Src;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_COLOR:
        return WGPUBlendFactor_OneMinusSrc;
    case SDL_GPU_BLENDFACTOR_DST_COLOR:
        return WGPUBlendFactor_Dst;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_COLOR:
        return WGPUBlendFactor_OneMinusDst;
    case SDL_GPU_BLENDFACTOR_SRC_ALPHA:
        return WGPUBlendFactor_SrcAlpha;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA:
        return WGPUBlendFactor_OneMinusSrcAlpha;
    case SDL_GPU_BLENDFACTOR_DST_ALPHA:
        return WGPUBlendFactor_DstAlpha;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_DST_ALPHA:
        return WGPUBlendFactor_OneMinusDstAlpha;
    case SDL_GPU_BLENDFACTOR_CONSTANT_COLOR:
        return WGPUBlendFactor_Constant;
    case SDL_GPU_BLENDFACTOR_ONE_MINUS_CONSTANT_COLOR:
        return WGPUBlendFactor_OneMinusConstant;
    case SDL_GPU_BLENDFACTOR_SRC_ALPHA_SATURATE:
        return WGPUBlendFactor_SrcAlphaSaturated;
    case SDL_GPU_BLENDFACTOR_INVALID:
    default:
        return WGPUBlendFactor_Undefined;
    }
}

static WGPUBlendOperation WEBGPU_ToBlendOperation(SDL_GPUBlendOp blend_op)
{
    switch (blend_op) {
    case SDL_GPU_BLENDOP_ADD:
        return WGPUBlendOperation_Add;
    case SDL_GPU_BLENDOP_SUBTRACT:
        return WGPUBlendOperation_Subtract;
    case SDL_GPU_BLENDOP_REVERSE_SUBTRACT:
        return WGPUBlendOperation_ReverseSubtract;
    case SDL_GPU_BLENDOP_MIN:
        return WGPUBlendOperation_Min;
    case SDL_GPU_BLENDOP_MAX:
        return WGPUBlendOperation_Max;
    case SDL_GPU_BLENDOP_INVALID:
    default:
        return WGPUBlendOperation_Undefined;
    }
}

static void WEBGPU_NormalizeBlendComponent(WGPUBlendComponent *component)
{
    /* WebGPU requires one/one factors for min/max; SDL/native APIs ignore them. */
    if (component->operation == WGPUBlendOperation_Min ||
        component->operation == WGPUBlendOperation_Max) {
        component->srcFactor = WGPUBlendFactor_One;
        component->dstFactor = WGPUBlendFactor_One;
    }
}

static WGPUColorWriteMask WEBGPU_ToColorWriteMask(SDL_GPUColorComponentFlags color_write_mask)
{
    WGPUColorWriteMask result = WGPUColorWriteMask_None;

    if (color_write_mask & SDL_GPU_COLORCOMPONENT_R) {
        result |= WGPUColorWriteMask_Red;
    }
    if (color_write_mask & SDL_GPU_COLORCOMPONENT_G) {
        result |= WGPUColorWriteMask_Green;
    }
    if (color_write_mask & SDL_GPU_COLORCOMPONENT_B) {
        result |= WGPUColorWriteMask_Blue;
    }
    if (color_write_mask & SDL_GPU_COLORCOMPONENT_A) {
        result |= WGPUColorWriteMask_Alpha;
    }

    return result;
}

static bool WEBGPU_IsValidColorWriteMask(SDL_GPUColorComponentFlags color_write_mask)
{
    const Uint32 valid_mask =
        SDL_GPU_COLORCOMPONENT_R |
        SDL_GPU_COLORCOMPONENT_G |
        SDL_GPU_COLORCOMPONENT_B |
        SDL_GPU_COLORCOMPONENT_A;

    return (((Uint32)color_write_mask) & ~valid_mask) == 0;
}

static WGPUBufferUsage WEBGPU_ToBufferUsage(SDL_GPUBufferUsageFlags usage)
{
    // SDL_GPU copy operations can use created buffers as copy sources or
    // destinations without declaring WebGPU-style transfer usage upfront.
    WGPUBufferUsage result = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

    if (usage & SDL_GPU_BUFFERUSAGE_VERTEX) {
        result |= WGPUBufferUsage_Vertex;
    }
    if (usage & SDL_GPU_BUFFERUSAGE_INDEX) {
        result |= WGPUBufferUsage_Index;
    }
    if (usage & SDL_GPU_BUFFERUSAGE_INDIRECT) {
        result |= WGPUBufferUsage_Indirect;
    }
    if (usage & SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ) {
        result |= WGPUBufferUsage_Storage;
    }
    if (usage & (SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ |
                 SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE)) {
        result |= WGPUBufferUsage_Storage;
    }

    return result;
}

static bool WEBGPU_IsValidBufferUsage(SDL_GPUBufferUsageFlags usage)
{
    const SDL_GPUBufferUsageFlags valid_usage =
        SDL_GPU_BUFFERUSAGE_VERTEX |
        SDL_GPU_BUFFERUSAGE_INDEX |
        SDL_GPU_BUFFERUSAGE_INDIRECT |
        SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ |
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ |
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE;

    return usage != 0 && (usage & ~valid_usage) == 0;
}

static WGPUVertexFormat WEBGPU_ToVertexFormat(SDL_GPUVertexElementFormat format)
{
    switch (format) {
    case SDL_GPU_VERTEXELEMENTFORMAT_INT:
        return WGPUVertexFormat_Sint32;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT2:
        return WGPUVertexFormat_Sint32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT3:
        return WGPUVertexFormat_Sint32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT4:
        return WGPUVertexFormat_Sint32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT:
        return WGPUVertexFormat_Uint32;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT2:
        return WGPUVertexFormat_Uint32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT3:
        return WGPUVertexFormat_Uint32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT4:
        return WGPUVertexFormat_Uint32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT:
        return WGPUVertexFormat_Float32;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2:
        return WGPUVertexFormat_Float32x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3:
        return WGPUVertexFormat_Float32x3;
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4:
        return WGPUVertexFormat_Float32x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2:
        return WGPUVertexFormat_Sint8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4:
        return WGPUVertexFormat_Sint8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2:
        return WGPUVertexFormat_Uint8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4:
        return WGPUVertexFormat_Uint8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2_NORM:
        return WGPUVertexFormat_Snorm8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4_NORM:
        return WGPUVertexFormat_Snorm8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2_NORM:
        return WGPUVertexFormat_Unorm8x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM:
        return WGPUVertexFormat_Unorm8x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2:
        return WGPUVertexFormat_Sint16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4:
        return WGPUVertexFormat_Sint16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2:
        return WGPUVertexFormat_Uint16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4:
        return WGPUVertexFormat_Uint16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2_NORM:
        return WGPUVertexFormat_Snorm16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4_NORM:
        return WGPUVertexFormat_Snorm16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2_NORM:
        return WGPUVertexFormat_Unorm16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4_NORM:
        return WGPUVertexFormat_Unorm16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF2:
        return WGPUVertexFormat_Float16x2;
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF4:
        return WGPUVertexFormat_Float16x4;
    case SDL_GPU_VERTEXELEMENTFORMAT_INVALID:
    default:
        return WGPUVertexFormat_Force32;
    }
}

static Uint32 WEBGPU_VertexElementFormatByteSize(SDL_GPUVertexElementFormat format)
{
    switch (format) {
    case SDL_GPU_VERTEXELEMENTFORMAT_INT:
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT:
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT:
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4:
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4:
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE4_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2:
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2:
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT2_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT2_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF2:
        return 4;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT2:
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT2:
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2:
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4:
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4:
    case SDL_GPU_VERTEXELEMENTFORMAT_SHORT4_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_USHORT4_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_HALF4:
        return 8;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT3:
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT3:
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3:
        return 12;
    case SDL_GPU_VERTEXELEMENTFORMAT_INT4:
    case SDL_GPU_VERTEXELEMENTFORMAT_UINT4:
    case SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4:
        return 16;
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2:
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2:
    case SDL_GPU_VERTEXELEMENTFORMAT_BYTE2_NORM:
    case SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2_NORM:
        return 2;
    case SDL_GPU_VERTEXELEMENTFORMAT_INVALID:
    default:
        return 0;
    }
}

static Uint32 WEBGPU_VertexElementFormatAlignment(SDL_GPUVertexElementFormat format)
{
    Uint32 byte_size = WEBGPU_VertexElementFormatByteSize(format);
    return SDL_min(byte_size, 4);
}

static WGPUVertexStepMode WEBGPU_ToVertexStepMode(SDL_GPUVertexInputRate input_rate)
{
    switch (input_rate) {
    case SDL_GPU_VERTEXINPUTRATE_VERTEX:
        return WGPUVertexStepMode_Vertex;
    case SDL_GPU_VERTEXINPUTRATE_INSTANCE:
        return WGPUVertexStepMode_Instance;
    default:
        return WGPUVertexStepMode_Undefined;
    }
}

static WGPUIndexFormat WEBGPU_ToIndexFormat(SDL_GPUIndexElementSize index_element_size)
{
    switch (index_element_size) {
    case SDL_GPU_INDEXELEMENTSIZE_16BIT:
        return WGPUIndexFormat_Uint16;
    case SDL_GPU_INDEXELEMENTSIZE_32BIT:
        return WGPUIndexFormat_Uint32;
    default:
        return WGPUIndexFormat_Undefined;
    }
}

static bool WEBGPU_SurfaceStatusOK(WGPUSurfaceGetCurrentTextureStatus status)
{
    return status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
           status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
}

static void WEBGPU_ReleaseTextureHandles(WebGPUTexture *texture)
{
    if (!texture) {
        return;
    }
    if (texture->storage_view) {
        wgpuTextureViewRelease(texture->storage_view);
        texture->storage_view = NULL;
    }
    if (texture->view) {
        wgpuTextureViewRelease(texture->view);
        texture->view = NULL;
    }
    if (texture->texture) {
        wgpuTextureRelease(texture->texture);
        texture->texture = NULL;
    }
}

static void WEBGPU_DestroyWindowData(WebGPUWindowData *window_data)
{
    if (!window_data) {
        return;
    }

    SDL_assert(window_data->refcount == 0);
    SDL_assert(window_data->swapchain_proxy_refcount == 0);
    if (window_data->surface) {
        if (window_data->configured) {
            wgpuSurfaceUnconfigure(window_data->surface);
        }
        wgpuSurfaceRelease(window_data->surface);
    }
    SDL_free(window_data->canvas_selector);
    SDL_free(window_data);
}

static void WEBGPU_MaybeDestroyWindowData(WebGPUWindowData *window_data)
{
    if (!window_data) {
        return;
    }
    if (window_data->refcount == 0 && window_data->swapchain_proxy_refcount == 0) {
        WEBGPU_DestroyWindowData(window_data);
    }
}

static bool WEBGPU_RetainSwapchainWindowData(WebGPUWindowData *window_data)
{
    if (!WEBGPU_WindowDataIsActive(window_data)) {
        return SDL_SetError("Cannot retain swapchain texture for an inactive WebGPU window");
    }
    if (window_data->swapchain_proxy_refcount == SDL_MAX_UINT32) {
        return SDL_SetError("Too many pending WebGPU swapchain textures");
    }
    window_data->swapchain_proxy_refcount += 1;
    return true;
}

static void WEBGPU_ReleaseSwapchainWindowData(WebGPUWindowData *window_data)
{
    if (!window_data) {
        return;
    }

    SDL_assert(window_data->swapchain_proxy_refcount > 0);
    if (window_data->swapchain_proxy_refcount > 0) {
        window_data->swapchain_proxy_refcount -= 1;
    }
    WEBGPU_MaybeDestroyWindowData(window_data);
}

static void WEBGPU_ReleaseSwapchainTexture(WebGPUTexture *texture)
{
    if (!texture) {
        return;
    }
    WEBGPU_ReleaseTextureHandles(texture);
    WEBGPU_ReleaseSwapchainWindowData(texture->window_data);
    texture->window_data = NULL;
    SDL_free(texture);
}

static void WEBGPU_DestroyTexture(WebGPUTexture *texture)
{
    if (!texture) {
        return;
    }
    WEBGPU_ReleaseTextureHandles(texture);
    SDL_GPUTextureHeaderDestroy(&texture->header);
    SDL_free(texture);
}

static void WEBGPU_ReleaseTextureReference(WebGPUTexture *texture)
{
    if (!texture) {
        return;
    }
    SDL_assert(texture->refcount > 0);
    texture->refcount -= 1;
    if (texture->refcount == 0) {
        WEBGPU_DestroyTexture(texture);
    }
}

static void WEBGPU_DestroyBuffer(WebGPUBuffer *buffer)
{
    if (!buffer) {
        return;
    }
    if (buffer->buffer) {
        wgpuBufferRelease(buffer->buffer);
        buffer->buffer = NULL;
    }
    SDL_free(buffer);
}

static void WEBGPU_ReleaseBufferReference(WebGPUBuffer *buffer)
{
    if (!buffer) {
        return;
    }
    SDL_assert(buffer->refcount > 0);
    buffer->refcount -= 1;
    if (buffer->refcount == 0) {
        WEBGPU_DestroyBuffer(buffer);
    }
}

static void WEBGPU_DestroyShader(WebGPUShader *shader)
{
    if (!shader) {
        return;
    }
    if (shader->module) {
        wgpuShaderModuleRelease(shader->module);
    }
    SDL_free(shader->entrypoint);
    SDL_free(shader);
}

static void WEBGPU_DestroyGraphicsPipeline(WebGPUGraphicsPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    for (Uint32 i = 0; i < pipeline->bind_group_layout_count; i += 1) {
        if (pipeline->empty_bind_groups[i]) {
            wgpuBindGroupRelease(pipeline->empty_bind_groups[i]);
        }
    }
    for (Uint32 i = 0; i < pipeline->bind_group_layout_count; i += 1) {
        if (pipeline->bind_group_layouts[i]) {
            wgpuBindGroupLayoutRelease(pipeline->bind_group_layouts[i]);
        }
    }
    if (pipeline->pipeline) {
        wgpuRenderPipelineRelease(pipeline->pipeline);
    }
    SDL_free(pipeline);
}

static void WEBGPU_DestroyComputePipeline(WebGPUComputePipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    for (Uint32 i = 0; i < pipeline->bind_group_layout_count; i += 1) {
        if (pipeline->empty_bind_groups[i]) {
            wgpuBindGroupRelease(pipeline->empty_bind_groups[i]);
        }
    }
    for (Uint32 i = 0; i < pipeline->bind_group_layout_count; i += 1) {
        if (pipeline->bind_group_layouts[i]) {
            wgpuBindGroupLayoutRelease(pipeline->bind_group_layouts[i]);
        }
    }
    if (pipeline->pipeline) {
        wgpuComputePipelineRelease(pipeline->pipeline);
    }
    SDL_free(pipeline);
}

static void WEBGPU_RetainGraphicsPipelineReference(WebGPUGraphicsPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    SDL_assert(SDL_GetAtomicInt(&pipeline->refcount) > 0);
    (void)SDL_AtomicIncRef(&pipeline->refcount);
}

static void WEBGPU_ReleaseGraphicsPipelineReference(WebGPUGraphicsPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    SDL_assert(SDL_GetAtomicInt(&pipeline->refcount) > 0);
    if (SDL_AtomicDecRef(&pipeline->refcount)) {
        WEBGPU_DestroyGraphicsPipeline(pipeline);
    }
}

static void WEBGPU_RetainComputePipelineReference(WebGPUComputePipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    SDL_assert(SDL_GetAtomicInt(&pipeline->refcount) > 0);
    (void)SDL_AtomicIncRef(&pipeline->refcount);
}

static void WEBGPU_ReleaseComputePipelineReference(WebGPUComputePipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    SDL_assert(SDL_GetAtomicInt(&pipeline->refcount) > 0);
    if (SDL_AtomicDecRef(&pipeline->refcount)) {
        WEBGPU_DestroyComputePipeline(pipeline);
    }
}

static void WEBGPU_DrainRetiredSubmissionsIfAllowed(WebGPURenderer *renderer)
{
    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return;
    }
    WEBGPU_DrainRetiredSubmissions(renderer);
}

static void WEBGPU_RetainSwapchainSubmission(WebGPURenderer *renderer, WebGPUSubmission *submission)
{
    if (!submission) {
        return;
    }

    SDL_LockMutex(renderer->fence_lock);
    submission->presentation_refcount += 1;
    WEBGPU_RetainSubmissionReferenceLocked(submission);
    SDL_UnlockMutex(renderer->fence_lock);
}

static void WEBGPU_ReleaseSwapchainSubmissionReference(WebGPURenderer *renderer, WebGPUSubmission **submission_slot)
{
    WebGPUSubmission *submission;
    bool destroy = false;

    if (!submission_slot || !*submission_slot) {
        return;
    }

    submission = *submission_slot;
    SDL_LockMutex(renderer->fence_lock);
    SDL_assert(submission->presentation_refcount > 0);
    if (submission->presentation_refcount > 0) {
        submission->presentation_refcount -= 1;
    }
    *submission_slot = NULL;
    destroy = WEBGPU_ReleaseSubmissionReferenceLocked(submission);
    SDL_UnlockMutex(renderer->fence_lock);

    if (destroy) {
        WEBGPU_DestroySubmission(submission, false);
    }
    WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
}

static void WEBGPU_ClearWindowSwapchainSubmissions(WebGPURenderer *renderer, WebGPUWindowData *window_data)
{
    for (Uint32 i = 0; i < MAX_FRAMES_IN_FLIGHT; i += 1) {
        WEBGPU_ReleaseSwapchainSubmissionReference(renderer, &window_data->in_flight_submissions[i]);
    }
    window_data->frame_counter = 0;
}

static bool WEBGPU_WaitForSwapchainFrameSlot(
    WebGPURenderer *renderer,
    WebGPUWindowData *window_data,
    bool block,
    bool release_completed,
    bool *available,
    const char *context)
{
    WebGPUSubmission *submission = window_data->in_flight_submissions[window_data->frame_counter];
    bool completed = false;

    *available = true;
    if (!submission) {
        return true;
    }

    if (!WEBGPU_WaitForSubmission(renderer, submission, block ? SDL_MAX_UINT64 : 0, &completed)) {
        (void)WEBGPU_DrainDeviceLoss(renderer, context);
        return false;
    }
    if (!completed) {
        if (block) {
            return SDL_SetError("WebGPU swapchain submission wait did not complete");
        }
        *available = false;
        return true;
    }

    if (release_completed) {
        WEBGPU_ReleaseSwapchainSubmissionReference(renderer, &window_data->in_flight_submissions[window_data->frame_counter]);
    }
    return WEBGPU_DrainRuntimeErrors(renderer, context);
}

static void WEBGPU_RecordSubmittedSwapchainSubmission(WebGPUCommandBuffer *command_buffer, WebGPUSubmission *submission)
{
    WebGPURenderer *renderer = command_buffer->renderer;
    WebGPUWindowData *window_data;
    Uint32 frame_slot;

    if (!submission->swapchain_texture || !submission->swapchain_texture->texture) {
        return;
    }

    window_data = submission->swapchain_texture->window_data;
    if (!WEBGPU_WindowDataIsActive(window_data) ||
        window_data->renderer != renderer) {
        return;
    }

    frame_slot = window_data->frame_counter;
    SDL_assert(window_data->in_flight_submissions[frame_slot] == NULL);
    if (window_data->in_flight_submissions[frame_slot]) {
        WEBGPU_ReleaseSwapchainSubmissionReference(renderer, &window_data->in_flight_submissions[frame_slot]);
    }
    WEBGPU_RetainSwapchainSubmission(renderer, submission);
    window_data->in_flight_submissions[frame_slot] = submission;
    window_data->frame_counter = (window_data->frame_counter + 1) % renderer->allowed_frames_in_flight;
}

static bool WEBGPU_RetainTrackedBufferReference(WebGPURenderer *renderer, WGPUBuffer buffer)
{
    WebGPUBufferReference *references;

    if (!buffer) {
        return true;
    }
    if (!renderer->resource_lock) {
        return SDL_SetError("WebGPU resource tracking is not initialized");
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->buffer_reference_count; i += 1) {
        if (renderer->buffer_references[i].buffer == buffer) {
            renderer->buffer_references[i].count += 1;
            SDL_UnlockMutex(renderer->resource_lock);
            return true;
        }
    }

    if (renderer->buffer_reference_count >= renderer->buffer_reference_capacity) {
        Uint32 new_capacity = renderer->buffer_reference_capacity ? renderer->buffer_reference_capacity * 2 : 16;
        references = (WebGPUBufferReference *)SDL_realloc(renderer->buffer_references, new_capacity * sizeof(*references));
        if (!references) {
            SDL_UnlockMutex(renderer->resource_lock);
            return false;
        }
        renderer->buffer_references = references;
        renderer->buffer_reference_capacity = new_capacity;
    }

    renderer->buffer_references[renderer->buffer_reference_count].buffer = buffer;
    renderer->buffer_references[renderer->buffer_reference_count].count = 1;
    renderer->buffer_reference_count += 1;
    SDL_UnlockMutex(renderer->resource_lock);
    return true;
}

static void WEBGPU_ReleaseTrackedBufferReference(WebGPURenderer *renderer, WGPUBuffer buffer)
{
    if (!renderer || !buffer || !renderer->resource_lock) {
        return;
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->buffer_reference_count; i += 1) {
        if (renderer->buffer_references[i].buffer == buffer) {
            SDL_assert(renderer->buffer_references[i].count > 0);
            renderer->buffer_references[i].count -= 1;
            if (renderer->buffer_references[i].count == 0) {
                renderer->buffer_references[i] = renderer->buffer_references[renderer->buffer_reference_count - 1];
                renderer->buffer_reference_count -= 1;
            }
            break;
        }
    }
    SDL_UnlockMutex(renderer->resource_lock);
}

static Uint32 WEBGPU_GetTrackedBufferReferenceCount(WebGPURenderer *renderer, WGPUBuffer buffer)
{
    Uint32 count = 0;

    if (!renderer || !buffer || !renderer->resource_lock) {
        return 0;
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->buffer_reference_count; i += 1) {
        if (renderer->buffer_references[i].buffer == buffer) {
            count = renderer->buffer_references[i].count;
            break;
        }
    }
    SDL_UnlockMutex(renderer->resource_lock);
    return count;
}

static bool WEBGPU_RetainTrackedTextureReference(WebGPURenderer *renderer, WGPUTexture texture)
{
    WebGPUTextureReference *references;

    if (!texture) {
        return true;
    }
    if (!renderer->resource_lock) {
        return SDL_SetError("WebGPU resource tracking is not initialized");
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->texture_reference_count; i += 1) {
        if (renderer->texture_references[i].texture == texture) {
            renderer->texture_references[i].count += 1;
            SDL_UnlockMutex(renderer->resource_lock);
            return true;
        }
    }

    if (renderer->texture_reference_count >= renderer->texture_reference_capacity) {
        Uint32 new_capacity = renderer->texture_reference_capacity ? renderer->texture_reference_capacity * 2 : 16;
        references = (WebGPUTextureReference *)SDL_realloc(renderer->texture_references, new_capacity * sizeof(*references));
        if (!references) {
            SDL_UnlockMutex(renderer->resource_lock);
            return false;
        }
        renderer->texture_references = references;
        renderer->texture_reference_capacity = new_capacity;
    }

    renderer->texture_references[renderer->texture_reference_count].texture = texture;
    renderer->texture_references[renderer->texture_reference_count].count = 1;
    renderer->texture_reference_count += 1;
    SDL_UnlockMutex(renderer->resource_lock);
    return true;
}

static void WEBGPU_ReleaseTrackedTextureReference(WebGPURenderer *renderer, WGPUTexture texture)
{
    if (!renderer || !texture || !renderer->resource_lock) {
        return;
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->texture_reference_count; i += 1) {
        if (renderer->texture_references[i].texture == texture) {
            SDL_assert(renderer->texture_references[i].count > 0);
            renderer->texture_references[i].count -= 1;
            if (renderer->texture_references[i].count == 0) {
                renderer->texture_references[i] = renderer->texture_references[renderer->texture_reference_count - 1];
                renderer->texture_reference_count -= 1;
            }
            break;
        }
    }
    SDL_UnlockMutex(renderer->resource_lock);
}

static Uint32 WEBGPU_GetTrackedTextureReferenceCount(WebGPURenderer *renderer, WGPUTexture texture)
{
    Uint32 count = 0;

    if (!renderer || !texture || !renderer->resource_lock) {
        return 0;
    }

    SDL_LockMutex(renderer->resource_lock);
    for (Uint32 i = 0; i < renderer->texture_reference_count; i += 1) {
        if (renderer->texture_references[i].texture == texture) {
            count = renderer->texture_references[i].count;
            break;
        }
    }
    SDL_UnlockMutex(renderer->resource_lock);
    return count;
}

static void WEBGPU_ReleaseCommandBufferPipelineWrappers(WebGPUCommandBuffer *command_buffer)
{
    for (Uint32 i = 0; i < command_buffer->used_graphics_pipeline_wrapper_count; i += 1) {
        WEBGPU_ReleaseGraphicsPipelineReference(command_buffer->used_graphics_pipeline_wrappers[i]);
    }
    SDL_free(command_buffer->used_graphics_pipeline_wrappers);
    command_buffer->used_graphics_pipeline_wrappers = NULL;
    command_buffer->used_graphics_pipeline_wrapper_count = 0;
    command_buffer->used_graphics_pipeline_wrapper_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_compute_pipeline_wrapper_count; i += 1) {
        WEBGPU_ReleaseComputePipelineReference(command_buffer->used_compute_pipeline_wrappers[i]);
    }
    SDL_free(command_buffer->used_compute_pipeline_wrappers);
    command_buffer->used_compute_pipeline_wrappers = NULL;
    command_buffer->used_compute_pipeline_wrapper_count = 0;
    command_buffer->used_compute_pipeline_wrapper_capacity = 0;

    command_buffer->current_graphics_pipeline = NULL;
    command_buffer->current_compute_pipeline = NULL;
}

static void WEBGPU_ReleaseCommandBufferResources(WebGPUCommandBuffer *command_buffer)
{
    WEBGPU_ClearResourceBindGroupCache(command_buffer);
    WEBGPU_ReleaseCommandBufferPipelineWrappers(command_buffer);

    for (Uint32 i = 0; i < command_buffer->used_buffer_count; i += 1) {
        WEBGPU_ReleaseTrackedBufferReference(command_buffer->renderer, command_buffer->used_buffers[i]);
        wgpuBufferRelease(command_buffer->used_buffers[i]);
    }
    SDL_free(command_buffer->used_buffers);
    command_buffer->used_buffers = NULL;
    command_buffer->used_buffer_count = 0;
    command_buffer->used_buffer_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_graphics_pipeline_count; i += 1) {
        wgpuRenderPipelineRelease(command_buffer->used_graphics_pipelines[i]);
    }
    SDL_free(command_buffer->used_graphics_pipelines);
    command_buffer->used_graphics_pipelines = NULL;
    command_buffer->used_graphics_pipeline_count = 0;
    command_buffer->used_graphics_pipeline_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_compute_pipeline_count; i += 1) {
        wgpuComputePipelineRelease(command_buffer->used_compute_pipelines[i]);
    }
    SDL_free(command_buffer->used_compute_pipelines);
    command_buffer->used_compute_pipelines = NULL;
    command_buffer->used_compute_pipeline_count = 0;
    command_buffer->used_compute_pipeline_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_bind_group_count; i += 1) {
        wgpuBindGroupRelease(command_buffer->used_bind_groups[i]);
    }
    SDL_free(command_buffer->used_bind_groups);
    command_buffer->used_bind_groups = NULL;
    command_buffer->used_bind_group_count = 0;
    command_buffer->used_bind_group_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->texture_download_count; i += 1) {
        WEBGPU_DestroyTextureDownload(command_buffer->texture_downloads[i]);
    }
    SDL_free(command_buffer->texture_downloads);
    command_buffer->texture_downloads = NULL;
    command_buffer->texture_download_count = 0;
    command_buffer->texture_download_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->buffer_download_count; i += 1) {
        WEBGPU_DestroyBufferDownload(command_buffer->buffer_downloads[i]);
    }
    SDL_free(command_buffer->buffer_downloads);
    command_buffer->buffer_downloads = NULL;
    command_buffer->buffer_download_count = 0;
    command_buffer->buffer_download_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_sampler_count; i += 1) {
        wgpuSamplerRelease(command_buffer->used_samplers[i]);
    }
    SDL_free(command_buffer->used_samplers);
    command_buffer->used_samplers = NULL;
    command_buffer->used_sampler_count = 0;
    command_buffer->used_sampler_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_texture_view_count; i += 1) {
        wgpuTextureViewRelease(command_buffer->used_texture_views[i]);
    }
    SDL_free(command_buffer->used_texture_views);
    command_buffer->used_texture_views = NULL;
    command_buffer->used_texture_view_count = 0;
    command_buffer->used_texture_view_capacity = 0;

    for (Uint32 i = 0; i < command_buffer->used_texture_count; i += 1) {
        WEBGPU_ReleaseTrackedTextureReference(command_buffer->renderer, command_buffer->used_textures[i]);
        wgpuTextureRelease(command_buffer->used_textures[i]);
    }
    SDL_free(command_buffer->used_textures);
    command_buffer->used_textures = NULL;
    command_buffer->used_texture_count = 0;
    command_buffer->used_texture_capacity = 0;
}

static bool WEBGPU_TrackCommandBufferBuffer(WebGPUCommandBuffer *command_buffer, WGPUBuffer buffer)
{
    WGPUBuffer *buffers;

    if (!buffer) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_buffer_count; i += 1) {
        if (command_buffer->used_buffers[i] == buffer) {
            return true;
        }
    }

    if (command_buffer->used_buffer_count >= command_buffer->used_buffer_capacity) {
        Uint32 new_capacity = command_buffer->used_buffer_capacity ? command_buffer->used_buffer_capacity * 2 : 4;
        buffers = (WGPUBuffer *)SDL_realloc(command_buffer->used_buffers, new_capacity * sizeof(*buffers));
        if (!buffers) {
            return false;
        }
        command_buffer->used_buffers = buffers;
        command_buffer->used_buffer_capacity = new_capacity;
    }

    if (!WEBGPU_RetainTrackedBufferReference(command_buffer->renderer, buffer)) {
        return false;
    }
    wgpuBufferAddRef(buffer);
    command_buffer->used_buffers[command_buffer->used_buffer_count] = buffer;
    command_buffer->used_buffer_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferGraphicsPipeline(WebGPUCommandBuffer *command_buffer, WebGPUGraphicsPipeline *pipeline)
{
    WebGPUGraphicsPipeline **pipeline_wrappers;
    WGPURenderPipeline *pipelines;

    if (!pipeline || !pipeline->pipeline) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_graphics_pipeline_wrapper_count; i += 1) {
        if (command_buffer->used_graphics_pipeline_wrappers[i] == pipeline) {
            goto track_raw_handle;
        }
    }

    if (command_buffer->used_graphics_pipeline_wrapper_count >= command_buffer->used_graphics_pipeline_wrapper_capacity) {
        Uint32 new_capacity = command_buffer->used_graphics_pipeline_wrapper_capacity ? command_buffer->used_graphics_pipeline_wrapper_capacity * 2 : 2;
        pipeline_wrappers = (WebGPUGraphicsPipeline **)SDL_realloc(command_buffer->used_graphics_pipeline_wrappers, new_capacity * sizeof(*pipeline_wrappers));
        if (!pipeline_wrappers) {
            return false;
        }
        command_buffer->used_graphics_pipeline_wrappers = pipeline_wrappers;
        command_buffer->used_graphics_pipeline_wrapper_capacity = new_capacity;
    }

    WEBGPU_RetainGraphicsPipelineReference(pipeline);
    command_buffer->used_graphics_pipeline_wrappers[command_buffer->used_graphics_pipeline_wrapper_count] = pipeline;
    command_buffer->used_graphics_pipeline_wrapper_count += 1;

track_raw_handle:
    for (Uint32 i = 0; i < command_buffer->used_graphics_pipeline_count; i += 1) {
        if (command_buffer->used_graphics_pipelines[i] == pipeline->pipeline) {
            return true;
        }
    }

    if (command_buffer->used_graphics_pipeline_count >= command_buffer->used_graphics_pipeline_capacity) {
        Uint32 new_capacity = command_buffer->used_graphics_pipeline_capacity ? command_buffer->used_graphics_pipeline_capacity * 2 : 4;
        pipelines = (WGPURenderPipeline *)SDL_realloc(command_buffer->used_graphics_pipelines, new_capacity * sizeof(*pipelines));
        if (!pipelines) {
            return false;
        }
        command_buffer->used_graphics_pipelines = pipelines;
        command_buffer->used_graphics_pipeline_capacity = new_capacity;
    }

    wgpuRenderPipelineAddRef(pipeline->pipeline);
    command_buffer->used_graphics_pipelines[command_buffer->used_graphics_pipeline_count] = pipeline->pipeline;
    command_buffer->used_graphics_pipeline_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferComputePipeline(WebGPUCommandBuffer *command_buffer, WebGPUComputePipeline *pipeline)
{
    WebGPUComputePipeline **pipeline_wrappers;
    WGPUComputePipeline *pipelines;

    if (!pipeline || !pipeline->pipeline) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_compute_pipeline_wrapper_count; i += 1) {
        if (command_buffer->used_compute_pipeline_wrappers[i] == pipeline) {
            goto track_raw_handle;
        }
    }

    if (command_buffer->used_compute_pipeline_wrapper_count >= command_buffer->used_compute_pipeline_wrapper_capacity) {
        Uint32 new_capacity = command_buffer->used_compute_pipeline_wrapper_capacity ? command_buffer->used_compute_pipeline_wrapper_capacity * 2 : 2;
        pipeline_wrappers = (WebGPUComputePipeline **)SDL_realloc(command_buffer->used_compute_pipeline_wrappers, new_capacity * sizeof(*pipeline_wrappers));
        if (!pipeline_wrappers) {
            return false;
        }
        command_buffer->used_compute_pipeline_wrappers = pipeline_wrappers;
        command_buffer->used_compute_pipeline_wrapper_capacity = new_capacity;
    }

    WEBGPU_RetainComputePipelineReference(pipeline);
    command_buffer->used_compute_pipeline_wrappers[command_buffer->used_compute_pipeline_wrapper_count] = pipeline;
    command_buffer->used_compute_pipeline_wrapper_count += 1;

track_raw_handle:
    for (Uint32 i = 0; i < command_buffer->used_compute_pipeline_count; i += 1) {
        if (command_buffer->used_compute_pipelines[i] == pipeline->pipeline) {
            return true;
        }
    }

    if (command_buffer->used_compute_pipeline_count >= command_buffer->used_compute_pipeline_capacity) {
        Uint32 new_capacity = command_buffer->used_compute_pipeline_capacity ? command_buffer->used_compute_pipeline_capacity * 2 : 4;
        pipelines = (WGPUComputePipeline *)SDL_realloc(command_buffer->used_compute_pipelines, new_capacity * sizeof(*pipelines));
        if (!pipelines) {
            return false;
        }
        command_buffer->used_compute_pipelines = pipelines;
        command_buffer->used_compute_pipeline_capacity = new_capacity;
    }

    wgpuComputePipelineAddRef(pipeline->pipeline);
    command_buffer->used_compute_pipelines[command_buffer->used_compute_pipeline_count] = pipeline->pipeline;
    command_buffer->used_compute_pipeline_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferTexture(WebGPUCommandBuffer *command_buffer, WGPUTexture texture)
{
    WGPUTexture *textures;

    if (!texture) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_texture_count; i += 1) {
        if (command_buffer->used_textures[i] == texture) {
            return true;
        }
    }

    if (command_buffer->used_texture_count >= command_buffer->used_texture_capacity) {
        Uint32 new_capacity = command_buffer->used_texture_capacity ? command_buffer->used_texture_capacity * 2 : 4;
        textures = (WGPUTexture *)SDL_realloc(command_buffer->used_textures, new_capacity * sizeof(*textures));
        if (!textures) {
            return false;
        }
        command_buffer->used_textures = textures;
        command_buffer->used_texture_capacity = new_capacity;
    }

    if (!WEBGPU_RetainTrackedTextureReference(command_buffer->renderer, texture)) {
        return false;
    }
    wgpuTextureAddRef(texture);
    command_buffer->used_textures[command_buffer->used_texture_count] = texture;
    command_buffer->used_texture_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferTextureView(WebGPUCommandBuffer *command_buffer, WGPUTextureView texture_view)
{
    WGPUTextureView *texture_views;

    if (!texture_view) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_texture_view_count; i += 1) {
        if (command_buffer->used_texture_views[i] == texture_view) {
            return true;
        }
    }

    if (command_buffer->used_texture_view_count >= command_buffer->used_texture_view_capacity) {
        Uint32 new_capacity = command_buffer->used_texture_view_capacity ? command_buffer->used_texture_view_capacity * 2 : 4;
        texture_views = (WGPUTextureView *)SDL_realloc(command_buffer->used_texture_views, new_capacity * sizeof(*texture_views));
        if (!texture_views) {
            return false;
        }
        command_buffer->used_texture_views = texture_views;
        command_buffer->used_texture_view_capacity = new_capacity;
    }

    wgpuTextureViewAddRef(texture_view);
    command_buffer->used_texture_views[command_buffer->used_texture_view_count] = texture_view;
    command_buffer->used_texture_view_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferSampler(WebGPUCommandBuffer *command_buffer, WGPUSampler sampler)
{
    WGPUSampler *samplers;

    if (!sampler) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_sampler_count; i += 1) {
        if (command_buffer->used_samplers[i] == sampler) {
            return true;
        }
    }

    if (command_buffer->used_sampler_count >= command_buffer->used_sampler_capacity) {
        Uint32 new_capacity = command_buffer->used_sampler_capacity ? command_buffer->used_sampler_capacity * 2 : 4;
        samplers = (WGPUSampler *)SDL_realloc(command_buffer->used_samplers, new_capacity * sizeof(*samplers));
        if (!samplers) {
            return false;
        }
        command_buffer->used_samplers = samplers;
        command_buffer->used_sampler_capacity = new_capacity;
    }

    wgpuSamplerAddRef(sampler);
    command_buffer->used_samplers[command_buffer->used_sampler_count] = sampler;
    command_buffer->used_sampler_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferBindGroup(WebGPUCommandBuffer *command_buffer, WGPUBindGroup bind_group)
{
    WGPUBindGroup *bind_groups;

    if (!bind_group) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->used_bind_group_count; i += 1) {
        if (command_buffer->used_bind_groups[i] == bind_group) {
            return true;
        }
    }

    if (command_buffer->used_bind_group_count >= command_buffer->used_bind_group_capacity) {
        Uint32 new_capacity = command_buffer->used_bind_group_capacity ? command_buffer->used_bind_group_capacity * 2 : 4;
        bind_groups = (WGPUBindGroup *)SDL_realloc(command_buffer->used_bind_groups, new_capacity * sizeof(*bind_groups));
        if (!bind_groups) {
            return false;
        }
        command_buffer->used_bind_groups = bind_groups;
        command_buffer->used_bind_group_capacity = new_capacity;
    }

    wgpuBindGroupAddRef(bind_group);
    command_buffer->used_bind_groups[command_buffer->used_bind_group_count] = bind_group;
    command_buffer->used_bind_group_count += 1;
    WEBGPU_RecordCommandBufferBindGroupPeak(command_buffer);
    return true;
}

static bool WEBGPU_AcquireResourceBindGroup(
    WebGPUCommandBuffer *command_buffer,
    const WGPUBindGroupDescriptor *bind_group_desc,
    const WebGPUResourceBindGroupCacheKey *cache_key,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    const char *scoped_create_context,
    const char *unscoped_create_error,
    const char *track_error,
    WGPUBindGroup *out_bind_group)
{
    WGPUBindGroup bind_group;
    bool scoped_error;

    *out_bind_group = NULL;

    bind_group = WEBGPU_FindResourceBindGroupCacheEntry(command_buffer, cache_key);
    if (bind_group) {
        *out_bind_group = bind_group;
        return true;
    }

    scoped_error = WEBGPU_CommandBufferCanWaitForErrorScope(command_buffer);
    if (scoped_error) {
        bind_group = WEBGPU_CreateBindGroup(command_buffer->renderer, bind_group_desc, scoped_create_context);
    } else {
        bind_group = wgpuDeviceCreateBindGroup(command_buffer->renderer->device, bind_group_desc);
    }
    if (!bind_group) {
        if (!scoped_error) {
            WEBGPU_SetStringError(unscoped_create_error);
        }
        return false;
    }

    WEBGPU_RecordBindGroupCreate(command_buffer->renderer, instrumentation_path);
    if (!WEBGPU_TrackCommandBufferBindGroup(command_buffer, bind_group)) {
        wgpuBindGroupRelease(bind_group);
        WEBGPU_SetStringError(track_error);
        return false;
    }
    WEBGPU_InsertResourceBindGroupCacheEntry(command_buffer, cache_key, bind_group);

    /* Return the command-buffer-tracked reference; resource cache entries are non-owning. */
    *out_bind_group = bind_group;
    wgpuBindGroupRelease(bind_group);
    return true;
}

static bool WEBGPU_TrackCommandBufferTextureDownload(WebGPUCommandBuffer *command_buffer, WebGPUTextureDownload *download)
{
    WebGPUTextureDownload **downloads;

    if (command_buffer->texture_download_count >= command_buffer->texture_download_capacity) {
        Uint32 new_capacity = command_buffer->texture_download_capacity ? command_buffer->texture_download_capacity * 2 : 2;
        downloads = (WebGPUTextureDownload **)SDL_realloc(command_buffer->texture_downloads, new_capacity * sizeof(*downloads));
        if (!downloads) {
            return false;
        }
        command_buffer->texture_downloads = downloads;
        command_buffer->texture_download_capacity = new_capacity;
    }

    command_buffer->texture_downloads[command_buffer->texture_download_count] = download;
    command_buffer->texture_download_count += 1;
    return true;
}

static bool WEBGPU_TrackCommandBufferBufferDownload(WebGPUCommandBuffer *command_buffer, WebGPUBufferDownload *download)
{
    WebGPUBufferDownload **downloads;

    if (command_buffer->buffer_download_count >= command_buffer->buffer_download_capacity) {
        Uint32 new_capacity = command_buffer->buffer_download_capacity ? command_buffer->buffer_download_capacity * 2 : 2;
        downloads = (WebGPUBufferDownload **)SDL_realloc(command_buffer->buffer_downloads, new_capacity * sizeof(*downloads));
        if (!downloads) {
            return false;
        }
        command_buffer->buffer_downloads = downloads;
        command_buffer->buffer_download_capacity = new_capacity;
    }

    command_buffer->buffer_downloads[command_buffer->buffer_download_count] = download;
    command_buffer->buffer_download_count += 1;
    return true;
}

static void WEBGPU_TransferCommandBufferResourcesToSubmission(WebGPUCommandBuffer *command_buffer, WebGPUSubmission *submission)
{
    WEBGPU_ClearResourceBindGroupCache(command_buffer);
    WEBGPU_ReleaseCommandBufferPipelineWrappers(command_buffer);

    submission->buffers = command_buffer->used_buffers;
    submission->buffer_count = command_buffer->used_buffer_count;
    command_buffer->used_buffers = NULL;
    command_buffer->used_buffer_count = 0;
    command_buffer->used_buffer_capacity = 0;

    submission->graphics_pipelines = command_buffer->used_graphics_pipelines;
    submission->graphics_pipeline_count = command_buffer->used_graphics_pipeline_count;
    command_buffer->used_graphics_pipelines = NULL;
    command_buffer->used_graphics_pipeline_count = 0;
    command_buffer->used_graphics_pipeline_capacity = 0;

    submission->compute_pipelines = command_buffer->used_compute_pipelines;
    submission->compute_pipeline_count = command_buffer->used_compute_pipeline_count;
    command_buffer->used_compute_pipelines = NULL;
    command_buffer->used_compute_pipeline_count = 0;
    command_buffer->used_compute_pipeline_capacity = 0;

    submission->textures = command_buffer->used_textures;
    submission->texture_count = command_buffer->used_texture_count;
    command_buffer->used_textures = NULL;
    command_buffer->used_texture_count = 0;
    command_buffer->used_texture_capacity = 0;

    submission->texture_views = command_buffer->used_texture_views;
    submission->texture_view_count = command_buffer->used_texture_view_count;
    command_buffer->used_texture_views = NULL;
    command_buffer->used_texture_view_count = 0;
    command_buffer->used_texture_view_capacity = 0;

    if (command_buffer->swapchain_texture) {
        if (command_buffer->swapchain_texture->texture) {
            submission->swapchain_texture = command_buffer->swapchain_texture;
        } else {
            WEBGPU_ReleaseSwapchainTexture(command_buffer->swapchain_texture);
        }
    }
    command_buffer->swapchain_texture = NULL;

    submission->samplers = command_buffer->used_samplers;
    submission->sampler_count = command_buffer->used_sampler_count;
    command_buffer->used_samplers = NULL;
    command_buffer->used_sampler_count = 0;
    command_buffer->used_sampler_capacity = 0;

    submission->bind_groups = command_buffer->used_bind_groups;
    submission->bind_group_count = command_buffer->used_bind_group_count;
    command_buffer->used_bind_groups = NULL;
    command_buffer->used_bind_group_count = 0;
    command_buffer->used_bind_group_capacity = 0;

    submission->texture_downloads = command_buffer->texture_downloads;
    submission->texture_download_count = command_buffer->texture_download_count;
    command_buffer->texture_downloads = NULL;
    command_buffer->texture_download_count = 0;
    command_buffer->texture_download_capacity = 0;

    submission->buffer_downloads = command_buffer->buffer_downloads;
    submission->buffer_download_count = command_buffer->buffer_download_count;
    command_buffer->buffer_downloads = NULL;
    command_buffer->buffer_download_count = 0;
    command_buffer->buffer_download_capacity = 0;
}

static Uint32 WEBGPU_StorageViewArrayLayerCount(const SDL_GPUTextureCreateInfo *createinfo)
{
    if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY &&
        (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                              SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ))) {
        return createinfo->layer_count_or_depth;
    }
    return 1;
}

static bool WEBGPU_CreateTextureHandles(
    WebGPURenderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo,
    Uint64 generation,
    WGPUTexture *out_texture,
    WGPUTextureView *out_view,
    WGPUTextureView *out_storage_view)
{
    WGPUTextureDescriptor texture_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    WGPUTextureFormat format;
    WebGPUTextureViewUsage view_usage;
    WGPUTexture texture;
    WGPUTextureView view = NULL;
    WGPUTextureView storage_view = NULL;
    WebGPUTextureViewDescription view_description;
    WebGPUTextureViewDescription storage_view_description;
    const char *debug_name;
    Uint32 sample_count;
    Uint32 view_mip_level_count;
    Uint32 view_array_layer_count;

    *out_texture = NULL;
    *out_view = NULL;
    *out_storage_view = NULL;

    format = WEBGPU_ToWGPUTextureFormat(createinfo->format);
    sample_count = WEBGPU_ToSampleCount(createinfo->sample_count);
    debug_name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_TEXTURE_CREATE_NAME_STRING, NULL);

    texture_desc.label = WEBGPU_StringView(debug_name);
    texture_desc.usage = WEBGPU_ToTextureUsage(createinfo->format, createinfo->usage);
    if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
        texture_desc.usage = WGPUTextureUsage_RenderAttachment;
        if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
            texture_desc.usage |= WGPUTextureUsage_TextureBinding;
        }
    }
    texture_desc.dimension = WEBGPU_ToTextureDimension(createinfo->type);
    texture_desc.size.width = createinfo->width;
    texture_desc.size.height = createinfo->height;
    texture_desc.size.depthOrArrayLayers = createinfo->layer_count_or_depth;
    texture_desc.format = format;
    texture_desc.mipLevelCount = createinfo->num_levels;
    texture_desc.sampleCount = sample_count;
    texture = wgpuDeviceCreateTexture(renderer->device, &texture_desc);
    if (!texture) {
        WEBGPU_SetStringError("CreateTexture failed");
        return false;
    }

    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) {
            view_usage = WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED;
        } else if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
            view_usage = WEBGPU_TEXTURE_VIEW_USAGE_COLOR_ATTACHMENT;
        } else {
            view_usage = WEBGPU_TEXTURE_VIEW_USAGE_DEPTH_STENCIL_ATTACHMENT;
        }
        view_mip_level_count = (view_usage == WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED) ? createinfo->num_levels : 1;
        view_array_layer_count = (view_usage == WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED &&
                                  (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
                                   createinfo->type == SDL_GPU_TEXTURETYPE_CUBE ||
                                   createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY))
                                     ? createinfo->layer_count_or_depth
                                     : 1;
        view_description = WEBGPU_TextureViewDescription(
            createinfo->format,
            createinfo->type,
            view_usage,
            generation,
            0,
            view_mip_level_count,
            0,
            view_array_layer_count);
        WEBGPU_UseDepthOnlySampledTextureViewDescription(&view_description);
        view = WEBGPU_CreateTextureViewFromDescription(texture, &view_description);
        if (!view) {
            wgpuTextureRelease(texture);
            WEBGPU_SetStringError("CreateTextureView failed");
            return false;
        }
    }

    if (createinfo->usage & (SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ |
                             SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ)) {
        storage_view_description = WEBGPU_TextureViewDescription(
            createinfo->format,
            createinfo->type,
            WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ,
            generation,
            0,
            1,
            0,
            WEBGPU_StorageViewArrayLayerCount(createinfo));
        storage_view = WEBGPU_CreateTextureViewFromDescription(texture, &storage_view_description);
        if (!storage_view) {
            if (view) {
                wgpuTextureViewRelease(view);
            }
            wgpuTextureRelease(texture);
            WEBGPU_SetStringError("CreateTextureView failed for storage texture");
            return false;
        }
    }

    if (debug_name) {
        if (view) {
            wgpuTextureViewSetLabel(view, WEBGPU_StringView(debug_name));
        }
        if (storage_view) {
            wgpuTextureViewSetLabel(storage_view, WEBGPU_StringView(debug_name));
        }
    }

    *out_texture = texture;
    *out_view = view;
    *out_storage_view = storage_view;
    return true;
}

static bool WEBGPU_CreateBufferHandle(
    WebGPURenderer *renderer,
    SDL_GPUBufferUsageFlags usage,
    Uint32 size,
    const char *debug_name,
    WGPUBuffer *out_buffer,
    Uint64 *out_allocation_size)
{
    WGPUBufferDescriptor buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    Uint64 allocation_size = WEBGPU_BufferAllocationSize(size);

    *out_buffer = NULL;
    buffer_desc.label = WEBGPU_StringView(debug_name);
    buffer_desc.usage = WEBGPU_ToBufferUsage(usage);
    buffer_desc.size = allocation_size;

    *out_buffer = wgpuDeviceCreateBuffer(renderer->device, &buffer_desc);
    if (!*out_buffer) {
        WEBGPU_SetStringError("CreateBuffer failed");
        return false;
    }
    if (out_allocation_size) {
        *out_allocation_size = allocation_size;
    }
    return true;
}

static bool WEBGPU_BufferIsBound(WebGPUCommandBuffer *command_buffer, WGPUBuffer buffer)
{
    if (!buffer) {
        return false;
    }
    return WEBGPU_GetTrackedBufferReferenceCount(command_buffer->renderer, buffer) > 0;
}

static bool WEBGPU_TextureIsBound(WebGPUCommandBuffer *command_buffer, WGPUTexture texture)
{
    if (!texture) {
        return false;
    }
    return WEBGPU_GetTrackedTextureReferenceCount(command_buffer->renderer, texture) > 0;
}

static bool WEBGPU_CycleBufferIfBound(WebGPUCommandBuffer *command_buffer, WebGPUBuffer *buffer, bool cycle, const char *operation)
{
    WGPUBuffer replacement;
    Uint64 replacement_generation;

    if (buffer && buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot cycle a released buffer");
        return false;
    }
    if (!cycle || !buffer || !WEBGPU_BufferIsBound(command_buffer, buffer->buffer)) {
        return true;
    }

    replacement_generation = WEBGPU_NextResourceGeneration(buffer->generation);
    if (!WEBGPU_CreateBufferHandle(command_buffer->renderer, buffer->usage, buffer->size, NULL, &replacement, &buffer->allocation_size)) {
        WEBGPU_FailCommandBuffer(command_buffer, operation);
        return false;
    }

    wgpuBufferRelease(buffer->buffer);
    buffer->buffer = replacement;
    buffer->generation = replacement_generation;
    WEBGPU_RecordBufferGenerationCycle(command_buffer->renderer);
    return true;
}

static bool WEBGPU_CycleTextureIfBound(WebGPUCommandBuffer *command_buffer, WebGPUTexture *texture, bool cycle, const char *operation)
{
    WGPUTexture replacement_texture;
    WGPUTextureView replacement_view;
    WGPUTextureView replacement_storage_view;
    Uint64 replacement_generation;

    if (texture && texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot cycle a released texture");
        return false;
    }
    if (!cycle || !texture || texture->from_surface || !WEBGPU_TextureIsBound(command_buffer, texture->texture)) {
        return true;
    }

    replacement_generation = WEBGPU_NextResourceGeneration(texture->generation);
    if (!WEBGPU_CreateTextureHandles(
            command_buffer->renderer,
            &texture->header.info,
            replacement_generation,
            &replacement_texture,
            &replacement_view,
            &replacement_storage_view)) {
        WEBGPU_FailCommandBuffer(command_buffer, operation);
        return false;
    }

    WEBGPU_ReleaseTextureHandles(texture);
    texture->texture = replacement_texture;
    texture->view = replacement_view;
    texture->storage_view = replacement_storage_view;
    texture->generation = replacement_generation;
    WEBGPU_RecordTextureGenerationCycle(command_buffer->renderer);
    return true;
}

static bool WEBGPU_ChooseSurfaceFormat(WebGPUWindowData *window_data)
{
    WebGPURenderer *renderer = window_data->renderer;
    WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    WGPUStatus status;

    status = wgpuSurfaceGetCapabilities(window_data->surface, renderer->adapter, &capabilities);
    if (status != WGPUStatus_Success || capabilities.formatCount == 0 || !capabilities.formats) {
        return SDL_SetError("WebGPU surface reports no formats");
    }
    for (size_t i = 0; i < capabilities.formatCount; i += 1) {
        SDL_GPUTextureFormat sdl_format = WEBGPU_ToSwapchainSDLFormat(capabilities.formats[i], window_data->swapchain_composition);
        if (sdl_format != SDL_GPU_TEXTUREFORMAT_INVALID &&
            WEBGPU_ToSwapchainSDLFormat(capabilities.formats[i], SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR) != SDL_GPU_TEXTUREFORMAT_INVALID) {
            window_data->surface_format = capabilities.formats[i];
            window_data->sdl_format = sdl_format;
            wgpuSurfaceCapabilitiesFreeMembers(capabilities);
            return true;
        }
    }
    for (size_t i = 0; i < capabilities.formatCount; i += 1) {
        SDL_GPUTextureFormat sdl_format = WEBGPU_ToSwapchainSDLFormat(capabilities.formats[i], window_data->swapchain_composition);
        if (sdl_format != SDL_GPU_TEXTUREFORMAT_INVALID) {
            window_data->surface_format = capabilities.formats[i];
            window_data->sdl_format = sdl_format;
            wgpuSurfaceCapabilitiesFreeMembers(capabilities);
            return true;
        }
    }

    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    return SDL_SetError("WebGPU surface has no SDL-compatible color format");
}

static bool WEBGPU_ConfigureWindow(WebGPUWindowData *window_data, Uint32 width, Uint32 height)
{
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    WGPUCompositeAlphaMode alpha_mode;
    WGPUPresentMode present_mode;
    WGPUTextureFormat view_format;
    SDL_GPUTextureFormat sdl_format;
    WGPUTextureFormat view_formats[1];

    if (width == 0 || height == 0) {
        return SDL_SetError("WebGPU cannot configure a zero-sized window");
    }

    if (window_data->surface_format == WGPUTextureFormat_Undefined) {
        if (!WEBGPU_ChooseSurfaceFormat(window_data)) {
            return false;
        }
    }

    if (!WEBGPU_ToCompositeAlphaMode(window_data->swapchain_composition, &alpha_mode)) {
        return SDL_SetError("Unsupported WebGPU swapchain composition");
    }
    if (!WEBGPU_ToPresentMode(window_data->present_mode, &present_mode)) {
        return SDL_SetError("Unsupported WebGPU present mode");
    }
    view_format = WEBGPU_ToSwapchainViewFormat(window_data->surface_format, window_data->swapchain_composition);
    sdl_format = WEBGPU_ToSDLTextureFormat(view_format);
    if (sdl_format == SDL_GPU_TEXTUREFORMAT_INVALID) {
        return SDL_SetError("Unsupported WebGPU swapchain composition for surface format");
    }

    if (window_data->configured) {
        wgpuSurfaceUnconfigure(window_data->surface);
        window_data->configured = false;
    }

    window_data->width = width;
    window_data->height = height;

    config.device = window_data->renderer->device;
    config.format = window_data->surface_format;
    config.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
    config.width = window_data->width;
    config.height = window_data->height;
    config.presentMode = present_mode;
    config.alphaMode = alpha_mode;
    if (view_format != window_data->surface_format) {
        view_formats[0] = view_format;
        config.viewFormatCount = 1;
        config.viewFormats = view_formats;
    }

    wgpuSurfaceConfigure(window_data->surface, &config);
    window_data->configuration_generation = WEBGPU_NextResourceGeneration(window_data->configuration_generation);
    window_data->sdl_format = sdl_format;
    window_data->configured_usage = config.usage;
    window_data->configured = true;
    window_data->needs_configure = false;
    return true;
}

static bool WEBGPU_PrepareSwapchainBlitPipelines(SDL_GPUDevice *device, WebGPURenderer *renderer, SDL_GPUTextureFormat swapchain_format)
{
    if (!SDL_GPU_FetchBlitPipeline(
            device,
            SDL_GPU_TEXTURETYPE_2D,
            swapchain_format,
            renderer->blit_vertex_shader,
            renderer->blit_from_2d_shader,
            NULL,
            renderer->blit_from_3d_shader,
            NULL,
            NULL,
            &renderer->blit_pipelines,
            &renderer->blit_pipeline_count,
            &renderer->blit_pipeline_capacity)) {
        return false;
    }

    if (!SDL_GPU_FetchBlitPipeline(
            device,
            SDL_GPU_TEXTURETYPE_3D,
            swapchain_format,
            renderer->blit_vertex_shader,
            renderer->blit_from_2d_shader,
            NULL,
            renderer->blit_from_3d_shader,
            NULL,
            NULL,
            &renderer->blit_pipelines,
            &renderer->blit_pipeline_count,
            &renderer->blit_pipeline_capacity)) {
        return false;
    }

    return true;
}

static bool WEBGPU_GetWindowPixelSize(SDL_Window *window, Uint32 *width, Uint32 *height)
{
    int pixel_width = 0;
    int pixel_height = 0;

    if (!SDL_GetWindowSizeInPixels(window, &pixel_width, &pixel_height)) {
        return false;
    }

    *width = pixel_width > 0 ? (Uint32)pixel_width : 0;
    *height = pixel_height > 0 ? (Uint32)pixel_height : 0;
    return true;
}

static bool WEBGPU_EnsureWindowConfigured(WebGPUWindowData *window_data)
{
    Uint32 width = 0;
    Uint32 height = 0;

    if (!WEBGPU_WindowDataIsActive(window_data)) {
        return SDL_SetError("WebGPU window is not active");
    }

    if (!WEBGPU_GetWindowPixelSize(window_data->window, &width, &height)) {
        return false;
    }

    if (width == 0 || height == 0 || (SDL_GetWindowFlags(window_data->window) & SDL_WINDOW_HIDDEN)) {
        if (window_data->configured) {
            wgpuSurfaceUnconfigure(window_data->surface);
            window_data->configured = false;
            window_data->configuration_generation = WEBGPU_NextResourceGeneration(window_data->configuration_generation);
        }
        window_data->width = 0;
        window_data->height = 0;
        window_data->needs_configure = true;
        return true;
    }

    if (!window_data->configured ||
        window_data->needs_configure ||
        width != window_data->width ||
        height != window_data->height) {
        return WEBGPU_ConfigureWindow(window_data, width, height);
    }

    return true;
}

static bool WEBGPU_EnsureSwapchainMaterializationSlot(WebGPUCommandBuffer *command_buffer, WebGPUWindowData *window_data)
{
    bool slot_available = true;

    if (WEBGPU_HasSwapchainSubmitWindow(command_buffer->renderer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot use another swapchain texture while one is awaiting submission");
        return false;
    }
    if (!WEBGPU_WaitForSwapchainFrameSlot(
            command_buffer->renderer,
            window_data,
            false,
            true,
            &slot_available,
            "UseSwapchainTexture")) {
        command_buffer->failed = true;
        return false;
    }
    if (!slot_available) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture frame slot is no longer available; acquire a new swapchain texture");
        return false;
    }
    return true;
}

static bool WEBGPU_MaterializeSwapchainTextureDestination(
    WebGPUCommandBuffer *command_buffer,
    const WebGPUSwapchainTextureDestination *destination,
    const char *operation)
{
    WebGPUTexture *texture;
    WebGPUWindowData *window_data;
    WGPUSurfaceTexture surface_texture = WGPU_SURFACE_TEXTURE_INIT;
    WebGPUTextureViewDescription view_description;
    bool should_retry = false;

    if (!destination || !destination->is_swapchain) {
        return true;
    }

    texture = destination->texture;
    if (!texture || !texture->from_surface || texture->texture) {
        return true;
    }
    if (texture != command_buffer->swapchain_texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture must be acquired by this command buffer");
        return false;
    }

    window_data = destination->window_data;
    if (!WEBGPU_WindowDataIsActive(window_data) ||
        texture->window_data != window_data ||
        window_data->renderer != command_buffer->renderer) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture has no active WebGPU window");
        return false;
    }

    if (window_data->needs_configure ||
        !window_data->configured ||
        window_data->width != texture->header.info.width ||
        window_data->height != texture->header.info.height ||
        window_data->configuration_generation != texture->window_configuration_generation) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture was invalidated before use; acquire a new swapchain texture");
        return false;
    }
    if (!WEBGPU_EnsureSwapchainMaterializationSlot(command_buffer, window_data)) {
        return false;
    }

    wgpuSurfaceGetCurrentTexture(window_data->surface, &surface_texture);
    if (!surface_texture.texture || !WEBGPU_SurfaceStatusOK(surface_texture.status)) {
        should_retry = surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
                       surface_texture.status == WGPUSurfaceGetCurrentTextureStatus_Lost;
        if (surface_texture.texture) {
            wgpuTextureRelease(surface_texture.texture);
            surface_texture.texture = NULL;
        }
        if (should_retry) {
            window_data->needs_configure = true;
            if (!WEBGPU_EnsureWindowConfigured(window_data)) {
                command_buffer->failed = true;
                return false;
            }
            if (!window_data->configured ||
                window_data->width != texture->header.info.width ||
                window_data->height != texture->header.info.height ||
                window_data->configuration_generation != texture->window_configuration_generation) {
                WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture was invalidated before use; acquire a new swapchain texture");
                return false;
            }
            wgpuSurfaceGetCurrentTexture(window_data->surface, &surface_texture);
        }
        if (!surface_texture.texture || !WEBGPU_SurfaceStatusOK(surface_texture.status)) {
            if (surface_texture.texture) {
                wgpuTextureRelease(surface_texture.texture);
            }
            WEBGPU_FailCommandBuffer(command_buffer, operation);
            return false;
        }
    }

    texture->texture = surface_texture.texture;
    view_description = WEBGPU_TextureViewDescription(
        window_data->sdl_format,
        SDL_GPU_TEXTURETYPE_2D,
        WEBGPU_TEXTURE_VIEW_USAGE_SWAPCHAIN_ATTACHMENT,
        texture->generation,
        0,
        1,
        0,
        1);
    texture->view = WEBGPU_CreateTextureViewFromDescription(texture->texture, &view_description);
    if (!texture->view) {
        WEBGPU_ReleaseTextureHandles(texture);
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for swapchain texture");
        return false;
    }

    WEBGPU_BeginSwapchainSubmitWindow(command_buffer);
    return true;
}

static bool WEBGPU_OnWindowSurfaceStateChanged(void *userdata, SDL_Event *event)
{
    SDL_Window *window = (SDL_Window *)userdata;
    WebGPUWindowData *window_data;

    if ((event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED ||
         event->type == SDL_EVENT_WINDOW_HIDDEN ||
         event->type == SDL_EVENT_WINDOW_SHOWN) &&
        event->window.windowID == SDL_GetWindowID(window)) {
        window_data = WEBGPU_FetchWindowData(window);
        if (window_data) {
            window_data->needs_configure = true;
        }
    }

    return true;
}

static void WEBGPU_DestroyCommandBuffer(WebGPUCommandBuffer *command_buffer)
{
    if (!command_buffer) {
        return;
    }
    WEBGPU_EndSwapchainSubmitWindow(command_buffer);
    if (command_buffer->render_pass) {
        wgpuRenderPassEncoderRelease(command_buffer->render_pass);
    }
    if (command_buffer->compute_pass) {
        wgpuComputePassEncoderRelease(command_buffer->compute_pass);
    }
    if (command_buffer->encoder) {
        wgpuCommandEncoderRelease(command_buffer->encoder);
    }
    WEBGPU_ReleaseCommandBufferResources(command_buffer);
    WEBGPU_ReleaseSwapchainTexture(command_buffer->swapchain_texture);
    SDL_free(command_buffer);
}

static bool WEBGPU_PrepareDriver(SDL_VideoDevice *_this, SDL_PropertiesID props)
{
    (void)_this;
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_XR_ENABLE_BOOLEAN, false)) {
        return SDL_SetError("WebGPU backend does not support OpenXR");
    }
    return SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_WGSL_BOOLEAN, false);
}

static SDL_PropertiesID WEBGPU_GetDeviceProperties(SDL_GPUDevice *device)
{
    WebGPURenderer *renderer = (WebGPURenderer *)device->driverData;

    WEBGPU_SyncBindGroupInstrumentationProperties(renderer);
    WEBGPU_SyncResourceGenerationInstrumentationProperties(renderer);
    return renderer->props;
}

static XrResult WEBGPU_DestroyXRSwapchain(
    SDL_GPURenderer *driverData,
    XrSwapchain swapchain,
    SDL_GPUTexture **swapchainImages)
{
    (void)driverData;
    (void)swapchain;
    (void)swapchainImages;
    return WEBGPU_UnsupportedXR("DestroyXRSwapchain");
}

static void WEBGPU_ReleaseBindGroupLayouts(WGPUBindGroupLayout *bind_group_layouts, Uint32 bind_group_layout_count)
{
    for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
        if (bind_group_layouts[i]) {
            wgpuBindGroupLayoutRelease(bind_group_layouts[i]);
            bind_group_layouts[i] = NULL;
        }
    }
}

static bool WEBGPU_TrySampledTextureViewDimension(
    SDL_GPUTextureType texture_type,
    WGPUTextureViewDimension *view_dimension)
{
    switch (texture_type) {
    case SDL_GPU_TEXTURETYPE_2D:
        *view_dimension = WGPUTextureViewDimension_2D;
        return true;
    case SDL_GPU_TEXTURETYPE_2D_ARRAY:
        *view_dimension = WGPUTextureViewDimension_2DArray;
        return true;
    case SDL_GPU_TEXTURETYPE_3D:
        *view_dimension = WGPUTextureViewDimension_3D;
        return true;
    case SDL_GPU_TEXTURETYPE_CUBE:
        *view_dimension = WGPUTextureViewDimension_Cube;
        return true;
    case SDL_GPU_TEXTURETYPE_CUBE_ARRAY:
        *view_dimension = WGPUTextureViewDimension_CubeArray;
        return true;
    default:
        *view_dimension = WGPUTextureViewDimension_Undefined;
        return false;
    }
}

static bool WEBGPU_ToSampledTextureViewDimension(
    SDL_GPUTextureType texture_type,
    WGPUTextureViewDimension *view_dimension)
{
    if (!WEBGPU_TrySampledTextureViewDimension(texture_type, view_dimension)) {
        WEBGPU_SetStringError("unsupported sampled texture type");
        return false;
    }
    return true;
}

static WGPUTextureSampleType WEBGPU_ToWGPUTextureSampleType(WebGPUSampledTextureSampleKind sample_type)
{
    switch (sample_type) {
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT:
        return WGPUTextureSampleType_Float;
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT:
        return WGPUTextureSampleType_UnfilterableFloat;
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH:
        return WGPUTextureSampleType_Depth;
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_SINT:
        return WGPUTextureSampleType_Sint;
    case WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UINT:
        return WGPUTextureSampleType_Uint;
    default:
        return WGPUTextureSampleType_Undefined;
    }
}

static WGPUSamplerBindingType WEBGPU_ToWGPUSamplerBindingType(WebGPUSamplerBindingKind sampler_type)
{
    switch (sampler_type) {
    case WEBGPU_SAMPLER_BINDINGKIND_FILTERING:
        return WGPUSamplerBindingType_Filtering;
    case WEBGPU_SAMPLER_BINDINGKIND_NONFILTERING:
        return WGPUSamplerBindingType_NonFiltering;
    case WEBGPU_SAMPLER_BINDINGKIND_COMPARISON:
        return WGPUSamplerBindingType_Comparison;
    default:
        return WGPUSamplerBindingType_Undefined;
    }
}

static WGPUStorageTextureAccess WEBGPU_ToWGPUStorageTextureAccess(WebGPUStorageTextureAccessKind access_kind)
{
    switch (access_kind) {
    case WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY:
        return WGPUStorageTextureAccess_ReadOnly;
    case WEBGPU_STORAGE_TEXTURE_ACCESSKIND_WRITE_ONLY:
        return WGPUStorageTextureAccess_WriteOnly;
    case WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_WRITE:
        return WGPUStorageTextureAccess_ReadWrite;
    default:
        return WGPUStorageTextureAccess_Undefined;
    }
}

static bool WEBGPU_IsNonFilteringSamplerDescriptor(const WGPUSamplerDescriptor *sampler_desc)
{
    return sampler_desc->magFilter == WGPUFilterMode_Nearest &&
           sampler_desc->minFilter == WGPUFilterMode_Nearest &&
           sampler_desc->mipmapFilter == WGPUMipmapFilterMode_Nearest;
}

static bool WEBGPU_SamplerBindingTypeMatchesLayout(
    WGPUSamplerBindingType layout_type,
    WGPUSamplerBindingType sampler_type)
{
    switch (layout_type) {
    case WGPUSamplerBindingType_Filtering:
        return sampler_type == WGPUSamplerBindingType_Filtering ||
               sampler_type == WGPUSamplerBindingType_NonFiltering;
    case WGPUSamplerBindingType_NonFiltering:
        return sampler_type == WGPUSamplerBindingType_NonFiltering;
    case WGPUSamplerBindingType_Comparison:
        return sampler_type == WGPUSamplerBindingType_Comparison;
    default:
        return false;
    }
}

/* SDL-owned layout facts are normalized below before lowering to WebGPU bind-group layouts. */
static void WEBGPU_InitDefaultSampledTextureLayoutFacts(WebGPUSampledTextureLayoutFacts *backend_facts)
{
    backend_facts->texture_type = SDL_GPU_TEXTURETYPE_2D;
    backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT;
    backend_facts->sampler_type = WEBGPU_SAMPLER_BINDINGKIND_FILTERING;
    backend_facts->multisampled = false;
}

static void WEBGPU_InitDefaultStorageTextureLayoutFacts(
    WebGPUStorageTextureLayoutFacts *backend_facts,
    WebGPUStorageTextureAccessKind access_kind)
{
    backend_facts->access_kind = access_kind;
    backend_facts->format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    backend_facts->texture_type = SDL_GPU_TEXTURETYPE_2D;
}

static bool WEBGPU_SampledTextureLayoutFactsToBindingLayout(
    const WebGPUSampledTextureLayoutFacts *backend_facts,
    WebGPUSampledTextureBindingLayout *layout)
{
    layout->sample_type = WEBGPU_ToWGPUTextureSampleType(backend_facts->sample_type);
    if (layout->sample_type == WGPUTextureSampleType_Undefined) {
        WEBGPU_SetStringError("invalid sampled texture sample type layout");
        return false;
    }
    if (!WEBGPU_ToSampledTextureViewDimension(backend_facts->texture_type, &layout->view_dimension)) {
        return false;
    }
    layout->has_sampler = backend_facts->sampler_type != WEBGPU_SAMPLER_BINDINGKIND_NONE;
    if (layout->has_sampler) {
        layout->sampler_type = WEBGPU_ToWGPUSamplerBindingType(backend_facts->sampler_type);
        if (layout->sampler_type == WGPUSamplerBindingType_Undefined) {
            WEBGPU_SetStringError("invalid sampler binding type layout");
            return false;
        }
    } else {
        layout->sampler_type = WGPUSamplerBindingType_Undefined;
    }
    if (!WEBGPU_ValidateSampledTextureLayoutSupport(
            backend_facts->sample_type,
            backend_facts->sampler_type,
            backend_facts->multisampled)) {
        return false;
    }

    layout->multisampled = backend_facts->multisampled;
    return true;
}

static bool WEBGPU_StorageTextureLayoutFactsToBindingLayout(
    const WebGPUStorageTextureLayoutFacts *backend_facts,
    WebGPUStorageTextureBindingLayout *layout)
{
    layout->access = WEBGPU_ToWGPUStorageTextureAccess(backend_facts->access_kind);
    if (layout->access == WGPUStorageTextureAccess_Undefined) {
        WEBGPU_SetStringError("invalid storage texture access layout");
        return false;
    }
    layout->format = WEBGPU_ToWGPUTextureFormat(backend_facts->format);
    if (layout->format == WGPUTextureFormat_Undefined) {
        WEBGPU_SetStringError("invalid storage texture format layout");
        return false;
    }
    if (!WEBGPU_ToStorageTextureViewDimension(backend_facts->texture_type, &layout->view_dimension)) {
        return false;
    }

    return true;
}

static bool WEBGPU_ApplySDLSampledTextureLayoutFacts(
    WebGPUSampledTextureLayoutFacts *backend_facts,
    const SDL_GPUSampledTextureSlotDescription *slot)
{
    backend_facts->texture_type = slot->texture_type;
    backend_facts->multisampled = false;

    switch (slot->sample_type) {
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_FILTERABLE_FLOAT:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_FILTERABLE_FLOAT;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_UNFILTERABLE_FLOAT:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_DEPTH:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_SINT:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_SINT;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_UINT:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UINT;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_MULTISAMPLED_UNFILTERABLE_FLOAT:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_UNFILTERABLE_FLOAT;
        backend_facts->multisampled = true;
        break;
    case SDL_GPU_SHADERTEXTURESAMPLETYPE_MULTISAMPLED_DEPTH:
        backend_facts->sample_type = WEBGPU_SAMPLED_TEXTURE_SAMPLEKIND_DEPTH;
        backend_facts->multisampled = true;
        break;
    default:
        WEBGPU_SetStringError("invalid sampled texture layout");
        return false;
    }

    switch (slot->sampler_type) {
    case SDL_GPU_SHADERSAMPLERTYPE_FILTERING:
        backend_facts->sampler_type = WEBGPU_SAMPLER_BINDINGKIND_FILTERING;
        break;
    case SDL_GPU_SHADERSAMPLERTYPE_NONFILTERING:
        backend_facts->sampler_type = WEBGPU_SAMPLER_BINDINGKIND_NONFILTERING;
        break;
    case SDL_GPU_SHADERSAMPLERTYPE_COMPARISON:
        backend_facts->sampler_type = WEBGPU_SAMPLER_BINDINGKIND_COMPARISON;
        break;
    case SDL_GPU_SHADERSAMPLERTYPE_NONE:
        backend_facts->sampler_type = WEBGPU_SAMPLER_BINDINGKIND_NONE;
        break;
    default:
        WEBGPU_SetStringError("invalid sampler layout");
        return false;
    }

    return true;
}

static bool WEBGPU_ApplySDLStorageTextureLayoutFacts(
    WebGPUStorageTextureLayoutFacts *backend_facts,
    const SDL_GPUStorageTextureSlotDescription *slot)
{
    backend_facts->format = slot->format;
    backend_facts->texture_type = slot->texture_type;

    switch (slot->access) {
    case SDL_GPU_STORAGETEXTUREACCESS_READ_ONLY:
        backend_facts->access_kind = WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY;
        break;
    case SDL_GPU_STORAGETEXTUREACCESS_WRITE_ONLY:
        backend_facts->access_kind = WEBGPU_STORAGE_TEXTURE_ACCESSKIND_WRITE_ONLY;
        break;
    case SDL_GPU_STORAGETEXTUREACCESS_READ_WRITE:
        backend_facts->access_kind = WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_WRITE;
        break;
    default:
        WEBGPU_SetStringError("invalid storage texture layout");
        return false;
    }

    return true;
}

static bool WEBGPU_ApplySDLShaderResourceLayoutFacts(
    WebGPUShaderResourceLayoutFacts *backend_facts,
    const SDL_GPUShaderResourceLayoutFacts *facts)
{
    for (Uint32 i = 0; facts->sampled_texture_slots && i < facts->num_samplers; i += 1) {
        if (!WEBGPU_ApplySDLSampledTextureLayoutFacts(&backend_facts->samplers[i], &facts->sampled_texture_slots[i])) {
            return false;
        }
    }

    for (Uint32 i = 0; facts->storage_texture_slots && i < facts->num_storage_textures; i += 1) {
        if (!WEBGPU_ApplySDLStorageTextureLayoutFacts(&backend_facts->storage_textures[i], &facts->storage_texture_slots[i])) {
            return false;
        }
    }

    return true;
}

static bool WEBGPU_ApplySDLComputeResourceLayoutFacts(
    WebGPUComputeResourceLayoutFacts *backend_facts,
    const SDL_GPUComputePipelineResourceLayoutFacts *facts)
{
    for (Uint32 i = 0; facts->sampled_texture_slots && i < facts->num_samplers; i += 1) {
        if (!WEBGPU_ApplySDLSampledTextureLayoutFacts(&backend_facts->samplers[i], &facts->sampled_texture_slots[i])) {
            return false;
        }
    }

    for (Uint32 i = 0; facts->readonly_storage_texture_slots && i < facts->num_readonly_storage_textures; i += 1) {
        if (!WEBGPU_ApplySDLStorageTextureLayoutFacts(&backend_facts->readonly_storage_textures[i], &facts->readonly_storage_texture_slots[i])) {
            return false;
        }
    }

    for (Uint32 i = 0; facts->readwrite_storage_texture_slots && i < facts->num_readwrite_storage_textures; i += 1) {
        if (!WEBGPU_ApplySDLStorageTextureLayoutFacts(&backend_facts->readwrite_storage_textures[i], &facts->readwrite_storage_texture_slots[i])) {
            return false;
        }
    }

    return true;
}

static void WEBGPU_InitDefaultShaderResourceLayoutFacts(
    WebGPUShaderResourceLayoutFacts *backend_facts,
    Uint32 num_samplers,
    Uint32 num_storage_textures,
    Uint32 num_storage_buffers,
    Uint32 num_uniform_buffers)
{
    Uint32 sampler_count;
    Uint32 storage_texture_count;

    SDL_zero(*backend_facts);
    backend_facts->sampler_count = num_samplers;
    backend_facts->storage_texture_count = num_storage_textures;
    backend_facts->storage_buffer_count = num_storage_buffers;
    backend_facts->uniform_buffer_count = num_uniform_buffers;

    sampler_count = SDL_min(backend_facts->sampler_count, (Uint32)MAX_TEXTURE_SAMPLERS_PER_STAGE);
    for (Uint32 i = 0; i < sampler_count; i += 1) {
        WEBGPU_InitDefaultSampledTextureLayoutFacts(&backend_facts->samplers[i]);
    }

    storage_texture_count = SDL_min(backend_facts->storage_texture_count, (Uint32)MAX_STORAGE_TEXTURES_PER_STAGE);
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        WEBGPU_InitDefaultStorageTextureLayoutFacts(&backend_facts->storage_textures[i], WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY);
    }
}

static bool WEBGPU_ShaderResourceLayoutFactsToLayout(
    const WebGPUShaderResourceLayoutFacts *backend_facts,
    WebGPUShaderResourceLayout *layout)
{
    Uint32 sampler_count;
    Uint32 storage_texture_count;

    SDL_zero(*layout);
    layout->sampler_count = backend_facts->sampler_count;
    layout->storage_texture_count = backend_facts->storage_texture_count;
    layout->storage_buffer_count = backend_facts->storage_buffer_count;
    layout->uniform_buffer_count = backend_facts->uniform_buffer_count;

    sampler_count = SDL_min(backend_facts->sampler_count, (Uint32)MAX_TEXTURE_SAMPLERS_PER_STAGE);
    for (Uint32 i = 0; i < sampler_count; i += 1) {
        if (!WEBGPU_SampledTextureLayoutFactsToBindingLayout(&backend_facts->samplers[i], &layout->samplers[i])) {
            return false;
        }
    }

    storage_texture_count = SDL_min(backend_facts->storage_texture_count, (Uint32)MAX_STORAGE_TEXTURES_PER_STAGE);
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        if (!WEBGPU_StorageTextureLayoutFactsToBindingLayout(&backend_facts->storage_textures[i], &layout->storage_textures[i])) {
            return false;
        }
    }

    return true;
}

static bool WEBGPU_ValidateShaderResourceLayoutFactCounts(const WebGPUShaderResourceLayoutFacts *backend_facts)
{
    if (backend_facts->sampler_count > MAX_TEXTURE_SAMPLERS_PER_STAGE) {
        WEBGPU_SetStringError("shader sampler count exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE");
        return false;
    }
    if (backend_facts->storage_texture_count > MAX_STORAGE_TEXTURES_PER_STAGE) {
        WEBGPU_SetStringError("shader storage texture count exceeds MAX_STORAGE_TEXTURES_PER_STAGE");
        return false;
    }
    if (backend_facts->storage_buffer_count > MAX_STORAGE_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("shader storage buffer count exceeds MAX_STORAGE_BUFFERS_PER_STAGE");
        return false;
    }
    if (backend_facts->uniform_buffer_count > MAX_UNIFORM_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("shader uniform buffer count exceeds MAX_UNIFORM_BUFFERS_PER_STAGE");
        return false;
    }

    return true;
}

static bool WEBGPU_ValidateShaderResourceLayoutFactSupport(
    WebGPURenderer *renderer,
    const WebGPUShaderResourceLayoutFacts *backend_facts)
{
    if (backend_facts->storage_texture_count > renderer->limits.maxStorageTexturesPerShaderStage) {
        WEBGPU_SetStringError("shader storage texture count exceeds WebGPU maxStorageTexturesPerShaderStage");
        return false;
    }
    if (backend_facts->storage_texture_count > 0 &&
        !renderer->supports_readonly_and_readwrite_storage_textures) {
        WEBGPU_SetStringError("WebGPU read-only storage textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        return false;
    }

    for (Uint32 i = 0; i < backend_facts->storage_texture_count; i += 1) {
        if (backend_facts->storage_textures[i].access_kind == WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY &&
            WEBGPU_StorageTextureReadOnlyFormatRequiresCoreFeaturesAndLimits(backend_facts->storage_textures[i].format) &&
            !renderer->supports_core_features_and_limits) {
            WEBGPU_SetStringError("WebGPU graphics storage read textures with this format require core-features-and-limits");
            return false;
        }
    }

    return true;
}

static bool WEBGPU_InitShaderResourceLayoutFromFacts(
    WebGPURenderer *renderer,
    WebGPUShaderResourceLayout *layout,
    const SDL_GPUShaderResourceLayoutFacts *facts)
{
    WebGPUShaderResourceLayoutFacts backend_facts;

    if (!facts) {
        WEBGPU_SetStringError("shader resource layout facts are required");
        return false;
    }

    // Current SDL WebGPU WGSL defaults. Layout facts override these explicitly;
    // the backend must not infer durable layout by parsing WGSL.
    WEBGPU_InitDefaultShaderResourceLayoutFacts(
        &backend_facts,
        facts->num_samplers,
        facts->num_storage_textures,
        facts->num_storage_buffers,
        facts->num_uniform_buffers);
    if (!WEBGPU_ApplySDLShaderResourceLayoutFacts(&backend_facts, facts)) {
        return false;
    }
    if (!WEBGPU_ValidateShaderResourceLayoutFactCounts(&backend_facts)) {
        return false;
    }
    if (!WEBGPU_ValidateShaderResourceLayoutFactSupport(renderer, &backend_facts)) {
        return false;
    }

    return WEBGPU_ShaderResourceLayoutFactsToLayout(&backend_facts, layout);
}

static bool WEBGPU_InitDefaultShaderResourceLayout(
    WebGPURenderer *renderer,
    WebGPUShaderResourceLayout *layout,
    const SDL_GPUShaderCreateInfo *createinfo,
    const SDL_GPUShaderResourceLayoutFacts *layout_facts)
{
    SDL_GPUShaderResourceLayoutFacts fallback_facts;

    if (layout_facts) {
        return WEBGPU_InitShaderResourceLayoutFromFacts(renderer, layout, layout_facts);
    }

    SDL_zero(fallback_facts);
    fallback_facts.stage = createinfo->stage;
    fallback_facts.num_samplers = createinfo->num_samplers;
    fallback_facts.num_storage_textures = createinfo->num_storage_textures;
    fallback_facts.num_storage_buffers = createinfo->num_storage_buffers;
    fallback_facts.num_uniform_buffers = createinfo->num_uniform_buffers;
    return WEBGPU_InitShaderResourceLayoutFromFacts(renderer, layout, &fallback_facts);
}

static void WEBGPU_InitDefaultComputeResourceLayoutFacts(
    WebGPUComputeResourceLayoutFacts *backend_facts,
    Uint32 num_samplers,
    Uint32 num_readonly_storage_textures,
    Uint32 num_readonly_storage_buffers,
    Uint32 num_readwrite_storage_textures,
    Uint32 num_readwrite_storage_buffers,
    Uint32 num_uniform_buffers)
{
    SDL_zero(*backend_facts);
    backend_facts->sampler_count = num_samplers;
    backend_facts->readonly_storage_texture_count = num_readonly_storage_textures;
    backend_facts->readonly_storage_buffer_count = num_readonly_storage_buffers;
    backend_facts->readwrite_storage_texture_count = num_readwrite_storage_textures;
    backend_facts->readwrite_storage_buffer_count = num_readwrite_storage_buffers;
    backend_facts->uniform_buffer_count = num_uniform_buffers;

    for (Uint32 i = 0; i < SDL_min(backend_facts->sampler_count, (Uint32)MAX_TEXTURE_SAMPLERS_PER_STAGE); i += 1) {
        WEBGPU_InitDefaultSampledTextureLayoutFacts(&backend_facts->samplers[i]);
    }
    for (Uint32 i = 0; i < SDL_min(backend_facts->readonly_storage_texture_count, (Uint32)MAX_STORAGE_TEXTURES_PER_STAGE); i += 1) {
        WEBGPU_InitDefaultStorageTextureLayoutFacts(&backend_facts->readonly_storage_textures[i], WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_ONLY);
    }
    for (Uint32 i = 0; i < SDL_min(backend_facts->readwrite_storage_texture_count, (Uint32)MAX_COMPUTE_WRITE_TEXTURES); i += 1) {
        WEBGPU_InitDefaultStorageTextureLayoutFacts(&backend_facts->readwrite_storage_textures[i], WEBGPU_STORAGE_TEXTURE_ACCESSKIND_WRITE_ONLY);
    }
}

static bool WEBGPU_ValidateComputeResourceLayoutFactCounts(const WebGPUComputeResourceLayoutFacts *backend_facts)
{
    if (backend_facts->sampler_count > MAX_TEXTURE_SAMPLERS_PER_STAGE) {
        WEBGPU_SetStringError("compute sampler count exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE");
        return false;
    }
    if (backend_facts->readonly_storage_texture_count > MAX_STORAGE_TEXTURES_PER_STAGE) {
        WEBGPU_SetStringError("compute read-only storage texture count exceeds MAX_STORAGE_TEXTURES_PER_STAGE");
        return false;
    }
    if (backend_facts->readwrite_storage_texture_count > MAX_COMPUTE_WRITE_TEXTURES) {
        WEBGPU_SetStringError("compute read-write storage texture count exceeds MAX_COMPUTE_WRITE_TEXTURES");
        return false;
    }
    if (backend_facts->readonly_storage_buffer_count > MAX_STORAGE_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("compute read-only storage buffer count exceeds MAX_STORAGE_BUFFERS_PER_STAGE");
        return false;
    }
    if (backend_facts->readwrite_storage_buffer_count > MAX_COMPUTE_WRITE_BUFFERS) {
        WEBGPU_SetStringError("compute read-write storage buffer count exceeds MAX_COMPUTE_WRITE_BUFFERS");
        return false;
    }
    if (backend_facts->uniform_buffer_count > MAX_UNIFORM_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("compute uniform buffer count exceeds MAX_UNIFORM_BUFFERS_PER_STAGE");
        return false;
    }

    return true;
}

static bool WEBGPU_ValidateComputeResourceLayoutFactSupport(
    WebGPURenderer *renderer,
    const WebGPUComputeResourceLayoutFacts *backend_facts)
{
    if (backend_facts->readonly_storage_texture_count + backend_facts->readwrite_storage_texture_count > renderer->limits.maxStorageTexturesPerShaderStage) {
        WEBGPU_SetStringError("compute storage texture count exceeds WebGPU maxStorageTexturesPerShaderStage");
        return false;
    }
    if (backend_facts->readonly_storage_texture_count > 0 &&
        !renderer->supports_readonly_and_readwrite_storage_textures) {
        WEBGPU_SetStringError("WebGPU compute read-only storage textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        return false;
    }

    for (Uint32 i = 0; i < backend_facts->readonly_storage_texture_count; i += 1) {
        SDL_GPUTextureFormat format = backend_facts->readonly_storage_textures[i].format;

        if (WEBGPU_StorageTextureReadOnlyFormatRequiresCoreFeaturesAndLimits(format) &&
            !renderer->supports_core_features_and_limits) {
            WEBGPU_SetStringError("WebGPU compute storage read textures with this format require core-features-and-limits");
            return false;
        }
    }

    for (Uint32 i = 0; i < backend_facts->readwrite_storage_texture_count; i += 1) {
        if (backend_facts->readwrite_storage_textures[i].access_kind == WEBGPU_STORAGE_TEXTURE_ACCESSKIND_READ_WRITE) {
            SDL_GPUTextureFormat format = backend_facts->readwrite_storage_textures[i].format;

            if (!renderer->supports_readonly_and_readwrite_storage_textures) {
                WEBGPU_SetStringError("WebGPU compute read-write storage textures require the readonly_and_readwrite_storage_textures WGSL language feature");
                return false;
            }
            if (WEBGPU_StorageTextureReadWriteFormatRequiresTier2(format) &&
                !renderer->supports_texture_formats_tier2) {
                WEBGPU_SetStringError("WebGPU compute read-write storage textures with this format require the texture-formats-tier2 feature");
                return false;
            }
        }
    }

    return true;
}

static bool WEBGPU_ComputeResourceLayoutFactsToLayout(
    const WebGPUComputeResourceLayoutFacts *backend_facts,
    WebGPUComputeResourceLayout *layout)
{
    SDL_zero(*layout);
    layout->sampler_count = backend_facts->sampler_count;
    layout->readonly_storage_texture_count = backend_facts->readonly_storage_texture_count;
    layout->readonly_storage_buffer_count = backend_facts->readonly_storage_buffer_count;
    layout->readwrite_storage_texture_count = backend_facts->readwrite_storage_texture_count;
    layout->readwrite_storage_buffer_count = backend_facts->readwrite_storage_buffer_count;
    layout->uniform_buffer_count = backend_facts->uniform_buffer_count;

    for (Uint32 i = 0; i < backend_facts->sampler_count; i += 1) {
        if (!WEBGPU_SampledTextureLayoutFactsToBindingLayout(&backend_facts->samplers[i], &layout->samplers[i])) {
            return false;
        }
    }
    for (Uint32 i = 0; i < backend_facts->readonly_storage_texture_count; i += 1) {
        if (!WEBGPU_StorageTextureLayoutFactsToBindingLayout(&backend_facts->readonly_storage_textures[i], &layout->readonly_storage_textures[i])) {
            return false;
        }
    }
    for (Uint32 i = 0; i < backend_facts->readwrite_storage_texture_count; i += 1) {
        if (!WEBGPU_StorageTextureLayoutFactsToBindingLayout(&backend_facts->readwrite_storage_textures[i], &layout->readwrite_storage_textures[i])) {
            return false;
        }
    }

    return true;
}

static bool WEBGPU_InitDefaultComputeResourceLayout(
    WebGPURenderer *renderer,
    WebGPUComputeResourceLayout *layout,
    const SDL_GPUComputePipelineCreateInfo *createinfo,
    const SDL_GPUComputePipelineResourceLayoutFacts *layout_facts)
{
    WebGPUComputeResourceLayoutFacts backend_facts;

    // Current default WGSL compute layout. Layout facts should populate this
    // layout instead of being consumed directly by bind code.
    WEBGPU_InitDefaultComputeResourceLayoutFacts(
        &backend_facts,
        createinfo->num_samplers,
        createinfo->num_readonly_storage_textures,
        createinfo->num_readonly_storage_buffers,
        createinfo->num_readwrite_storage_textures,
        createinfo->num_readwrite_storage_buffers,
        createinfo->num_uniform_buffers);
    if (!WEBGPU_ValidateComputeResourceLayoutFactCounts(&backend_facts)) {
        return false;
    }
    if (layout_facts && !WEBGPU_ApplySDLComputeResourceLayoutFacts(&backend_facts, layout_facts)) {
        return false;
    }
    if (!WEBGPU_ValidateComputeResourceLayoutFactSupport(renderer, &backend_facts)) {
        return false;
    }

    return WEBGPU_ComputeResourceLayoutFactsToLayout(&backend_facts, layout);
}

static void WEBGPU_AppendSampledTextureBindGroupLayoutEntries(
    WGPUBindGroupLayoutEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    const WebGPUSampledTextureBindingLayout *sampled_textures,
    Uint32 sampled_texture_count,
    WGPUShaderStage visibility)
{
    for (Uint32 i = 0; i < sampled_texture_count; i += 1) {
        const WebGPUSampledTextureBindingLayout *sampled_texture = &sampled_textures[i];
        WGPUBindGroupLayoutEntry *texture_entry = &entries[*entry_count];

        *texture_entry = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        texture_entry->binding = first_binding + (i * 2);
        texture_entry->visibility = visibility;
        texture_entry->texture.sampleType = sampled_texture->sample_type;
        texture_entry->texture.viewDimension = sampled_texture->view_dimension;
        texture_entry->texture.multisampled = sampled_texture->multisampled;
        *entry_count += 1;

        if (sampled_texture->has_sampler) {
            WGPUBindGroupLayoutEntry *sampler_entry = &entries[*entry_count];

            *sampler_entry = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            sampler_entry->binding = first_binding + (i * 2) + 1;
            sampler_entry->visibility = visibility;
            sampler_entry->sampler.type = sampled_texture->sampler_type;
            *entry_count += 1;
        }
    }
}

static void WEBGPU_AppendStorageTextureBindGroupLayoutEntries(
    WGPUBindGroupLayoutEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    const WebGPUStorageTextureBindingLayout *storage_textures,
    Uint32 storage_texture_count,
    WGPUShaderStage visibility)
{
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        WGPUBindGroupLayoutEntry *storage_texture_entry = &entries[*entry_count];

        *storage_texture_entry = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        storage_texture_entry->binding = first_binding + i;
        storage_texture_entry->visibility = visibility;
        storage_texture_entry->storageTexture.access = storage_textures[i].access;
        storage_texture_entry->storageTexture.format = storage_textures[i].format;
        storage_texture_entry->storageTexture.viewDimension = storage_textures[i].view_dimension;
        *entry_count += 1;
    }
}

static void WEBGPU_AppendStorageBufferBindGroupLayoutEntries(
    WGPUBindGroupLayoutEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    Uint32 storage_buffer_count,
    WGPUShaderStage visibility,
    WGPUBufferBindingType buffer_type)
{
    for (Uint32 i = 0; i < storage_buffer_count; i += 1) {
        WGPUBindGroupLayoutEntry *storage_buffer_entry = &entries[*entry_count];

        *storage_buffer_entry = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        storage_buffer_entry->binding = first_binding + i;
        storage_buffer_entry->visibility = visibility;
        storage_buffer_entry->buffer.type = buffer_type;
        *entry_count += 1;
    }
}

static void WEBGPU_AppendUniformBufferBindGroupLayoutEntries(
    WGPUBindGroupLayoutEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    Uint32 uniform_buffer_count,
    WGPUShaderStage visibility)
{
    for (Uint32 i = 0; i < uniform_buffer_count; i += 1) {
        WGPUBindGroupLayoutEntry *uniform_entry = &entries[*entry_count];

        *uniform_entry = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        uniform_entry->binding = first_binding + i;
        uniform_entry->visibility = visibility;
        uniform_entry->buffer.type = WGPUBufferBindingType_Uniform;
        uniform_entry->buffer.hasDynamicOffset = true;
        *entry_count += 1;
    }
}

static WGPUBindGroupLayout WEBGPU_CreateGraphicsBindGroupLayout(
    WebGPURenderer *renderer,
    const WebGPUShaderResourceLayout *shader_layout,
    bool resource_group,
    WGPUShaderStage visibility,
    Uint32 *entry_count_out)
{
    WGPUBindGroupLayoutEntry entries[MAX_TEXTURE_SAMPLERS_PER_STAGE * 2 + MAX_STORAGE_TEXTURES_PER_STAGE + MAX_STORAGE_BUFFERS_PER_STAGE + MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroupLayoutDescriptor layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    Uint32 sampler_count = resource_group ? shader_layout->sampler_count : 0;
    Uint32 storage_texture_count = resource_group ? shader_layout->storage_texture_count : 0;
    Uint32 storage_buffer_count = resource_group ? shader_layout->storage_buffer_count : 0;
    Uint32 uniform_count = resource_group ? 0 : shader_layout->uniform_buffer_count;
    Uint32 entry_count = 0;

    if (sampler_count > MAX_TEXTURE_SAMPLERS_PER_STAGE) {
        SDL_SetError("sampler count exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE");
        return NULL;
    }
    if (storage_texture_count > MAX_STORAGE_TEXTURES_PER_STAGE) {
        SDL_SetError("storage texture count exceeds MAX_STORAGE_TEXTURES_PER_STAGE");
        return NULL;
    }
    if (storage_texture_count > renderer->limits.maxStorageTexturesPerShaderStage) {
        SDL_SetError("storage texture count exceeds WebGPU maxStorageTexturesPerShaderStage");
        return NULL;
    }
    if (storage_buffer_count > MAX_STORAGE_BUFFERS_PER_STAGE) {
        SDL_SetError("storage buffer count exceeds MAX_STORAGE_BUFFERS_PER_STAGE");
        return NULL;
    }
    if (uniform_count > MAX_UNIFORM_BUFFERS_PER_STAGE) {
        SDL_SetError("uniform count exceeds MAX_UNIFORM_BUFFERS_PER_STAGE");
        return NULL;
    }

    WEBGPU_AppendSampledTextureBindGroupLayoutEntries(
        entries,
        &entry_count,
        0,
        shader_layout->samplers,
        sampler_count,
        visibility);
    WEBGPU_AppendStorageTextureBindGroupLayoutEntries(
        entries,
        &entry_count,
        sampler_count * 2,
        shader_layout->storage_textures,
        storage_texture_count,
        visibility);
    WEBGPU_AppendStorageBufferBindGroupLayoutEntries(
        entries,
        &entry_count,
        sampler_count * 2 + storage_texture_count,
        storage_buffer_count,
        visibility,
        WGPUBufferBindingType_ReadOnlyStorage);
    WEBGPU_AppendUniformBufferBindGroupLayoutEntries(
        entries,
        &entry_count,
        0,
        uniform_count,
        visibility);

    layout_desc.entryCount = entry_count;
    layout_desc.entries = entry_count > 0 ? entries : NULL;
    if (entry_count_out) {
        *entry_count_out = entry_count;
    }
    return wgpuDeviceCreateBindGroupLayout(renderer->device, &layout_desc);
}

static WGPUBindGroupLayout WEBGPU_CreateComputeBindGroupLayout(
    WebGPURenderer *renderer,
    const WebGPUComputeResourceLayout *compute_layout,
    Uint32 group_index,
    Uint32 *entry_count_out)
{
    WGPUBindGroupLayoutEntry entries[MAX_TEXTURE_SAMPLERS_PER_STAGE * 2 + MAX_STORAGE_TEXTURES_PER_STAGE + MAX_STORAGE_BUFFERS_PER_STAGE + MAX_COMPUTE_WRITE_TEXTURES + MAX_COMPUTE_WRITE_BUFFERS + MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroupLayoutDescriptor layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    Uint32 entry_count = 0;

    if (group_index == WEBGPU_COMPUTE_READONLY_GROUP) {
        WEBGPU_AppendSampledTextureBindGroupLayoutEntries(
            entries,
            &entry_count,
            0,
            compute_layout->samplers,
            compute_layout->sampler_count,
            WGPUShaderStage_Compute);
        WEBGPU_AppendStorageTextureBindGroupLayoutEntries(
            entries,
            &entry_count,
            compute_layout->sampler_count * 2,
            compute_layout->readonly_storage_textures,
            compute_layout->readonly_storage_texture_count,
            WGPUShaderStage_Compute);
        WEBGPU_AppendStorageBufferBindGroupLayoutEntries(
            entries,
            &entry_count,
            compute_layout->sampler_count * 2 + compute_layout->readonly_storage_texture_count,
            compute_layout->readonly_storage_buffer_count,
            WGPUShaderStage_Compute,
            WGPUBufferBindingType_ReadOnlyStorage);
    } else if (group_index == WEBGPU_COMPUTE_READWRITE_GROUP) {
        WEBGPU_AppendStorageTextureBindGroupLayoutEntries(
            entries,
            &entry_count,
            0,
            compute_layout->readwrite_storage_textures,
            compute_layout->readwrite_storage_texture_count,
            WGPUShaderStage_Compute);
        WEBGPU_AppendStorageBufferBindGroupLayoutEntries(
            entries,
            &entry_count,
            compute_layout->readwrite_storage_texture_count,
            compute_layout->readwrite_storage_buffer_count,
            WGPUShaderStage_Compute,
            WGPUBufferBindingType_Storage);
    } else if (group_index == WEBGPU_COMPUTE_UNIFORM_GROUP) {
        WEBGPU_AppendUniformBufferBindGroupLayoutEntries(
            entries,
            &entry_count,
            0,
            compute_layout->uniform_buffer_count,
            WGPUShaderStage_Compute);
    }

    layout_desc.entryCount = entry_count;
    layout_desc.entries = entry_count > 0 ? entries : NULL;
    if (entry_count_out) {
        *entry_count_out = entry_count;
    }
    return wgpuDeviceCreateBindGroupLayout(renderer->device, &layout_desc);
}

static SDL_GPUComputePipeline *WEBGPU_CreateComputePipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUComputePipelineCreateInfo *createinfo,
    const SDL_GPUComputePipelineResourceLayoutFacts *layout_facts)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    WGPUShaderModuleDescriptor shader_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    WGPUShaderModule shader_module = NULL;
    WGPUBindGroupLayout bind_group_layouts[WEBGPU_COMPUTE_BIND_GROUP_COUNT] = { NULL };
    Uint32 bind_group_layout_entry_counts[WEBGPU_COMPUTE_BIND_GROUP_COUNT] = { 0 };
    WebGPUComputeResourceLayout compute_layout;
    WGPUPipelineLayoutDescriptor layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    WGPUPipelineLayout layout = NULL;
    WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    WebGPUComputePipeline *pipeline = NULL;
    const char *entrypoint = createinfo->entrypoint ? createinfo->entrypoint : "main";
    const char *debug_name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_COMPUTEPIPELINE_CREATE_NAME_STRING, NULL);
    Uint32 bind_group_layout_count = 0;
    Uint32 highest_used_bind_group = 0;
    bool needs_bind_group_layouts = false;

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateComputePipeline")) {
        return NULL;
    }
    if (createinfo->format != SDL_GPU_SHADERFORMAT_WGSL) {
        WEBGPU_SetStringError("only WGSL compute shaders are supported");
        return NULL;
    }
    if (createinfo->threadcount_x == 0 ||
        createinfo->threadcount_y == 0 ||
        createinfo->threadcount_z == 0) {
        WEBGPU_SetStringError("compute pipeline threadCount dimensions must be at least 1");
        return NULL;
    }
    if (!WEBGPU_InitDefaultComputeResourceLayout(renderer, &compute_layout, createinfo, layout_facts)) {
        return NULL;
    }
    if (!WEBGPU_CanWaitForErrorScope(renderer, "CreateComputePipeline")) {
        return NULL;
    }
    wgsl.code = WEBGPU_ByteView(createinfo->code, createinfo->code_size);
    shader_desc.nextInChain = &wgsl.chain;

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    shader_module = wgpuDeviceCreateShaderModule(renderer->device, &shader_desc);
    if (!WEBGPU_PopErrorScope(renderer, "CreateShaderModule for compute pipeline")) {
        if (shader_module) {
            wgpuShaderModuleRelease(shader_module);
        }
        return NULL;
    }
    if (!shader_module) {
        WEBGPU_SetStringError("CreateShaderModule failed for compute pipeline");
        return NULL;
    }

    if (compute_layout.sampler_count > 0 ||
        compute_layout.readonly_storage_texture_count > 0 ||
        compute_layout.readonly_storage_buffer_count > 0) {
        highest_used_bind_group = WEBGPU_COMPUTE_READONLY_GROUP;
        needs_bind_group_layouts = true;
    }
    if (compute_layout.readwrite_storage_texture_count > 0 ||
        compute_layout.readwrite_storage_buffer_count > 0) {
        highest_used_bind_group = WEBGPU_COMPUTE_READWRITE_GROUP;
        needs_bind_group_layouts = true;
    }
    if (compute_layout.uniform_buffer_count > 0) {
        highest_used_bind_group = WEBGPU_COMPUTE_UNIFORM_GROUP;
        needs_bind_group_layouts = true;
    }
    if (needs_bind_group_layouts) {
        bind_group_layout_count = highest_used_bind_group + 1;
        if (bind_group_layout_count > renderer->limits.maxBindGroups) {
            WEBGPU_SetStringError("compute bind group layout count exceeds WebGPU maxBindGroups");
            wgpuShaderModuleRelease(shader_module);
            return NULL;
        }

        for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
            bind_group_layouts[i] = WEBGPU_CreateComputeBindGroupLayout(renderer, &compute_layout, i, &bind_group_layout_entry_counts[i]);
            if (!bind_group_layouts[i]) {
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                WEBGPU_SetStringError("CreateBindGroupLayout failed for compute pipeline");
                wgpuShaderModuleRelease(shader_module);
                return NULL;
            }
        }
    }

    layout_desc.bindGroupLayoutCount = bind_group_layout_count;
    layout_desc.bindGroupLayouts = bind_group_layout_count > 0 ? bind_group_layouts : NULL;
    layout = wgpuDeviceCreatePipelineLayout(renderer->device, &layout_desc);
    if (!layout) {
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        WEBGPU_SetStringError("CreatePipelineLayout failed for compute pipeline");
        wgpuShaderModuleRelease(shader_module);
        return NULL;
    }

    pipeline = (WebGPUComputePipeline *)SDL_calloc(1, sizeof(*pipeline));
    if (!pipeline) {
        wgpuPipelineLayoutRelease(layout);
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        wgpuShaderModuleRelease(shader_module);
        return NULL;
    }
    SDL_SetAtomicInt(&pipeline->refcount, 1);

    pipeline_desc.layout = layout;
    pipeline_desc.label = WEBGPU_StringView(debug_name);
    pipeline_desc.compute.module = shader_module;
    pipeline_desc.compute.entryPoint = WEBGPU_StringView(entrypoint);

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    pipeline->pipeline = wgpuDeviceCreateComputePipeline(renderer->device, &pipeline_desc);
    if (!WEBGPU_PopErrorScope(renderer, "CreateComputePipeline")) {
        if (pipeline->pipeline) {
            wgpuComputePipelineRelease(pipeline->pipeline);
        }
        wgpuPipelineLayoutRelease(layout);
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        wgpuShaderModuleRelease(shader_module);
        SDL_free(pipeline);
        return NULL;
    }
    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(shader_module);
    if (!pipeline->pipeline) {
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        SDL_free(pipeline);
        WEBGPU_SetStringError("CreateComputePipeline failed");
        return NULL;
    }

    for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
        pipeline->bind_group_layouts[i] = bind_group_layouts[i];
    }
    pipeline->bind_group_layout_count = bind_group_layout_count;
    if (!WEBGPU_CreatePipelineEmptyBindGroups(
            renderer,
            bind_group_layouts,
            bind_group_layout_entry_counts,
            bind_group_layout_count,
            pipeline->empty_bind_groups,
            WEBGPU_BIND_GROUP_PATH_COMPUTE_EMPTY,
            "CreateBindGroup for empty compute pipeline layout")) {
        WEBGPU_DestroyComputePipeline(pipeline);
        return NULL;
    }
    pipeline->header.numSamplers = compute_layout.sampler_count;
    pipeline->header.numReadonlyStorageTextures = compute_layout.readonly_storage_texture_count;
    pipeline->header.numReadonlyStorageBuffers = compute_layout.readonly_storage_buffer_count;
    pipeline->header.numReadWriteStorageTextures = compute_layout.readwrite_storage_texture_count;
    pipeline->header.numReadWriteStorageBuffers = compute_layout.readwrite_storage_buffer_count;
    pipeline->header.numUniformBuffers = compute_layout.uniform_buffer_count;
    for (Uint32 i = 0; i < pipeline->header.numReadWriteStorageTextures; i += 1) {
        pipeline->header.readWriteStorageTextureTypes[i] =
            WEBGPU_StorageTextureTypeFromViewDimension(compute_layout.readwrite_storage_textures[i].view_dimension);
        pipeline->header.readWriteStorageTextureTypesKnown[i] = true;
    }
    pipeline->resources = compute_layout;

    return (SDL_GPUComputePipeline *)pipeline;
}

static SDL_GPUGraphicsPipeline *WEBGPU_CreateGraphicsPipeline(
    SDL_GPURenderer *driverData,
    const SDL_GPUGraphicsPipelineCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUShader *vertex_shader = (WebGPUShader *)createinfo->vertex_shader;
    WebGPUShader *fragment_shader = (WebGPUShader *)createinfo->fragment_shader;
    const char *debug_name = SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_GRAPHICSPIPELINE_CREATE_NAME_STRING, NULL);
    WGPUBlendState blend_states[MAX_COLOR_TARGET_BINDINGS];
    WGPUColorTargetState color_targets[MAX_COLOR_TARGET_BINDINGS];
    WGPUFragmentState fragment_state = WGPU_FRAGMENT_STATE_INIT;
    WGPUDepthStencilState depth_stencil_state = WGPU_DEPTH_STENCIL_STATE_INIT;
    WGPUPipelineLayoutDescriptor layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    WGPUPipelineLayout layout;
    WGPURenderPipelineDescriptor pipeline_desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    WGPUBindGroupLayout bind_group_layouts[WEBGPU_GRAPHICS_BIND_GROUP_COUNT] = { NULL };
    Uint32 bind_group_layout_entry_counts[WEBGPU_GRAPHICS_BIND_GROUP_COUNT] = { 0 };
    Uint32 bind_group_layout_count = 0;
    Uint32 highest_used_bind_group = 0;
    bool needs_bind_group_layouts = false;
    WGPUVertexBufferLayout vertex_buffers[MAX_VERTEX_BUFFERS];
    WGPUVertexAttribute vertex_attributes[MAX_VERTEX_ATTRIBUTES];
    Uint64 vertex_buffer_strides[MAX_VERTEX_BUFFERS] = { 0 };
    Uint64 vertex_buffer_last_strides[MAX_VERTEX_BUFFERS] = { 0 };
    Uint32 vertex_buffer_wgpu_slots[MAX_VERTEX_BUFFERS];
    Uint32 described_vertex_buffer_mask = 0;
    Uint32 vertex_attribute_index = 0;
    Uint32 vertex_buffer_count = 0;
    Uint32 required_vertex_buffer_mask = 0;
    Uint32 sample_count;
    WGPUPrimitiveTopology primitive_topology;
    WGPUFrontFace front_face;
    WGPUCullMode cull_mode;
    bool vertex_buffer_instance_step[MAX_VERTEX_BUFFERS] = { false };
    WebGPUGraphicsPipeline *pipeline;

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateGraphicsPipeline")) {
        return NULL;
    }
    if (!vertex_shader) {
        WEBGPU_SetStringError("vertex shader is required");
        return NULL;
    }
    if (!fragment_shader) {
        WEBGPU_SetStringError("fragment shader is required");
        return NULL;
    }
    if (vertex_shader->stage != SDL_GPU_SHADERSTAGE_VERTEX) {
        WEBGPU_SetStringError("vertex shader stage must be SDL_GPU_SHADERSTAGE_VERTEX");
        return NULL;
    }
    if (fragment_shader->stage != SDL_GPU_SHADERSTAGE_FRAGMENT) {
        WEBGPU_SetStringError("fragment shader stage must be SDL_GPU_SHADERSTAGE_FRAGMENT");
        return NULL;
    }

    if (createinfo->target_info.num_color_targets == 0 && !createinfo->target_info.has_depth_stencil_target) {
        WEBGPU_SetStringError("at least one color or depth-stencil target is required");
        return NULL;
    }
    if (createinfo->target_info.num_color_targets > 0 && !createinfo->target_info.color_target_descriptions) {
        WEBGPU_SetStringError("color target descriptions are required");
        return NULL;
    }
    sample_count = WEBGPU_ToSampleCount(createinfo->multisample_state.sample_count);
    if (sample_count == 0) {
        WEBGPU_SetStringError("unsupported sample count");
        return NULL;
    }
    if (createinfo->multisample_state.enable_mask) {
        WEBGPU_SetStringError("multisample enable_mask must be false");
        return NULL;
    }
    if (createinfo->multisample_state.sample_mask != 0) {
        WEBGPU_SetStringError("multisample sample_mask must be 0");
        return NULL;
    }
    if (createinfo->multisample_state.enable_alpha_to_coverage &&
        createinfo->target_info.num_color_targets == 0) {
        WEBGPU_SetStringError("alpha-to-coverage requires a color target");
        return NULL;
    }
    primitive_topology = WEBGPU_ToPrimitiveTopology(createinfo->primitive_type);
    if (primitive_topology == WGPUPrimitiveTopology_Undefined) {
        WEBGPU_SetStringError("invalid primitive type");
        return NULL;
    }
    if (createinfo->rasterizer_state.fill_mode == SDL_GPU_FILLMODE_LINE) {
        WEBGPU_SetStringError("WebGPU does not support SDL_GPU_FILLMODE_LINE polygon fill mode");
        return NULL;
    }
    if (!WEBGPU_IsValidFillMode(createinfo->rasterizer_state.fill_mode)) {
        WEBGPU_SetStringError("invalid rasterizer fill mode");
        return NULL;
    }
    front_face = WEBGPU_ToFrontFace(createinfo->rasterizer_state.front_face);
    if (front_face == WGPUFrontFace_Undefined) {
        WEBGPU_SetStringError("invalid rasterizer front face");
        return NULL;
    }
    cull_mode = WEBGPU_ToCullMode(createinfo->rasterizer_state.cull_mode);
    if (cull_mode == WGPUCullMode_Undefined) {
        WEBGPU_SetStringError("invalid rasterizer cull mode");
        return NULL;
    }
    if (!createinfo->rasterizer_state.enable_depth_clip &&
        !renderer->supports_depth_clip_control) {
        WEBGPU_SetStringError("disabled depth clip requires WebGPU DepthClipControl feature");
        return NULL;
    }
    if (createinfo->vertex_input_state.num_vertex_buffers > 0 &&
        !createinfo->vertex_input_state.vertex_buffer_descriptions) {
        WEBGPU_SetStringError("vertex buffer descriptions are required");
        return NULL;
    }
    if (createinfo->vertex_input_state.num_vertex_buffers > MAX_VERTEX_BUFFERS) {
        WEBGPU_SetStringError("vertex buffer description count exceeds MAX_VERTEX_BUFFERS");
        return NULL;
    }
    if (createinfo->vertex_input_state.num_vertex_attributes > 0 &&
        !createinfo->vertex_input_state.vertex_attributes) {
        WEBGPU_SetStringError("vertex attributes are required");
        return NULL;
    }
    if (createinfo->vertex_input_state.num_vertex_attributes > MAX_VERTEX_ATTRIBUTES) {
        WEBGPU_SetStringError("vertex attribute count exceeds MAX_VERTEX_ATTRIBUTES");
        return NULL;
    }
    for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_attributes; i += 1) {
        const SDL_GPUVertexAttribute *attribute = &createinfo->vertex_input_state.vertex_attributes[i];

        for (Uint32 j = 0; j < i; j += 1) {
            if (createinfo->vertex_input_state.vertex_attributes[j].location == attribute->location) {
                WEBGPU_SetStringError("duplicate vertex attribute location");
                return NULL;
            }
        }
    }
    if (createinfo->target_info.num_color_targets > SDL_min(renderer->limits.maxColorAttachments, (Uint32)MAX_COLOR_TARGET_BINDINGS)) {
        WEBGPU_SetStringError("color target count exceeds WebGPU maxColorAttachments");
        return NULL;
    }
    if (createinfo->target_info.has_depth_stencil_target) {
        WGPUTextureFormat depth_format;
        const bool has_stencil = WEBGPU_IsStencilFormat(createinfo->target_info.depth_stencil_format);

        if (!WEBGPU_IsDepthStencilFormat(createinfo->target_info.depth_stencil_format)) {
            WEBGPU_SetStringError("unsupported depth-stencil target format");
            return NULL;
        }
        if (IsD24Format(createinfo->target_info.depth_stencil_format) &&
            createinfo->multisample_state.sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("WebGPU backend D24 depth formats require sample count 1 graphics pipelines");
            return NULL;
        }
        if (createinfo->target_info.depth_stencil_format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT &&
            !renderer->supports_depth32float_stencil8) {
            WEBGPU_SetStringError("D32_FLOAT_S8_UINT requires WebGPU Depth32FloatStencil8 feature");
            return NULL;
        }
        if (createinfo->depth_stencil_state.enable_stencil_test && !has_stencil) {
            WEBGPU_SetStringError("stencil testing requires a stencil target format");
            return NULL;
        }
        depth_format = WEBGPU_ToWGPUTextureFormat(createinfo->target_info.depth_stencil_format);
        if (depth_format == WGPUTextureFormat_Undefined) {
            WEBGPU_SetStringError("unsupported depth-stencil target format");
            return NULL;
        }

        depth_stencil_state.format = depth_format;
        depth_stencil_state.depthWriteEnabled =
            createinfo->depth_stencil_state.enable_depth_test && createinfo->depth_stencil_state.enable_depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
        if (createinfo->depth_stencil_state.enable_depth_test) {
            depth_stencil_state.depthCompare = WEBGPU_ToCompareFunction(createinfo->depth_stencil_state.compare_op);
            if (depth_stencil_state.depthCompare == WGPUCompareFunction_Undefined) {
                WEBGPU_SetStringError("invalid depth compare op");
                return NULL;
            }
        } else {
            /* WebGPU has no pipeline depth-test enable bit; Always with writes
             * disabled preserves stencil-only SDL pipelines. */
            depth_stencil_state.depthCompare = WGPUCompareFunction_Always;
        }
        if (createinfo->depth_stencil_state.enable_stencil_test) {
            if (!WEBGPU_InitStencilFaceState(&depth_stencil_state.stencilFront, &createinfo->depth_stencil_state.front_stencil_state) ||
                !WEBGPU_InitStencilFaceState(&depth_stencil_state.stencilBack, &createinfo->depth_stencil_state.back_stencil_state)) {
                return NULL;
            }
            depth_stencil_state.stencilReadMask = createinfo->depth_stencil_state.compare_mask;
            depth_stencil_state.stencilWriteMask = createinfo->depth_stencil_state.write_mask;
        }
        if (!WEBGPU_InitDepthBiasState(renderer, &depth_stencil_state, &createinfo->rasterizer_state, createinfo->primitive_type)) {
            return NULL;
        }
    }
    for (Uint32 i = 0; i < MAX_VERTEX_BUFFERS; i += 1) {
        vertex_buffers[i] = (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        vertex_buffer_wgpu_slots[i] = MAX_VERTEX_BUFFERS;
    }

    for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_buffers; i += 1) {
        const SDL_GPUVertexBufferDescription *buffer_desc = &createinfo->vertex_input_state.vertex_buffer_descriptions[i];
        WGPUVertexBufferLayout *buffer_layout;
        Uint32 attribute_start = vertex_attribute_index;
        Uint64 buffer_last_stride = 0;
        WGPUVertexStepMode step_mode;

        if (buffer_desc->instance_step_rate != 0) {
            WEBGPU_SetStringError("vertex buffer instance_step_rate must be 0");
            return NULL;
        }
        if (buffer_desc->slot >= MAX_VERTEX_BUFFERS) {
            WEBGPU_SetStringError("vertex buffer slot exceeds MAX_VERTEX_BUFFERS");
            return NULL;
        }
        if (described_vertex_buffer_mask & (1u << buffer_desc->slot)) {
            WEBGPU_SetStringError("duplicate vertex buffer slot");
            return NULL;
        }
        described_vertex_buffer_mask |= 1u << buffer_desc->slot;

        step_mode = WEBGPU_ToVertexStepMode(buffer_desc->input_rate);
        if (step_mode == WGPUVertexStepMode_Undefined) {
            WEBGPU_SetStringError("invalid vertex input rate");
            return NULL;
        }

        buffer_layout = &vertex_buffers[vertex_buffer_count];
        buffer_layout->arrayStride = buffer_desc->pitch;
        buffer_layout->stepMode = step_mode;
        buffer_layout->attributes = &vertex_attributes[vertex_attribute_index];

        for (Uint32 j = 0; j < createinfo->vertex_input_state.num_vertex_attributes; j += 1) {
            const SDL_GPUVertexAttribute *attribute = &createinfo->vertex_input_state.vertex_attributes[j];
            WGPUVertexFormat vertex_format;
            Uint32 vertex_format_size;
            Uint32 vertex_format_alignment;
            Uint64 attribute_end;

            if (attribute->buffer_slot != buffer_desc->slot) {
                continue;
            }

            vertex_format = WEBGPU_ToVertexFormat(attribute->format);
            if (vertex_format == WGPUVertexFormat_Force32) {
                WEBGPU_SetStringError("unsupported vertex attribute format");
                return NULL;
            }
            vertex_format_size = WEBGPU_VertexElementFormatByteSize(attribute->format);
            if (vertex_format_size == 0) {
                WEBGPU_SetStringError("unsupported vertex attribute format");
                return NULL;
            }
            vertex_format_alignment = WEBGPU_VertexElementFormatAlignment(attribute->format);
            if (vertex_format_alignment == 0 || (attribute->offset % vertex_format_alignment) != 0) {
                WEBGPU_SetStringError("vertex attribute offset is not aligned for WebGPU");
                return NULL;
            }
            if (attribute->location >= SDL_min(renderer->limits.maxVertexAttributes, (Uint32)MAX_VERTEX_ATTRIBUTES)) {
                WEBGPU_SetStringError("vertex attribute location exceeds WebGPU maxVertexAttributes");
                return NULL;
            }
            attribute_end = (Uint64)attribute->offset + vertex_format_size;

            vertex_attributes[vertex_attribute_index] = (WGPUVertexAttribute)WGPU_VERTEX_ATTRIBUTE_INIT;
            vertex_attributes[vertex_attribute_index].format = vertex_format;
            vertex_attributes[vertex_attribute_index].offset = attribute->offset;
            vertex_attributes[vertex_attribute_index].shaderLocation = attribute->location;
            vertex_attribute_index += 1;
            buffer_layout->attributeCount += 1;
            buffer_last_stride = SDL_max(buffer_last_stride, attribute_end);
        }

        if (buffer_layout->attributeCount > 0) {
            if (buffer_desc->pitch > renderer->limits.maxVertexBufferArrayStride) {
                WEBGPU_SetStringError("vertex buffer pitch exceeds WebGPU maxVertexBufferArrayStride");
                return NULL;
            }
            if ((buffer_desc->pitch % 4) != 0) {
                WEBGPU_SetStringError("vertex buffer pitch must be a multiple of 4 for WebGPU");
                return NULL;
            }
            if (buffer_desc->pitch == 0 && buffer_last_stride > renderer->limits.maxVertexBufferArrayStride) {
                WEBGPU_SetStringError("vertex attribute range exceeds WebGPU maxVertexBufferArrayStride");
                return NULL;
            }
            if (buffer_desc->pitch != 0 && buffer_last_stride > buffer_desc->pitch) {
                WEBGPU_SetStringError("vertex attribute range exceeds vertex buffer pitch");
                return NULL;
            }
            vertex_buffer_strides[buffer_desc->slot] = buffer_desc->pitch;
            vertex_buffer_last_strides[buffer_desc->slot] = buffer_last_stride;
            vertex_buffer_instance_step[buffer_desc->slot] = buffer_desc->input_rate == SDL_GPU_VERTEXINPUTRATE_INSTANCE;
            vertex_buffer_wgpu_slots[buffer_desc->slot] = vertex_buffer_count;
            required_vertex_buffer_mask |= 1u << buffer_desc->slot;
            vertex_buffer_count += 1;
        } else {
            vertex_attribute_index = attribute_start;
            *buffer_layout = (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        }
    }

    if (vertex_buffer_count > renderer->limits.maxVertexBuffers) {
        WEBGPU_SetStringError("vertex buffer layout count exceeds WebGPU maxVertexBuffers");
        return NULL;
    }
    if (vertex_attribute_index > renderer->limits.maxVertexAttributes) {
        WEBGPU_SetStringError("vertex attribute count exceeds WebGPU maxVertexAttributes");
        return NULL;
    }

    for (Uint32 i = 0; i < createinfo->vertex_input_state.num_vertex_attributes; i += 1) {
        const SDL_GPUVertexAttribute *attribute = &createinfo->vertex_input_state.vertex_attributes[i];

        if (attribute->buffer_slot >= MAX_VERTEX_BUFFERS) {
            WEBGPU_SetStringError("vertex attribute buffer slot exceeds MAX_VERTEX_BUFFERS");
            return NULL;
        }
        if ((described_vertex_buffer_mask & (1u << attribute->buffer_slot)) == 0) {
            WEBGPU_SetStringError("vertex attribute references missing vertex buffer slot");
            return NULL;
        }
    }

    if (vertex_shader->resources.sampler_count > 0 || vertex_shader->resources.storage_texture_count > 0 || vertex_shader->resources.storage_buffer_count > 0) {
        highest_used_bind_group = WEBGPU_VERTEX_RESOURCE_GROUP;
        needs_bind_group_layouts = true;
    }
    if (vertex_shader->resources.uniform_buffer_count > 0) {
        highest_used_bind_group = SDL_max(highest_used_bind_group, WEBGPU_VERTEX_UNIFORM_GROUP);
        needs_bind_group_layouts = true;
    }
    if (fragment_shader->resources.sampler_count > 0 ||
        fragment_shader->resources.storage_texture_count > 0 ||
        fragment_shader->resources.storage_buffer_count > 0) {
        highest_used_bind_group = SDL_max(highest_used_bind_group, WEBGPU_FRAGMENT_RESOURCE_GROUP);
        needs_bind_group_layouts = true;
    }
    if (fragment_shader->resources.uniform_buffer_count > 0) {
        highest_used_bind_group = SDL_max(highest_used_bind_group, WEBGPU_FRAGMENT_UNIFORM_GROUP);
        needs_bind_group_layouts = true;
    }
    if (needs_bind_group_layouts) {
        bind_group_layout_count = highest_used_bind_group + 1;
        if (bind_group_layout_count > renderer->limits.maxBindGroups) {
            WEBGPU_SetStringError("bind group layout count exceeds WebGPU maxBindGroups");
            return NULL;
        }
        if (bind_group_layout_count + vertex_buffer_count > renderer->limits.maxBindGroupsPlusVertexBuffers) {
            WEBGPU_SetStringError("bind group and vertex buffer slot count exceeds WebGPU maxBindGroupsPlusVertexBuffers");
            return NULL;
        }

        for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
            const WebGPUShaderResourceLayout *shader_layout = NULL;
            bool resource_group = false;
            WGPUShaderStage visibility = WGPUShaderStage_None;

            if (i == WEBGPU_VERTEX_RESOURCE_GROUP) {
                shader_layout = &vertex_shader->resources;
                resource_group = true;
                visibility = WGPUShaderStage_Vertex;
            } else if (i == WEBGPU_VERTEX_UNIFORM_GROUP) {
                shader_layout = &vertex_shader->resources;
                visibility = WGPUShaderStage_Vertex;
            } else if (i == WEBGPU_FRAGMENT_RESOURCE_GROUP) {
                shader_layout = &fragment_shader->resources;
                resource_group = true;
                visibility = WGPUShaderStage_Fragment;
            } else if (i == WEBGPU_FRAGMENT_UNIFORM_GROUP) {
                shader_layout = &fragment_shader->resources;
                visibility = WGPUShaderStage_Fragment;
            }

            bind_group_layouts[i] = WEBGPU_CreateGraphicsBindGroupLayout(renderer, shader_layout, resource_group, visibility, &bind_group_layout_entry_counts[i]);
            if (!bind_group_layouts[i]) {
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                WEBGPU_SetStringError("CreateBindGroupLayout failed");
                return NULL;
            }
        }
    }

    layout_desc.bindGroupLayoutCount = bind_group_layout_count;
    layout_desc.bindGroupLayouts = bind_group_layout_count > 0 ? bind_group_layouts : NULL;

    layout = wgpuDeviceCreatePipelineLayout(renderer->device, &layout_desc);
    if (!layout) {
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        WEBGPU_SetStringError("CreatePipelineLayout failed");
        return NULL;
    }

    for (Uint32 i = 0; i < createinfo->target_info.num_color_targets; i += 1) {
        const SDL_GPUColorTargetDescription *sdl_color_target = &createinfo->target_info.color_target_descriptions[i];
        const SDL_GPUColorTargetBlendState *sdl_blend_state = &sdl_color_target->blend_state;

        color_targets[i] = (WGPUColorTargetState)WGPU_COLOR_TARGET_STATE_INIT;
        blend_states[i] = (WGPUBlendState)WGPU_BLEND_STATE_INIT;

        if (WEBGPU_IsDepthStencilFormat(sdl_color_target->format)) {
            WEBGPU_SetStringError("color target format must not be depth-stencil");
            wgpuPipelineLayoutRelease(layout);
            WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
            return NULL;
        }
        if (!WEBGPU_TextureFormatSupportsColorTarget(renderer, sdl_color_target->format)) {
            WEBGPU_SetStringError("unsupported color target format");
            wgpuPipelineLayoutRelease(layout);
            WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
            return NULL;
        }
        if (createinfo->multisample_state.sample_count != SDL_GPU_SAMPLECOUNT_1 &&
            !WEBGPU_TextureFormatSupportsMultisampleColorTarget(renderer, sdl_color_target->format)) {
            WEBGPU_SetStringError("unsupported multisample color target format");
            wgpuPipelineLayoutRelease(layout);
            WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
            return NULL;
        }
        color_targets[i].format = WEBGPU_ToWGPUTextureFormat(sdl_color_target->format);
        if (color_targets[i].format == WGPUTextureFormat_Undefined) {
            WEBGPU_SetStringError("unsupported color target format");
            wgpuPipelineLayoutRelease(layout);
            WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
            return NULL;
        }
        if (sdl_blend_state->enable_color_write_mask && !WEBGPU_IsValidColorWriteMask(sdl_blend_state->color_write_mask)) {
            WEBGPU_SetStringError("invalid color write mask");
            wgpuPipelineLayoutRelease(layout);
            WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
            return NULL;
        }
        color_targets[i].writeMask = sdl_blend_state->enable_color_write_mask ? WEBGPU_ToColorWriteMask(sdl_blend_state->color_write_mask) : WGPUColorWriteMask_All;
        if (sdl_blend_state->enable_blend) {
            if (!WEBGPU_TextureFormatSupportsBlend(renderer, sdl_color_target->format)) {
                WEBGPU_SetStringError("unsupported blend color target format");
                wgpuPipelineLayoutRelease(layout);
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                return NULL;
            }
            blend_states[i].color.operation = WEBGPU_ToBlendOperation(sdl_blend_state->color_blend_op);
            blend_states[i].color.srcFactor = WEBGPU_ToBlendFactor(sdl_blend_state->src_color_blendfactor);
            blend_states[i].color.dstFactor = WEBGPU_ToBlendFactor(sdl_blend_state->dst_color_blendfactor);
            blend_states[i].alpha.operation = WEBGPU_ToBlendOperation(sdl_blend_state->alpha_blend_op);
            blend_states[i].alpha.srcFactor = WEBGPU_ToBlendFactor(sdl_blend_state->src_alpha_blendfactor);
            blend_states[i].alpha.dstFactor = WEBGPU_ToBlendFactor(sdl_blend_state->dst_alpha_blendfactor);
            if (blend_states[i].color.operation == WGPUBlendOperation_Undefined) {
                WEBGPU_SetStringError("invalid color blend operation");
                wgpuPipelineLayoutRelease(layout);
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                return NULL;
            }
            if (blend_states[i].color.srcFactor == WGPUBlendFactor_Undefined ||
                blend_states[i].color.dstFactor == WGPUBlendFactor_Undefined) {
                WEBGPU_SetStringError("invalid color blend factor");
                wgpuPipelineLayoutRelease(layout);
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                return NULL;
            }
            if (blend_states[i].alpha.operation == WGPUBlendOperation_Undefined) {
                WEBGPU_SetStringError("invalid alpha blend operation");
                wgpuPipelineLayoutRelease(layout);
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                return NULL;
            }
            if (blend_states[i].alpha.srcFactor == WGPUBlendFactor_Undefined ||
                blend_states[i].alpha.dstFactor == WGPUBlendFactor_Undefined) {
                WEBGPU_SetStringError("invalid alpha blend factor");
                wgpuPipelineLayoutRelease(layout);
                WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
                return NULL;
            }
            WEBGPU_NormalizeBlendComponent(&blend_states[i].color);
            WEBGPU_NormalizeBlendComponent(&blend_states[i].alpha);
            color_targets[i].blend = &blend_states[i];
        }
    }
    fragment_state.module = fragment_shader->module;
    fragment_state.entryPoint = WEBGPU_StringView(fragment_shader->entrypoint);
    fragment_state.targetCount = createinfo->target_info.num_color_targets;
    fragment_state.targets = createinfo->target_info.num_color_targets > 0 ? color_targets : NULL;

    pipeline_desc.layout = layout;
    pipeline_desc.label = WEBGPU_StringView(debug_name);
    pipeline_desc.vertex.module = vertex_shader->module;
    pipeline_desc.vertex.entryPoint = WEBGPU_StringView(vertex_shader->entrypoint);
    pipeline_desc.vertex.bufferCount = vertex_buffer_count;
    pipeline_desc.vertex.buffers = vertex_buffer_count > 0 ? vertex_buffers : NULL;
    pipeline_desc.primitive.topology = primitive_topology;
    pipeline_desc.primitive.frontFace = front_face;
    pipeline_desc.primitive.cullMode = cull_mode;
    pipeline_desc.primitive.unclippedDepth = !createinfo->rasterizer_state.enable_depth_clip;
    pipeline_desc.multisample.count = sample_count;
    pipeline_desc.multisample.mask = 0xFFFFFFFFu;
    pipeline_desc.multisample.alphaToCoverageEnabled = createinfo->multisample_state.enable_alpha_to_coverage;
    pipeline_desc.fragment = &fragment_state;
    if (createinfo->target_info.has_depth_stencil_target) {
        pipeline_desc.depthStencil = &depth_stencil_state;
    }

    pipeline = (WebGPUGraphicsPipeline *)SDL_calloc(1, sizeof(*pipeline));
    if (!pipeline) {
        wgpuPipelineLayoutRelease(layout);
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        return NULL;
    }
    SDL_SetAtomicInt(&pipeline->refcount, 1);

    if (!WEBGPU_CanWaitForErrorScope(renderer, "CreateRenderPipeline")) {
        wgpuPipelineLayoutRelease(layout);
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        SDL_free(pipeline);
        return NULL;
    }

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    pipeline->pipeline = wgpuDeviceCreateRenderPipeline(renderer->device, &pipeline_desc);
    if (!WEBGPU_PopErrorScope(renderer, "CreateRenderPipeline")) {
        if (pipeline->pipeline) {
            wgpuRenderPipelineRelease(pipeline->pipeline);
        }
        wgpuPipelineLayoutRelease(layout);
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        SDL_free(pipeline);
        return NULL;
    }
    wgpuPipelineLayoutRelease(layout);
    if (!pipeline->pipeline) {
        WEBGPU_ReleaseBindGroupLayouts(bind_group_layouts, bind_group_layout_count);
        SDL_free(pipeline);
        WEBGPU_SetStringError("CreateRenderPipeline failed");
        return NULL;
    }

    for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
        pipeline->bind_group_layouts[i] = bind_group_layouts[i];
    }
    pipeline->bind_group_layout_count = bind_group_layout_count;
    if (!WEBGPU_CreatePipelineEmptyBindGroups(
            renderer,
            bind_group_layouts,
            bind_group_layout_entry_counts,
            bind_group_layout_count,
            pipeline->empty_bind_groups,
            WEBGPU_BIND_GROUP_PATH_GRAPHICS_EMPTY,
            "CreateBindGroup for empty graphics pipeline layout")) {
        WEBGPU_DestroyGraphicsPipeline(pipeline);
        return NULL;
    }
    SDL_memcpy(pipeline->vertex_buffer_strides, vertex_buffer_strides, sizeof(vertex_buffer_strides));
    SDL_memcpy(pipeline->vertex_buffer_last_strides, vertex_buffer_last_strides, sizeof(vertex_buffer_last_strides));
    SDL_memcpy(pipeline->vertex_buffer_wgpu_slots, vertex_buffer_wgpu_slots, sizeof(vertex_buffer_wgpu_slots));
    pipeline->required_vertex_buffer_mask = required_vertex_buffer_mask;
    SDL_memcpy(pipeline->vertex_buffer_instance_step, vertex_buffer_instance_step, sizeof(vertex_buffer_instance_step));
    pipeline->color_target_count = createinfo->target_info.num_color_targets;
    for (Uint32 i = 0; i < createinfo->target_info.num_color_targets; i += 1) {
        pipeline->color_target_formats[i] = createinfo->target_info.color_target_descriptions[i].format;
    }
    pipeline->sample_count = createinfo->multisample_state.sample_count;
    pipeline->has_depth_stencil_target = createinfo->target_info.has_depth_stencil_target;
    pipeline->depth_stencil_format = createinfo->target_info.depth_stencil_format;
    pipeline->vertex_resources = vertex_shader->resources;
    pipeline->fragment_resources = fragment_shader->resources;

    pipeline->header.num_vertex_samplers = vertex_shader->resources.sampler_count;
    pipeline->header.num_vertex_storage_textures = vertex_shader->resources.storage_texture_count;
    pipeline->header.num_vertex_storage_buffers = vertex_shader->resources.storage_buffer_count;
    pipeline->header.num_vertex_uniform_buffers = vertex_shader->resources.uniform_buffer_count;
    pipeline->header.num_fragment_samplers = fragment_shader->resources.sampler_count;
    pipeline->header.num_fragment_storage_textures = fragment_shader->resources.storage_texture_count;
    pipeline->header.num_fragment_storage_buffers = fragment_shader->resources.storage_buffer_count;
    pipeline->header.num_fragment_uniform_buffers = fragment_shader->resources.uniform_buffer_count;

    return (SDL_GPUGraphicsPipeline *)pipeline;
}

static SDL_GPUSampler *WEBGPU_CreateSampler(
    SDL_GPURenderer *driverData,
    const SDL_GPUSamplerCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    float min_lod = createinfo->min_lod;
    float max_lod = createinfo->max_lod;
    WebGPUSampler *sampler;

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateSampler")) {
        return NULL;
    }
    if (createinfo->mip_lod_bias != 0.0f) {
        WEBGPU_SetStringError("sampler mip LOD bias is not supported by WebGPU sampler state");
        return NULL;
    }

    if (SDL_isnanf(min_lod) || SDL_isinff(min_lod) ||
        SDL_isnanf(max_lod) || SDL_isinff(max_lod)) {
        WEBGPU_SetStringError("invalid sampler LOD clamp");
        return NULL;
    }

    if (min_lod < 0.0f) {
        min_lod = 0.0f;
    }
    if (max_lod < 0.0f) {
        max_lod = 0.0f;
    }
    if (max_lod < min_lod) {
        WEBGPU_SetStringError("invalid sampler LOD clamp");
        return NULL;
    }

    sampler_desc.label = WEBGPU_StringView(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_SAMPLER_CREATE_NAME_STRING, NULL));
    sampler_desc.addressModeU = WEBGPU_ToAddressMode(createinfo->address_mode_u);
    sampler_desc.addressModeV = WEBGPU_ToAddressMode(createinfo->address_mode_v);
    sampler_desc.addressModeW = WEBGPU_ToAddressMode(createinfo->address_mode_w);
    sampler_desc.magFilter = WEBGPU_ToFilterMode(createinfo->mag_filter);
    sampler_desc.minFilter = WEBGPU_ToFilterMode(createinfo->min_filter);
    sampler_desc.mipmapFilter = WEBGPU_ToMipmapFilterMode(createinfo->mipmap_mode);
    if (sampler_desc.addressModeU == WGPUAddressMode_Undefined ||
        sampler_desc.addressModeV == WGPUAddressMode_Undefined ||
        sampler_desc.addressModeW == WGPUAddressMode_Undefined) {
        WEBGPU_SetStringError("invalid sampler address mode");
        return NULL;
    }
    if (sampler_desc.magFilter == WGPUFilterMode_Undefined ||
        sampler_desc.minFilter == WGPUFilterMode_Undefined) {
        WEBGPU_SetStringError("invalid sampler filter");
        return NULL;
    }
    if (sampler_desc.mipmapFilter == WGPUMipmapFilterMode_Undefined) {
        WEBGPU_SetStringError("invalid sampler mipmap mode");
        return NULL;
    }
    sampler_desc.lodMinClamp = min_lod;
    sampler_desc.lodMaxClamp = max_lod;
    sampler_desc.maxAnisotropy = 1;
    if (createinfo->enable_anisotropy) {
        if (SDL_isnanf(createinfo->max_anisotropy) ||
            SDL_isinff(createinfo->max_anisotropy) ||
            createinfo->max_anisotropy < 1.0f) {
            WEBGPU_SetStringError("invalid sampler max anisotropy");
            return NULL;
        }

        if (createinfo->max_anisotropy > 1.0f &&
            (sampler_desc.magFilter != WGPUFilterMode_Linear ||
             sampler_desc.minFilter != WGPUFilterMode_Linear ||
             sampler_desc.mipmapFilter != WGPUMipmapFilterMode_Linear)) {
            WEBGPU_SetStringError("anisotropic filtering requires linear sampler filters");
            return NULL;
        }

        sampler_desc.maxAnisotropy = (uint16_t)SDL_clamp(createinfo->max_anisotropy, 1.0f, 16.0f);
    }
    if (createinfo->enable_compare) {
        sampler_desc.compare = WEBGPU_ToCompareFunction(createinfo->compare_op);
        if (sampler_desc.compare == WGPUCompareFunction_Undefined) {
            WEBGPU_SetStringError("invalid sampler compare operation");
            return NULL;
        }
    }

    sampler = (WebGPUSampler *)SDL_calloc(1, sizeof(*sampler));
    if (!sampler) {
        return NULL;
    }
    if (createinfo->enable_compare) {
        sampler->binding_type = WGPUSamplerBindingType_Comparison;
    } else if (WEBGPU_IsNonFilteringSamplerDescriptor(&sampler_desc)) {
        sampler->binding_type = WGPUSamplerBindingType_NonFiltering;
    } else {
        sampler->binding_type = WGPUSamplerBindingType_Filtering;
    }

    sampler->sampler = wgpuDeviceCreateSampler(renderer->device, &sampler_desc);
    if (!sampler->sampler) {
        SDL_free(sampler);
        WEBGPU_SetStringError("CreateSampler failed");
        return NULL;
    }

    return (SDL_GPUSampler *)sampler;
}

static bool WEBGPU_IsValidShaderStage(SDL_GPUShaderStage stage)
{
    return stage == SDL_GPU_SHADERSTAGE_VERTEX ||
           stage == SDL_GPU_SHADERSTAGE_FRAGMENT;
}

static SDL_GPUShader *WEBGPU_CreateShader(
    SDL_GPURenderer *driverData,
    const SDL_GPUShaderCreateInfo *createinfo,
    const SDL_GPUShaderResourceLayoutFacts *layout_facts)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    WGPUShaderModuleDescriptor shader_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    WebGPUShader *shader;
    const char *entrypoint = createinfo->entrypoint ? createinfo->entrypoint : "main";

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateShader")) {
        return NULL;
    }
    if (createinfo->format != SDL_GPU_SHADERFORMAT_WGSL) {
        WEBGPU_SetStringError("only WGSL shaders are supported");
        return NULL;
    }
    if (!WEBGPU_IsValidShaderStage(createinfo->stage)) {
        WEBGPU_SetStringError("invalid shader stage");
        return NULL;
    }
    if (layout_facts &&
        layout_facts->stage != createinfo->stage) {
        WEBGPU_SetStringError("shader resource layout stage does not match shader stage");
        return NULL;
    }
    if (createinfo->num_samplers > MAX_TEXTURE_SAMPLERS_PER_STAGE) {
        WEBGPU_SetStringError("shader sampler count exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE");
        return NULL;
    }
    if (createinfo->num_storage_textures > MAX_STORAGE_TEXTURES_PER_STAGE) {
        WEBGPU_SetStringError("shader storage texture count exceeds MAX_STORAGE_TEXTURES_PER_STAGE");
        return NULL;
    }
    if (createinfo->num_storage_buffers > MAX_STORAGE_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("shader storage buffer count exceeds MAX_STORAGE_BUFFERS_PER_STAGE");
        return NULL;
    }
    if (createinfo->num_uniform_buffers > MAX_UNIFORM_BUFFERS_PER_STAGE) {
        WEBGPU_SetStringError("shader uniform buffer count exceeds MAX_UNIFORM_BUFFERS_PER_STAGE");
        return NULL;
    }
    if (createinfo->num_storage_textures > 0 &&
        !renderer->supports_readonly_and_readwrite_storage_textures) {
        WEBGPU_SetStringError("WebGPU read-only storage textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        return NULL;
    }

    wgsl.code = WEBGPU_ByteView(createinfo->code, createinfo->code_size);
    shader_desc.label = WEBGPU_StringView(SDL_GetStringProperty(createinfo->props, SDL_PROP_GPU_SHADER_CREATE_NAME_STRING, NULL));
    shader_desc.nextInChain = &wgsl.chain;

    shader = (WebGPUShader *)SDL_calloc(1, sizeof(*shader));
    if (!shader) {
        return NULL;
    }

    if (!WEBGPU_CanWaitForErrorScope(renderer, "CreateShaderModule")) {
        SDL_free(shader);
        return NULL;
    }

    wgpuDevicePushErrorScope(renderer->device, WGPUErrorFilter_Validation);
    shader->module = wgpuDeviceCreateShaderModule(renderer->device, &shader_desc);
    if (!WEBGPU_PopErrorScope(renderer, "CreateShaderModule")) {
        if (shader->module) {
            wgpuShaderModuleRelease(shader->module);
        }
        SDL_free(shader);
        return NULL;
    }
    if (!shader->module) {
        SDL_free(shader);
        WEBGPU_SetStringError("CreateShaderModule failed");
        return NULL;
    }

    shader->entrypoint = SDL_strdup(entrypoint);
    if (!shader->entrypoint) {
        WEBGPU_DestroyShader(shader);
        return NULL;
    }
    shader->stage = createinfo->stage;
    if (!WEBGPU_InitDefaultShaderResourceLayout(renderer, &shader->resources, createinfo, layout_facts)) {
        WEBGPU_DestroyShader(shader);
        return NULL;
    }
    return (SDL_GPUShader *)shader;
}

static bool WEBGPU_FailStorageTexturePolicy(bool set_error, const char *message)
{
    if (set_error) {
        WEBGPU_SetStringError(message);
    }
    return false;
}

static bool WEBGPU_IsReadOnlyStorageTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
           type == SDL_GPU_TEXTURETYPE_3D;
}

static bool WEBGPU_IsComputeReadWriteStorageTextureType(SDL_GPUTextureType type)
{
    return type == SDL_GPU_TEXTURETYPE_2D ||
           type == SDL_GPU_TEXTURETYPE_2D_ARRAY ||
           type == SDL_GPU_TEXTURETYPE_3D;
}

static bool WEBGPU_ValidateStorageTexturePolicy(
    WebGPURenderer *renderer,
    SDL_GPUTextureFormat format,
    SDL_GPUTextureType type,
    SDL_GPUTextureUsageFlags usage,
    bool set_error)
{
    if (usage & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) {
        if (usage & ~(SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "graphics storage read textures only support optional color target usage");
        }
        if (!renderer->supports_readonly_and_readwrite_storage_textures) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU graphics storage read textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        }
        if (!WEBGPU_IsReadOnlyStorageTextureType(type)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "graphics storage read textures only support 2D, 2D-array, and 3D textures");
        }
        if (!SDL_GPUStorageTextureFormatSupportsReadOnly(format)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "graphics storage read textures only support SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT, SDL_GPU_TEXTUREFORMAT_R32_INT, or SDL_GPU_TEXTUREFORMAT_R32_FLOAT");
        }
        if (WEBGPU_StorageTextureReadOnlyFormatRequiresCoreFeaturesAndLimits(format) &&
            !renderer->supports_core_features_and_limits) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU graphics storage read textures with this format require core-features-and-limits");
        }
    }

    if (usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ) {
        if (usage != SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage read textures only support standalone usage");
        }
        if (!renderer->supports_readonly_and_readwrite_storage_textures) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU compute storage read textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        }
        if (!WEBGPU_IsReadOnlyStorageTextureType(type)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage read textures only support 2D, 2D-array, and 3D textures");
        }
        if (!SDL_GPUStorageTextureFormatSupportsComputeReadOnly(format)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage read textures only support SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT, SDL_GPU_TEXTUREFORMAT_R32_INT, or SDL_GPU_TEXTUREFORMAT_R32_FLOAT");
        }
        if (WEBGPU_StorageTextureReadOnlyFormatRequiresCoreFeaturesAndLimits(format) &&
            !renderer->supports_core_features_and_limits) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU compute storage read textures with this format require core-features-and-limits");
        }
    }

    if (usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) {
        if (usage & ~(SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage write textures only support optional sampler usage");
        }
        if (!WEBGPU_IsComputeReadWriteStorageTextureType(type)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage write textures only support 2D, 2D-array, and 3D textures");
        }
        if (!SDL_GPUStorageTextureFormatSupportsWriteOnly(format)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage write textures only support SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_SNORM, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT, SDL_GPU_TEXTUREFORMAT_R32_INT, or SDL_GPU_TEXTUREFORMAT_R32_FLOAT");
        }
    }

    if (usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
        if (usage != SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage simultaneous read/write textures only support standalone COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE usage");
        }
        if (!renderer->supports_readonly_and_readwrite_storage_textures) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU compute simultaneous read/write storage textures require the readonly_and_readwrite_storage_textures WGSL language feature");
        }
        if (WEBGPU_StorageTextureReadWriteFormatRequiresTier2(format) &&
            !renderer->supports_texture_formats_tier2) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "WebGPU compute simultaneous read/write storage textures with this format require the texture-formats-tier2 feature");
        }
        if (!WEBGPU_IsComputeReadWriteStorageTextureType(type)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage simultaneous read/write textures only support 2D, 2D-array, and 3D textures");
        }
        if (!SDL_GPUStorageTextureFormatSupportsReadWrite(format)) {
            return WEBGPU_FailStorageTexturePolicy(set_error, "compute storage simultaneous read/write textures only support SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UINT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_UINT, SDL_GPU_TEXTUREFORMAT_R32_UINT, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_INT, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_INT, SDL_GPU_TEXTUREFORMAT_R32_INT, or SDL_GPU_TEXTUREFORMAT_R32_FLOAT");
        }
    }

    return true;
}

static bool WEBGPU_ValidateStorageTextureCreateInfo(
    WebGPURenderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    if (!WEBGPU_ValidateStorageTexturePolicy(
            renderer,
            createinfo->format,
            createinfo->type,
            createinfo->usage,
            true)) {
        return false;
    }

    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ) {
        if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("graphics storage read textures only support sample count 1");
            return false;
        }
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ) {
        if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("compute storage read textures only support sample count 1");
            return false;
        }
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) {
        if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("compute storage write textures only support sample count 1");
            return false;
        }
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
        if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("compute storage simultaneous read/write textures only support sample count 1");
            return false;
        }
    }
    return true;
}

static bool WEBGPU_ValidateD24TextureCreateInfo(
    const SDL_GPUTextureCreateInfo *createinfo)
{
    if (!IsD24Format(createinfo->format)) {
        return true;
    }
    if (IsD24AcceptedTextureCreateInfo(createinfo)) {
        return true;
    }
    if (createinfo->type != SDL_GPU_TEXTURETYPE_2D &&
        createinfo->type != SDL_GPU_TEXTURETYPE_2D_ARRAY) {
        WEBGPU_SetStringError("WebGPU D24 depth formats only support 2D or 2D array target textures");
        return false;
    }
    if (createinfo->usage != SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET &&
        createinfo->usage != (SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
        WEBGPU_SetStringError("WebGPU D24 depth formats only support depth-stencil target usage, optionally combined with sampler usage");
        return false;
    }
    if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_SetStringError("WebGPU D24 depth formats only support sample count 1");
        return false;
    }
    WEBGPU_SetStringError("WebGPU D24 depth formats only support target-only or target-plus-sampler textures");
    return false;
}

static bool WEBGPU_ValidateTextureCreateInfoStatic(
    WebGPURenderer *renderer,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    WGPUTextureFormat format;
    Uint32 sample_count;

    if (!WEBGPU_IsTransferTextureType(createinfo->type)) {
        WEBGPU_SetStringError("unsupported texture type");
        return false;
    }
    if (createinfo->width == 0 || createinfo->height == 0) {
        WEBGPU_SetStringError("texture width and height must be at least 1");
        return false;
    }
    if (!WEBGPU_Is2DTextureType(createinfo->type) &&
        createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_SetStringError("3D, cube, and cube array textures only support sample count 1");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_3D &&
        (createinfo->usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE))) {
        WEBGPU_SetStringError("3D textures only support sampler, color target, graphics storage read, compute storage read, compute storage write, compute simultaneous read/write storage, or transfer-only usage");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE ||
        createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
        if (WEBGPU_IsDepthStencilFormat(createinfo->format)) {
            if (!WEBGPU_TextureFormatIsAcceptedDepthCube(createinfo->format)) {
                WEBGPU_SetStringError("cube and cube array depth textures only support D16_UNORM or D32_FLOAT");
                return false;
            }
            if (createinfo->usage == 0) {
                WEBGPU_SetStringError("cube and cube array depth textures require sampler or depth-stencil target usage");
                return false;
            }
            if (createinfo->usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
                WEBGPU_SetStringError("cube and cube array depth textures only support sampler or depth-stencil target usage");
                return false;
            }
        } else if (createinfo->usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
            WEBGPU_SetStringError("cube and cube array textures only support sampler, color target, or transfer-only usage");
            return false;
        }
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_3D &&
        WEBGPU_IsDepthStencilFormat(createinfo->format)) {
        WEBGPU_SetStringError("3D textures only support color formats");
        return false;
    }
    if ((createinfo->type == SDL_GPU_TEXTURETYPE_CUBE ||
         createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) &&
        createinfo->width != createinfo->height) {
        WEBGPU_SetStringError("cube and cube array textures require matching width and height");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE &&
        createinfo->layer_count_or_depth != 6) {
        WEBGPU_SetStringError("cube textures require exactly 6 layers");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY &&
        createinfo->layer_count_or_depth % 6 != 0) {
        WEBGPU_SetStringError("cube array textures require a layer count that is a multiple of 6");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_2D &&
        createinfo->layer_count_or_depth != 1) {
        WEBGPU_SetStringError("2D textures must have exactly one layer");
        return false;
    }
    if (createinfo->type == SDL_GPU_TEXTURETYPE_2D_ARRAY) {
        if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
            WEBGPU_SetStringError("multisample texture arrays are not supported");
            return false;
        }
    }
    if (createinfo->layer_count_or_depth == 0) {
        WEBGPU_SetStringError("texture must have at least one layer");
        return false;
    }
    if (createinfo->num_levels == 0) {
        WEBGPU_SetStringError("texture must have at least one mip level");
        return false;
    }
    if (!WEBGPU_ValidateTextureDimensions(renderer, createinfo)) {
        return false;
    }
    if (createinfo->num_levels > WEBGPU_TextureMaxMipLevels(createinfo->width, createinfo->height, createinfo->type == SDL_GPU_TEXTURETYPE_3D ? createinfo->layer_count_or_depth : 1)) {
        WEBGPU_SetStringError("texture mip level count exceeds texture dimensions");
        return false;
    }

    format = WEBGPU_ToWGPUTextureFormat(createinfo->format);
    if (format == WGPUTextureFormat_Undefined) {
        WEBGPU_SetStringError("unsupported texture format");
        return false;
    }
    if (!WEBGPU_TextureFormatSupportedByDevice(renderer, createinfo->format)) {
        WEBGPU_SetStringError("unsupported texture format");
        return false;
    }
    if (!WEBGPU_TextureFormatSupportedForType(renderer, createinfo->format, createinfo->type)) {
        WEBGPU_SetStringError("unsupported texture format for texture type");
        return false;
    }
    if (!WEBGPU_ValidateD24TextureCreateInfo(createinfo)) {
        return false;
    }
    if ((createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
        !WEBGPU_IsDepthStencilFormat(createinfo->format)) {
        if (!WEBGPU_TextureFormatSupportsColorTarget(renderer, createinfo->format)) {
            WEBGPU_SetStringError("unsupported color target format");
            return false;
        }
    }
    if (createinfo->format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT_S8_UINT &&
        !renderer->supports_depth32float_stencil8) {
        WEBGPU_SetStringError("D32_FLOAT_S8_UINT requires WebGPU Depth32FloatStencil8 feature");
        return false;
    }
    sample_count = WEBGPU_ToSampleCount(createinfo->sample_count);
    if (sample_count == 0) {
        WEBGPU_SetStringError("unsupported sample count");
        return false;
    }
    if (createinfo->usage & ~(SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        WEBGPU_SetStringError("only sampler, color target, depth-stencil target, graphics storage read, and compute storage texture usages are supported");
        return false;
    }
    if (!WEBGPU_ValidateStorageTextureCreateInfo(renderer, createinfo)) {
        return false;
    }
    if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        if (!WEBGPU_IsDepthStencilFormat(createinfo->format)) {
            WEBGPU_SetStringError("depth-stencil target usage requires a depth-stencil texture format");
            return false;
        }
        if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
            WEBGPU_SetStringError("depth-stencil textures do not support color target usage");
            return false;
        }
    } else if (WEBGPU_IsDepthStencilFormat(createinfo->format)) {
        if (!(createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
            WEBGPU_SetStringError("depth-stencil texture formats require depth-stencil target usage or sampler usage");
            return false;
        }
        if (createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
            WEBGPU_SetStringError("depth-stencil textures do not support color target usage");
            return false;
        }
    }
    if ((createinfo->usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) &&
        !WEBGPU_TextureFormatSupportsTextureBindingForType(createinfo->format, createinfo->type)) {
        WEBGPU_SetStringError("unsupported sampled texture format");
        return false;
    }
    if (createinfo->sample_count != SDL_GPU_SAMPLECOUNT_1) {
        const bool accepted_sampled_msaa = SDL_GPUTextureCreateInfoIsAcceptedMultisampledSampledTexture(createinfo);
        if (createinfo->usage != SDL_GPU_TEXTUREUSAGE_COLOR_TARGET &&
            createinfo->usage != SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET &&
            !accepted_sampled_msaa) {
            WEBGPU_SetStringError("multisample textures only support color target, depth-stencil target, or accepted multisampled sampled target usage");
            return false;
        }
        if ((createinfo->usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
            !WEBGPU_TextureFormatSupportsMultisampleColorTarget(renderer, createinfo->format)) {
            WEBGPU_SetStringError("unsupported multisample color target format");
            return false;
        }
        if (createinfo->num_levels != 1) {
            WEBGPU_SetStringError("multisample textures must have one mip level");
            return false;
        }
    }

    return true;
}

static SDL_GPUTexture *WEBGPU_CreateTexture(
    SDL_GPURenderer *driverData,
    const SDL_GPUTextureCreateInfo *createinfo)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUTexture *texture;

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateTexture")) {
        return NULL;
    }
    if (!WEBGPU_ValidateTextureCreateInfoStatic(
            renderer,
            createinfo)) {
        return NULL;
    }

    texture = (WebGPUTexture *)SDL_calloc(1, sizeof(*texture));
    if (!texture) {
        return NULL;
    }

    if (!WEBGPU_CreateTextureHandles(
            renderer,
            createinfo,
            WEBGPU_RESOURCE_GENERATION_INITIAL,
            &texture->texture,
            &texture->view,
            &texture->storage_view)) {
        SDL_free(texture);
        return NULL;
    }

    texture->generation = WEBGPU_RESOURCE_GENERATION_INITIAL;
    if (!SDL_GPUTextureHeaderInit(
            &texture->header,
            createinfo)) {
        WEBGPU_ReleaseTextureHandles(texture);
        SDL_free(texture);
        return NULL;
    }
    texture->refcount = 1;
    return (SDL_GPUTexture *)texture;
}










static SDL_GPUBuffer *WEBGPU_CreateBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUBufferUsageFlags usageFlags,
    Uint32 size,
    const char *debugName)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUBuffer *buffer;

    if (!WEBGPU_FailIfDeviceLost(renderer, "CreateBuffer")) {
        return NULL;
    }
    if (size == 0) {
        SDL_SetError("Cannot create a zero-sized WebGPU buffer");
        return NULL;
    }
    if (!WEBGPU_IsValidBufferUsage(usageFlags)) {
        if (usageFlags == 0) {
            SDL_SetError("WebGPU buffer usage must include at least one SDL_GPU_BUFFERUSAGE flag");
        } else {
            SDL_SetError("invalid WebGPU buffer usage flags");
        }
        return NULL;
    }
    buffer = (WebGPUBuffer *)SDL_calloc(1, sizeof(*buffer));
    if (!buffer) {
        return NULL;
    }

    if (!WEBGPU_CreateBufferHandle(renderer, usageFlags, size, debugName, &buffer->buffer, &buffer->allocation_size)) {
        SDL_free(buffer);
        return NULL;
    }

    SDL_GPUBufferHeaderInit(&buffer->header, usageFlags, size);
    buffer->usage = usageFlags;
    buffer->generation = WEBGPU_RESOURCE_GENERATION_INITIAL;
    buffer->size = size;
    buffer->refcount = 1;
    return (SDL_GPUBuffer *)buffer;
}

static SDL_GPUTransferBuffer *WEBGPU_CreateTransferBuffer(
    SDL_GPURenderer *driverData,
    SDL_GPUTransferBufferUsage usage,
    Uint32 size,
    const char *debugName)
{
    (void)driverData;
    WebGPUTransferBuffer *transfer_buffer;
    WebGPUTransferBufferGeneration *generation;

    if (size == 0) {
        SDL_SetError("Cannot create a zero-sized WebGPU transfer buffer");
        return NULL;
    }
    if (usage != SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD &&
        usage != SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD) {
        SDL_SetError("invalid WebGPU transfer buffer usage");
        return NULL;
    }

    transfer_buffer = (WebGPUTransferBuffer *)SDL_calloc(1, sizeof(*transfer_buffer));
    if (!transfer_buffer) {
        return NULL;
    }

    generation = WEBGPU_CreateTransferBufferGeneration(size);
    if (!generation) {
        SDL_free(transfer_buffer);
        return NULL;
    }

    if (!WEBGPU_AddTransferBufferGeneration(transfer_buffer, generation)) {
        WEBGPU_DestroyTransferBufferGeneration(generation);
        SDL_free(transfer_buffer);
        return NULL;
    }

    if (debugName) {
        transfer_buffer->debugName = SDL_strdup(debugName);
        if (!transfer_buffer->debugName) {
            WEBGPU_DestroyTransferBuffer(transfer_buffer);
            return NULL;
        }
    }

    transfer_buffer->usage = usage;
    transfer_buffer->size = size;
    return (SDL_GPUTransferBuffer *)transfer_buffer;
}

static XrResult WEBGPU_CreateXRSession(
    SDL_GPURenderer *driverData,
    const XrSessionCreateInfo *createinfo,
    XrSession *session)
{
    (void)driverData;
    (void)createinfo;
    (void)session;
    return WEBGPU_UnsupportedXR("CreateXRSession");
}

static SDL_GPUTextureFormat *WEBGPU_GetXRSwapchainFormats(
    SDL_GPURenderer *driverData,
    XrSession session,
    int *num_formats)
{
    (void)driverData;
    (void)session;
    if (num_formats) {
        *num_formats = 0;
    }
    return (SDL_GPUTextureFormat *)WEBGPU_UnsupportedPointer("GetXRSwapchainFormats");
}

static XrResult WEBGPU_CreateXRSwapchain(
    SDL_GPURenderer *driverData,
    XrSession session,
    const XrSwapchainCreateInfo *createinfo,
    SDL_GPUTextureFormat format,
    XrSwapchain *swapchain,
    SDL_GPUTexture ***textures)
{
    (void)driverData;
    (void)session;
    (void)createinfo;
    (void)format;
    (void)swapchain;
    (void)textures;
    return WEBGPU_UnsupportedXR("CreateXRSwapchain");
}

static void WEBGPU_SetBufferName(SDL_GPURenderer *driverData, SDL_GPUBuffer *buffer, const char *text)
{
    WebGPUBuffer *webgpu_buffer = (WebGPUBuffer *)buffer;
    (void)driverData;

    if (webgpu_buffer && webgpu_buffer->released) {
        SDL_SetError("buffer has been released");
        return;
    }

    if (webgpu_buffer && webgpu_buffer->buffer && !webgpu_buffer->released) {
        wgpuBufferSetLabel(webgpu_buffer->buffer, WEBGPU_StringView(text));
    }
}

static void WEBGPU_SetTextureName(SDL_GPURenderer *driverData, SDL_GPUTexture *texture, const char *text)
{
    WebGPUTexture *webgpu_texture = (WebGPUTexture *)texture;
    (void)driverData;

    if (!webgpu_texture) {
        return;
    }
    if (webgpu_texture->released) {
        SDL_SetError("cannot name a released WebGPU texture");
        return;
    }
    if (webgpu_texture->texture) {
        wgpuTextureSetLabel(webgpu_texture->texture, WEBGPU_StringView(text));
    }
    if (webgpu_texture->view) {
        wgpuTextureViewSetLabel(webgpu_texture->view, WEBGPU_StringView(text));
    }
    if (webgpu_texture->storage_view) {
        wgpuTextureViewSetLabel(webgpu_texture->storage_view, WEBGPU_StringView(text));
    }
}

static void WEBGPU_InsertDebugLabel(SDL_GPUCommandBuffer *commandBuffer, const char *text)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WGPUStringView label = WEBGPU_StringView(text);

    if (!command_buffer) {
        return;
    }
    if (command_buffer->render_pass) {
        wgpuRenderPassEncoderInsertDebugMarker(command_buffer->render_pass, label);
    } else if (command_buffer->compute_pass) {
        wgpuComputePassEncoderInsertDebugMarker(command_buffer->compute_pass, label);
    } else if (command_buffer->encoder) {
        wgpuCommandEncoderInsertDebugMarker(command_buffer->encoder, label);
    }
}

static void WEBGPU_PushDebugGroup(SDL_GPUCommandBuffer *commandBuffer, const char *name)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WGPUStringView label = WEBGPU_StringView(name);

    if (!command_buffer) {
        return;
    }
    if (command_buffer->render_pass) {
        wgpuRenderPassEncoderPushDebugGroup(command_buffer->render_pass, label);
    } else if (command_buffer->compute_pass) {
        wgpuComputePassEncoderPushDebugGroup(command_buffer->compute_pass, label);
    } else if (command_buffer->encoder) {
        wgpuCommandEncoderPushDebugGroup(command_buffer->encoder, label);
    }
}

static void WEBGPU_PopDebugGroup(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (!command_buffer) {
        return;
    }
    if (command_buffer->render_pass) {
        wgpuRenderPassEncoderPopDebugGroup(command_buffer->render_pass);
    } else if (command_buffer->compute_pass) {
        wgpuComputePassEncoderPopDebugGroup(command_buffer->compute_pass);
    } else if (command_buffer->encoder) {
        wgpuCommandEncoderPopDebugGroup(command_buffer->encoder);
    }
}

static void WEBGPU_ReleaseTexture(SDL_GPURenderer *driverData, SDL_GPUTexture *texture)
{
    WebGPUTexture *webgpu_texture = (WebGPUTexture *)texture;
    (void)driverData;

    if (!webgpu_texture) {
        return;
    }
    if (webgpu_texture->from_surface) {
        SDL_SetError("Swapchain textures are managed by the command buffer and must not be released by the user");
        return;
    }
    if (webgpu_texture->released) {
        return;
    }
    webgpu_texture->released = true;
    WEBGPU_ReleaseTextureReference(webgpu_texture);
}

static void WEBGPU_ReleaseSampler(SDL_GPURenderer *driverData, SDL_GPUSampler *sampler)
{
    WebGPUSampler *webgpu_sampler = (WebGPUSampler *)sampler;
    (void)driverData;

    if (!webgpu_sampler) {
        return;
    }
    if (webgpu_sampler->sampler) {
        wgpuSamplerRelease(webgpu_sampler->sampler);
    }
    SDL_free(webgpu_sampler);
}

static void WEBGPU_DestroyBlitResources(WebGPURenderer *renderer)
{
    if (!renderer) {
        return;
    }

    for (Uint32 i = 0; i < renderer->blit_pipeline_count; i += 1) {
        WEBGPU_ReleaseGraphicsPipelineReference((WebGPUGraphicsPipeline *)renderer->blit_pipelines[i].pipeline);
        renderer->blit_pipelines[i].pipeline = NULL;
    }
    SDL_free(renderer->blit_pipelines);
    renderer->blit_pipelines = NULL;
    renderer->blit_pipeline_count = 0;
    renderer->blit_pipeline_capacity = 0;

    WEBGPU_ReleaseSampler((SDL_GPURenderer *)renderer, renderer->blit_linear_sampler);
    renderer->blit_linear_sampler = NULL;
    WEBGPU_ReleaseSampler((SDL_GPURenderer *)renderer, renderer->blit_nearest_sampler);
    renderer->blit_nearest_sampler = NULL;

    WEBGPU_DestroyShader((WebGPUShader *)renderer->blit_vertex_shader);
    renderer->blit_vertex_shader = NULL;
    WEBGPU_DestroyShader((WebGPUShader *)renderer->blit_from_2d_shader);
    renderer->blit_from_2d_shader = NULL;
    WEBGPU_DestroyShader((WebGPUShader *)renderer->blit_from_3d_shader);
    renderer->blit_from_3d_shader = NULL;
}

static bool WEBGPU_CreateBlitResources(SDL_GPUDevice *device)
{
    WebGPURenderer *renderer = (WebGPURenderer *)device->driverData;
    SDL_GPUShaderCreateInfo shader_info;
    SDL_GPUShaderWithResourceLayoutCreateInfo shader_with_layout_info;
    SDL_GPUShaderResourceLayout layout_info;
    SDL_GPUSamplerCreateInfo sampler_info;
    SDL_GPUSampledTextureSlotDescription blit_from_3d_texture_sampler;

    SDL_zero(shader_info);
    shader_info.code = (const Uint8 *)WEBGPU_BlitVertexShaderSource;
    shader_info.code_size = sizeof(WEBGPU_BlitVertexShaderSource) - 1;
    shader_info.entrypoint = "main";
    shader_info.format = SDL_GPU_SHADERFORMAT_WGSL;
    shader_info.stage = SDL_GPU_SHADERSTAGE_VERTEX;
    renderer->blit_vertex_shader = SDL_CreateGPUShader(device, &shader_info);
    if (!renderer->blit_vertex_shader) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    SDL_zero(shader_info);
    shader_info.code = (const Uint8 *)WEBGPU_BlitFrom2DShaderSource;
    shader_info.code_size = sizeof(WEBGPU_BlitFrom2DShaderSource) - 1;
    shader_info.entrypoint = "main";
    shader_info.format = SDL_GPU_SHADERFORMAT_WGSL;
    shader_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    shader_info.num_samplers = 1;
    shader_info.num_uniform_buffers = 1;
    renderer->blit_from_2d_shader = SDL_CreateGPUShader(device, &shader_info);
    if (!renderer->blit_from_2d_shader) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    SDL_zero(shader_info);
    shader_info.code = (const Uint8 *)WEBGPU_BlitFrom3DShaderSource;
    shader_info.code_size = sizeof(WEBGPU_BlitFrom3DShaderSource) - 1;
    shader_info.entrypoint = "main";
    shader_info.format = SDL_GPU_SHADERFORMAT_WGSL;
    shader_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    shader_info.num_samplers = 1;
    shader_info.num_uniform_buffers = 1;

    SDL_zero(blit_from_3d_texture_sampler);
    blit_from_3d_texture_sampler.texture_type = SDL_GPU_TEXTURETYPE_3D;
    blit_from_3d_texture_sampler.sample_type = SDL_GPU_SHADERTEXTURESAMPLETYPE_FILTERABLE_FLOAT;
    blit_from_3d_texture_sampler.sampler_type = SDL_GPU_SHADERSAMPLERTYPE_FILTERING;

    SDL_zero(layout_info);
    layout_info.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    layout_info.num_samplers = shader_info.num_samplers;
    layout_info.num_storage_textures = shader_info.num_storage_textures;
    layout_info.num_storage_buffers = shader_info.num_storage_buffers;
    layout_info.num_uniform_buffers = shader_info.num_uniform_buffers;
    layout_info.sampled_texture_slots = &blit_from_3d_texture_sampler;

    SDL_zero(shader_with_layout_info);
    shader_with_layout_info.code = shader_info.code;
    shader_with_layout_info.code_size = shader_info.code_size;
    shader_with_layout_info.entrypoint = shader_info.entrypoint;
    shader_with_layout_info.format = shader_info.format;
    shader_with_layout_info.resource_layout = &layout_info;
    renderer->blit_from_3d_shader = SDL_CreateGPUShaderWithResourceLayout(device, &shader_with_layout_info);
    if (!renderer->blit_from_3d_shader) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    SDL_zero(sampler_info);
    sampler_info.min_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    sampler_info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
    sampler_info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    sampler_info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
    renderer->blit_linear_sampler = SDL_CreateGPUSampler(device, &sampler_info);
    if (!renderer->blit_linear_sampler) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    sampler_info.min_filter = SDL_GPU_FILTER_NEAREST;
    sampler_info.mag_filter = SDL_GPU_FILTER_NEAREST;
    renderer->blit_nearest_sampler = SDL_CreateGPUSampler(device, &sampler_info);
    if (!renderer->blit_nearest_sampler) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    renderer->blit_pipeline_capacity = 2;
    renderer->blit_pipelines = (BlitPipelineCacheEntry *)SDL_calloc(
        renderer->blit_pipeline_capacity,
        sizeof(BlitPipelineCacheEntry));
    if (!renderer->blit_pipelines) {
        WEBGPU_DestroyBlitResources(renderer);
        return false;
    }

    return true;
}

static void WEBGPU_ReleaseBuffer(SDL_GPURenderer *driverData, SDL_GPUBuffer *buffer)
{
    WebGPUBuffer *webgpu_buffer = (WebGPUBuffer *)buffer;
    (void)driverData;

    if (!webgpu_buffer) {
        return;
    }
    if (webgpu_buffer->released) {
        return;
    }
    webgpu_buffer->released = true;
    WEBGPU_ReleaseBufferReference(webgpu_buffer);
}



static void WEBGPU_ReleaseTransferBuffer(SDL_GPURenderer *driverData, SDL_GPUTransferBuffer *transferBuffer)
{
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)transferBuffer;
    (void)driverData;

    if (!transfer_buffer) {
        return;
    }
    if (transfer_buffer->pending_use_count > 0) {
        transfer_buffer->released = true;
        return;
    }
    WEBGPU_DestroyTransferBuffer(transfer_buffer);
}

static void WEBGPU_ReleaseShader(SDL_GPURenderer *driverData, SDL_GPUShader *shader)
{
    WebGPUShader *webgpu_shader = (WebGPUShader *)shader;
    (void)driverData;

    if (!webgpu_shader) {
        return;
    }
    WEBGPU_DestroyShader(webgpu_shader);
}

static void WEBGPU_ReleaseComputePipeline(SDL_GPURenderer *driverData, SDL_GPUComputePipeline *computePipeline)
{
    WebGPUComputePipeline *pipeline = (WebGPUComputePipeline *)computePipeline;
    (void)driverData;

    if (!pipeline) {
        return;
    }
    if (pipeline->released) {
        return;
    }
    pipeline->released = true;
    WEBGPU_ReleaseComputePipelineReference(pipeline);
}

static void WEBGPU_ReleaseGraphicsPipeline(SDL_GPURenderer *driverData, SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    WebGPUGraphicsPipeline *pipeline = (WebGPUGraphicsPipeline *)graphicsPipeline;
    (void)driverData;

    if (!pipeline) {
        return;
    }
    if (pipeline->released) {
        return;
    }
    pipeline->released = true;
    WEBGPU_ReleaseGraphicsPipelineReference(pipeline);
}

static void WEBGPU_InitRenderPassAttachmentViews(WebGPURenderPassAttachmentViews *views)
{
    SDL_zero(*views);
}

static void WEBGPU_ReleaseRenderPassAttachmentViews(WebGPURenderPassAttachmentViews *views, Uint32 color_target_count)
{
    WEBGPU_ReleaseTextureViews(views->color_views, color_target_count);
    WEBGPU_ReleaseTextureViews(views->resolve_views, color_target_count);
    WEBGPU_ReleaseTextures(views->resolve_temp_textures, color_target_count);
    if (views->depth_view) {
        wgpuTextureViewRelease(views->depth_view);
        views->depth_view = NULL;
    }
}

static WebGPUTextureViewDescription WEBGPU_RenderPassTextureViewDescription(
    WebGPUTexture *texture,
    WebGPUTextureViewUsage usage,
    Uint32 mip_level,
    Uint32 layer_or_depth_plane)
{
    return WEBGPU_TextureViewDescription(
        texture->header.info.format,
        texture->header.info.type,
        usage,
        texture->generation,
        mip_level,
        1,
        texture->header.info.type == SDL_GPU_TEXTURETYPE_3D ? 0 : layer_or_depth_plane,
        1);
}

static bool WEBGPU_RenderPassColorAttachmentInfoFromTarget(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUColorTargetInfo *color_target_info,
    WebGPURenderPassColorAttachmentInfo *attachment)
{
    WebGPUTexture *texture;
    WebGPUTexture *resolve_texture;

    SDL_zero(*attachment);
    if (!color_target_info->texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing color target texture");
        return false;
    }

    texture = (WebGPUTexture *)color_target_info->texture;
    resolve_texture = (WebGPUTexture *)color_target_info->resolve_texture;
    attachment->texture = texture;
    attachment->view_description = WEBGPU_RenderPassTextureViewDescription(
        texture,
        WEBGPU_TEXTURE_VIEW_USAGE_COLOR_ATTACHMENT,
        color_target_info->mip_level,
        color_target_info->layer_or_depth_plane);
    attachment->clear_color = color_target_info->clear_color;
    attachment->load_op = color_target_info->load_op;
    attachment->store_op = color_target_info->store_op;
    attachment->mip_level = color_target_info->mip_level;
    attachment->layer_or_depth_plane = color_target_info->layer_or_depth_plane;
    attachment->cycle = color_target_info->cycle;
    attachment->resolve_texture = resolve_texture;
    attachment->resolve_mip_level = color_target_info->resolve_mip_level;
    attachment->resolve_layer = color_target_info->resolve_layer;
    attachment->cycle_resolve_texture = color_target_info->cycle_resolve_texture;
    if (resolve_texture) {
        attachment->resolve_view_description = WEBGPU_RenderPassTextureViewDescription(
            resolve_texture,
            WEBGPU_TEXTURE_VIEW_USAGE_RESOLVE_ATTACHMENT,
            color_target_info->resolve_mip_level,
            color_target_info->resolve_layer);
    }

    return true;
}


static bool WEBGPU_RenderPassDepthStencilAttachmentInfoFromTarget(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUDepthStencilTargetInfo *depth_stencil_target_info,
    WebGPURenderPassDepthStencilAttachmentInfo *attachment)
{
    WebGPUTexture *texture;

    SDL_zero(*attachment);
    if (!depth_stencil_target_info->texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing depth-stencil target texture");
        return false;
    }

    texture = (WebGPUTexture *)depth_stencil_target_info->texture;
    attachment->texture = texture;
    attachment->view_description = WEBGPU_RenderPassTextureViewDescription(
        texture,
        WEBGPU_TEXTURE_VIEW_USAGE_DEPTH_STENCIL_ATTACHMENT,
        depth_stencil_target_info->mip_level,
        depth_stencil_target_info->layer);
    attachment->clear_depth = depth_stencil_target_info->clear_depth;
    attachment->load_op = depth_stencil_target_info->load_op;
    attachment->store_op = depth_stencil_target_info->store_op;
    attachment->stencil_load_op = depth_stencil_target_info->stencil_load_op;
    attachment->stencil_store_op = depth_stencil_target_info->stencil_store_op;
    attachment->mip_level = depth_stencil_target_info->mip_level;
    attachment->layer = depth_stencil_target_info->layer;
    attachment->cycle = depth_stencil_target_info->cycle;
    attachment->clear_stencil = depth_stencil_target_info->clear_stencil;
    return true;
}


static bool WEBGPU_CreateRenderPassColorAttachmentView(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassColorAttachmentInfo *color_attachment,
    WGPUTextureView *color_view)
{
    WebGPUTexture *texture = color_attachment->texture;
    WebGPUTextureViewDescription color_view_description = color_attachment->view_description;
    color_view_description.generation = texture->generation;

    if (!WEBGPU_MaterializeSwapchainTextureDestination(
            command_buffer,
            &color_attachment->swapchain_destination,
            "wgpuSurfaceGetCurrentTexture failed for color target")) {
        return false;
    }
    *color_view = WEBGPU_CreateTextureViewFromDescription(texture->texture, &color_view_description);
    if (!*color_view) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for color target");
        return false;
    }

    return true;
}

static bool WEBGPU_CreateRenderPass3DResolveTexture(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassColorAttachmentInfo *color_attachment,
    WebGPUTexture *resolve_texture,
    WGPUTexture *resolve_temp_texture,
    WebGPU3DResolveCopy *resolve_copy)
{
    WGPUTextureDescriptor temp_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;

    temp_desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    temp_desc.dimension = WGPUTextureDimension_2D;
    temp_desc.size.width = WEBGPU_TextureMipDimension(resolve_texture->header.info.width, color_attachment->resolve_mip_level);
    temp_desc.size.height = WEBGPU_TextureMipDimension(resolve_texture->header.info.height, color_attachment->resolve_mip_level);
    temp_desc.size.depthOrArrayLayers = 1;
    temp_desc.format = WEBGPU_ToWGPUTextureFormat(resolve_texture->header.info.format);
    temp_desc.mipLevelCount = 1;
    temp_desc.sampleCount = 1;

    *resolve_temp_texture = wgpuDeviceCreateTexture(command_buffer->renderer->device, &temp_desc);
    if (!*resolve_temp_texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTexture failed for 3D resolve target");
        return false;
    }

    resolve_copy->source_2d_texture = *resolve_temp_texture;
    resolve_copy->destination_3d_texture = resolve_texture->texture;
    resolve_copy->width = temp_desc.size.width;
    resolve_copy->height = temp_desc.size.height;
    resolve_copy->destination_mip_level = color_attachment->resolve_mip_level;
    resolve_copy->destination_depth_plane = color_attachment->resolve_layer;
    return true;
}

static bool WEBGPU_CreateRenderPassResolveAttachmentView(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassColorAttachmentInfo *color_attachment,
    Uint32 color_target_index,
    WebGPURenderPassAttachmentViews *attachment_views)
{
    WebGPUTexture *resolve_texture = color_attachment->resolve_texture;
    WGPUTexture resolve_view_texture;
    WebGPU3DResolveCopy *resolve_copy = NULL;
    WebGPUTextureViewDescription resolve_view_description = color_attachment->resolve_view_description;

    if (!WEBGPU_MaterializeSwapchainTextureDestination(
            command_buffer,
            &color_attachment->resolve_swapchain_destination,
            "wgpuSurfaceGetCurrentTexture failed for resolve target")) {
        return false;
    }

    resolve_view_texture = resolve_texture->texture;
    resolve_view_description.generation = resolve_texture->generation;

    if (resolve_texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        resolve_copy = &attachment_views->resolve_copies[attachment_views->resolve_copy_count];

        if (!WEBGPU_CreateRenderPass3DResolveTexture(
                command_buffer,
                color_attachment,
                resolve_texture,
                &attachment_views->resolve_temp_textures[color_target_index],
                resolve_copy)) {
            return false;
        }

        resolve_view_texture = attachment_views->resolve_temp_textures[color_target_index];
        resolve_view_description = WEBGPU_TextureViewDescription(
            resolve_texture->header.info.format,
            SDL_GPU_TEXTURETYPE_2D,
            WEBGPU_TEXTURE_VIEW_USAGE_RESOLVE_ATTACHMENT,
            WEBGPU_RESOURCE_GENERATION_INITIAL,
            0,
            1,
            0,
            1);
    }

    attachment_views->resolve_views[color_target_index] = WEBGPU_CreateTextureViewFromDescription(resolve_view_texture, &resolve_view_description);
    if (!attachment_views->resolve_views[color_target_index]) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for resolve target");
        return false;
    }
    if (resolve_copy) {
        attachment_views->resolve_copy_count += 1;
    }

    return true;
}

static bool WEBGPU_CreateRenderPassDepthStencilAttachmentView(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassDepthStencilAttachmentInfo *depth_attachment,
    WGPUTextureView *depth_view)
{
    WebGPUTexture *depth_texture = depth_attachment->texture;
    WebGPUTextureViewDescription depth_view_description = depth_attachment->view_description;
    depth_view_description.generation = depth_texture->generation;

    *depth_view = WEBGPU_CreateTextureViewFromDescription(depth_texture->texture, &depth_view_description);
    if (!*depth_view) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for depth-stencil target");
        return false;
    }

    return true;
}

static bool WEBGPU_CreateRenderPassAttachmentViews(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassColorAttachmentInfo *color_attachments,
    Uint32 color_target_count,
    const WebGPURenderPassDepthStencilAttachmentInfo *depth_attachment,
    WebGPURenderPassAttachmentViews *attachment_views)
{
    for (Uint32 i = 0; i < color_target_count; i += 1) {
        const WebGPURenderPassColorAttachmentInfo *color_attachment = &color_attachments[i];
        const bool has_resolve_op = color_attachment->store_op == SDL_GPU_STOREOP_RESOLVE ||
                                    color_attachment->store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE;

        if (!WEBGPU_CreateRenderPassColorAttachmentView(
                command_buffer,
                color_attachment,
                &attachment_views->color_views[i])) {
            return false;
        }
        if (has_resolve_op &&
            !WEBGPU_CreateRenderPassResolveAttachmentView(
                command_buffer,
                color_attachment,
                i,
                attachment_views)) {
            return false;
        }
    }

    if (depth_attachment &&
        !WEBGPU_CreateRenderPassDepthStencilAttachmentView(
            command_buffer,
            depth_attachment,
            &attachment_views->depth_view)) {
        return false;
    }

    return true;
}

static bool WEBGPU_TrackRenderPassAttachmentViews(
    WebGPUCommandBuffer *command_buffer,
    const WebGPURenderPassColorAttachmentInfo *color_attachments,
    Uint32 color_target_count,
    const WebGPURenderPassDepthStencilAttachmentInfo *depth_attachment,
    const WebGPURenderPassAttachmentViews *attachment_views)
{
    for (Uint32 i = 0; i < color_target_count; i += 1) {
        WebGPUTexture *texture = color_attachments[i].texture;
        WebGPUTexture *resolve_texture = color_attachments[i].resolve_texture;

        if (!WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture) ||
            !WEBGPU_TrackCommandBufferTextureView(command_buffer, attachment_views->color_views[i]) ||
            (resolve_texture &&
             (!WEBGPU_TrackCommandBufferTexture(command_buffer, resolve_texture->texture) ||
              (attachment_views->resolve_temp_textures[i] && !WEBGPU_TrackCommandBufferTexture(command_buffer, attachment_views->resolve_temp_textures[i])) ||
              !WEBGPU_TrackCommandBufferTextureView(command_buffer, attachment_views->resolve_views[i])))) {
            WEBGPU_FailCommandBuffer(command_buffer, "failed to track render target texture");
            return false;
        }
    }
    if (depth_attachment &&
        (!WEBGPU_TrackCommandBufferTexture(command_buffer, depth_attachment->texture->texture) ||
         !WEBGPU_TrackCommandBufferTextureView(command_buffer, attachment_views->depth_view))) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track depth-stencil target texture");
        return false;
    }

    return true;
}

static void WEBGPU_INTERNAL_BeginRenderPass(
    SDL_GPUCommandBuffer *commandBuffer,
    WebGPURenderPassColorAttachmentInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const WebGPURenderPassDepthStencilAttachmentInfo *depthStencilTargetInfo)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *depth_texture = NULL;
    WebGPURenderPassAttachmentViews attachment_views;
    WGPURenderPassColorAttachment color_attachments[MAX_COLOR_TARGET_BINDINGS];
    WGPULoadOp color_load_ops[MAX_COLOR_TARGET_BINDINGS];
    WGPUStoreOp color_store_ops[MAX_COLOR_TARGET_BINDINGS];
    WGPURenderPassDepthStencilAttachment depth_stencil_attachment = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    WGPURenderPassDescriptor render_pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    WGPULoadOp depth_load_op = WGPULoadOp_Undefined;
    WGPUStoreOp depth_store_op = WGPUStoreOp_Undefined;
    WGPULoadOp stencil_load_op = WGPULoadOp_Undefined;
    WGPUStoreOp stencil_store_op = WGPUStoreOp_Undefined;
    SDL_GPUSampleCount render_pass_sample_count = SDL_GPU_SAMPLECOUNT_1;
    Uint32 render_pass_attachment_width = 0;
    Uint32 render_pass_attachment_height = 0;
    WebGPURenderAttachmentRegion render_attachment_regions[MAX_COLOR_TARGET_BINDINGS * 2];
    Uint32 render_attachment_region_count = 0;

    WEBGPU_InitRenderPassAttachmentViews(&attachment_views);
    SDL_zeroa(color_attachments);
    SDL_zeroa(color_load_ops);
    SDL_zeroa(color_store_ops);
    SDL_zeroa(render_attachment_regions);

    if (command_buffer->failed || WEBGPU_FailIfAnyPassActive(command_buffer, "BeginRenderPass")) {
        return;
    }
    if (numColorTargets == 0 && !depthStencilTargetInfo) {
        WEBGPU_FailCommandBuffer(command_buffer, "at least one color or depth-stencil target is required");
        return;
    }
    if (numColorTargets > 0 && !colorTargetInfos) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing color target infos");
        return;
    }
    if (numColorTargets > SDL_min(command_buffer->renderer->limits.maxColorAttachments, (Uint32)MAX_COLOR_TARGET_BINDINGS)) {
        WEBGPU_FailCommandBuffer(command_buffer, "color target count exceeds WebGPU maxColorAttachments");
        return;
    }

    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        WebGPUTexture *texture = colorTargetInfos[i].texture;
        WebGPUTexture *resolve_texture = colorTargetInfos[i].resolve_texture;
        const bool has_resolve_op = colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE ||
                                    colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE;

        if (!texture) {
            WEBGPU_FailCommandBuffer(command_buffer, "missing color target texture");
            return;
        }
        if (texture->released) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target texture has been released");
            return;
        }
        if (!WEBGPU_ResolveSwapchainTextureDestination(
                command_buffer,
                texture,
                WGPUTextureUsage_RenderAttachment,
                "swapchain color target must be acquired by this command buffer",
                "swapchain color target is not supported by this surface",
                &colorTargetInfos[i].swapchain_destination)) {
            return;
        }
        color_load_ops[i] = WEBGPU_ToLoadOp(colorTargetInfos[i].load_op);
        if (color_load_ops[i] == WGPULoadOp_Undefined) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid color target load op");
            return;
        }
        if (colorTargetInfos[i].cycle && colorTargetInfos[i].load_op == SDL_GPU_LOADOP_LOAD) {
            WEBGPU_FailCommandBuffer(command_buffer, "cannot cycle color target when load op is LOAD");
            return;
        }
        if (color_load_ops[i] == WGPULoadOp_Clear &&
            !WEBGPU_FColorIsFinite(colorTargetInfos[i].clear_color)) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid color target clear color");
            return;
        }
        color_store_ops[i] = WEBGPU_ToStoreOp(colorTargetInfos[i].store_op);
        if (color_store_ops[i] == WGPUStoreOp_Undefined) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid color target store op");
            return;
        }
        if (!WEBGPU_IsColorTargetTextureType(texture->header.info.type)) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target texture type is not supported");
            return;
        }
        if (!(texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target texture was not created with SDL_GPU_TEXTUREUSAGE_COLOR_TARGET");
            return;
        }
        if (colorTargetInfos[i].mip_level >= texture->header.info.num_levels) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target mip level exceeds texture levels");
            return;
        }
        if (!WEBGPU_TextureRenderLayerInBounds(texture, colorTargetInfos[i].mip_level, colorTargetInfos[i].layer_or_depth_plane)) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target layer exceeds texture layer count");
            return;
        }
        if (!WEBGPU_RecordRenderPassAttachmentExtent(
                &render_pass_attachment_width,
                &render_pass_attachment_height,
                WEBGPU_TextureMipDimension(texture->header.info.width, colorTargetInfos[i].mip_level),
                WEBGPU_TextureMipDimension(texture->header.info.height, colorTargetInfos[i].mip_level))) {
            WEBGPU_FailCommandBuffer(command_buffer, "render pass attachment extent does not match");
            return;
        }
        render_attachment_regions[render_attachment_region_count] = (WebGPURenderAttachmentRegion){
            texture,
            colorTargetInfos[i].mip_level,
            colorTargetInfos[i].layer_or_depth_plane
        };
        if (WEBGPU_RenderAttachmentRegionIsAlreadyUsed(
                render_attachment_regions,
                render_attachment_region_count,
                &render_attachment_regions[render_attachment_region_count])) {
            WEBGPU_FailCommandBuffer(command_buffer, "render attachments must not alias");
            return;
        }
        render_attachment_region_count += 1;
        if (i == 0) {
            render_pass_sample_count = texture->header.info.sample_count;
        } else if (texture->header.info.sample_count != render_pass_sample_count) {
            WEBGPU_FailCommandBuffer(command_buffer, "color target sample count does not match render pass");
            return;
        }
        if (resolve_texture && !has_resolve_op) {
            WEBGPU_FailCommandBuffer(command_buffer, "resolve texture requires a resolve store op");
            return;
        }
        if (has_resolve_op) {
            if (!resolve_texture) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve store op requires a resolve texture");
                return;
            }
            if (resolve_texture->released) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture has been released");
                return;
            }
            if (!WEBGPU_ResolveSwapchainTextureDestination(
                    command_buffer,
                    resolve_texture,
                    WGPUTextureUsage_RenderAttachment,
                    "swapchain resolve target must be acquired by this command buffer",
                    "swapchain resolve target is not supported by this surface",
                    &colorTargetInfos[i].resolve_swapchain_destination)) {
                return;
            }
            if (texture->header.info.sample_count == SDL_GPU_SAMPLECOUNT_1) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve store op requires a multisample color target");
                return;
            }
            if (resolve_texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture must have sample count 1");
                return;
            }
            if (colorTargetInfos[i].resolve_view_description.format != colorTargetInfos[i].view_description.format) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture format does not match color target format");
                return;
            }
            if (!WEBGPU_IsColorTargetTextureType(resolve_texture->header.info.type)) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture type is not supported");
                return;
            }
            if (!(resolve_texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture was not created with SDL_GPU_TEXTUREUSAGE_COLOR_TARGET");
                return;
            }
            if (colorTargetInfos[i].resolve_mip_level >= resolve_texture->header.info.num_levels) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture mip level exceeds texture levels");
                return;
            }
            if (!WEBGPU_TextureRenderLayerInBounds(resolve_texture, colorTargetInfos[i].resolve_mip_level, colorTargetInfos[i].resolve_layer)) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture layer exceeds texture layer count");
                return;
            }
            if (WEBGPU_TextureMipDimension(texture->header.info.width, colorTargetInfos[i].mip_level) !=
                    WEBGPU_TextureMipDimension(resolve_texture->header.info.width, colorTargetInfos[i].resolve_mip_level) ||
                WEBGPU_TextureMipDimension(texture->header.info.height, colorTargetInfos[i].mip_level) !=
                    WEBGPU_TextureMipDimension(resolve_texture->header.info.height, colorTargetInfos[i].resolve_mip_level)) {
                WEBGPU_FailCommandBuffer(command_buffer, "resolve texture mip extent does not match color target mip extent");
                return;
            }
            render_attachment_regions[render_attachment_region_count] = (WebGPURenderAttachmentRegion){
                resolve_texture,
                colorTargetInfos[i].resolve_mip_level,
                colorTargetInfos[i].resolve_layer
            };
            if (WEBGPU_RenderAttachmentRegionIsAlreadyUsed(
                    render_attachment_regions,
                    render_attachment_region_count,
                    &render_attachment_regions[render_attachment_region_count])) {
                WEBGPU_FailCommandBuffer(command_buffer, "render attachments must not alias");
                return;
            }
            render_attachment_region_count += 1;
        }
    }

    if (depthStencilTargetInfo) {
        if (!depthStencilTargetInfo->texture) {
            WEBGPU_FailCommandBuffer(command_buffer, "missing depth-stencil target texture");
            return;
        }
        depth_load_op = WEBGPU_ToLoadOp(depthStencilTargetInfo->load_op);
        if (depth_load_op == WGPULoadOp_Undefined) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid depth-stencil target load op");
            return;
        }
        if (depthStencilTargetInfo->cycle &&
            (depthStencilTargetInfo->load_op == SDL_GPU_LOADOP_LOAD ||
             (WEBGPU_IsStencilFormat(depthStencilTargetInfo->view_description.format) &&
              depthStencilTargetInfo->stencil_load_op == SDL_GPU_LOADOP_LOAD))) {
            WEBGPU_FailCommandBuffer(command_buffer, "cannot cycle depth-stencil target when depth or stencil load op is LOAD");
            return;
        }
        if (depth_load_op == WGPULoadOp_Clear &&
            (SDL_isnanf(depthStencilTargetInfo->clear_depth) ||
             SDL_isinff(depthStencilTargetInfo->clear_depth) ||
             depthStencilTargetInfo->clear_depth < 0.0f ||
             depthStencilTargetInfo->clear_depth > 1.0f)) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid depth-stencil target clear depth");
            return;
        }
        depth_store_op = WEBGPU_ToStoreOp(depthStencilTargetInfo->store_op);
        if (depth_store_op == WGPUStoreOp_Undefined) {
            WEBGPU_FailCommandBuffer(command_buffer, "invalid depth-stencil target store op");
            return;
        }
        if (depthStencilTargetInfo->store_op == SDL_GPU_STOREOP_RESOLVE ||
            depthStencilTargetInfo->store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE ||
            depthStencilTargetInfo->stencil_store_op == SDL_GPU_STOREOP_RESOLVE ||
            depthStencilTargetInfo->stencil_store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil resolve is not supported");
            return;
        }
        depth_texture = depthStencilTargetInfo->texture;
        if (depth_texture->released) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target texture has been released");
            return;
        }
        if (!WEBGPU_IsDepthStencilFormat(depth_texture->header.info.format)) {
            WEBGPU_FailCommandBuffer(command_buffer, "unsupported depth-stencil target format");
            return;
        }
        if (!(depth_texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target texture was not created with SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET");
            return;
        }
        if (WEBGPU_IsStencilFormat(depth_texture->header.info.format)) {
            stencil_load_op = WEBGPU_ToLoadOp(depthStencilTargetInfo->stencil_load_op);
            if (stencil_load_op == WGPULoadOp_Undefined) {
                WEBGPU_FailCommandBuffer(command_buffer, "invalid depth-stencil target stencil load op");
                return;
            }
            stencil_store_op = WEBGPU_ToStoreOp(depthStencilTargetInfo->stencil_store_op);
            if (stencil_store_op == WGPUStoreOp_Undefined) {
                WEBGPU_FailCommandBuffer(command_buffer, "invalid depth-stencil target stencil store op");
                return;
            }
        }
        if (!WEBGPU_IsDepthStencilTargetTextureType(depth_texture->header.info.type)) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target texture type is not supported");
            return;
        }
        if (depth_texture->header.info.layer_count_or_depth > SDL_MAX_UINT8) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target texture layer count exceeds SDL depth target ABI limits");
            return;
        }
        if (depthStencilTargetInfo->mip_level >= depth_texture->header.info.num_levels) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target mip level exceeds texture levels");
            return;
        }
        if (!WEBGPU_TextureRenderLayerInBounds(
                depth_texture,
                depthStencilTargetInfo->mip_level,
                depthStencilTargetInfo->layer)) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target layer exceeds texture layer count");
            return;
        }
        if (!WEBGPU_RecordRenderPassAttachmentExtent(
                &render_pass_attachment_width,
                &render_pass_attachment_height,
                WEBGPU_TextureMipDimension(depth_texture->header.info.width, depthStencilTargetInfo->mip_level),
                WEBGPU_TextureMipDimension(depth_texture->header.info.height, depthStencilTargetInfo->mip_level))) {
            WEBGPU_FailCommandBuffer(command_buffer, "render pass attachment extent does not match");
            return;
        }
        if (numColorTargets == 0) {
            render_pass_sample_count = depth_texture->header.info.sample_count;
        } else if (depth_texture->header.info.sample_count != render_pass_sample_count) {
            WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil target sample count does not match render pass");
            return;
        }
    }

    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        WebGPUTexture *texture = colorTargetInfos[i].texture;
        WebGPUTexture *resolve_texture = colorTargetInfos[i].resolve_texture;
        const bool has_resolve_op = colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE ||
                                    colorTargetInfos[i].store_op == SDL_GPU_STOREOP_RESOLVE_AND_STORE;
        const bool cycle_color_target = colorTargetInfos[i].cycle &&
                                        colorTargetInfos[i].load_op != SDL_GPU_LOADOP_LOAD;

        if (!WEBGPU_CycleTextureIfBound(command_buffer, texture, cycle_color_target, "failed to cycle color target texture")) {
            return;
        }
        if (has_resolve_op &&
            !WEBGPU_CycleTextureIfBound(command_buffer, resolve_texture, colorTargetInfos[i].cycle_resolve_texture, "failed to cycle resolve texture")) {
            return;
        }
    }
    if (depth_texture) {
        if (!WEBGPU_CycleTextureIfBound(command_buffer, depth_texture, depthStencilTargetInfo->cycle, "failed to cycle depth-stencil target texture")) {
            return;
        }
    }

    if (!WEBGPU_CreateRenderPassAttachmentViews(
            command_buffer,
            colorTargetInfos,
            numColorTargets,
            depthStencilTargetInfo,
            &attachment_views)) {
        WEBGPU_ReleaseRenderPassAttachmentViews(&attachment_views, numColorTargets);
        return;
    }

    if (!WEBGPU_TrackRenderPassAttachmentViews(
            command_buffer,
            colorTargetInfos,
            numColorTargets,
            depthStencilTargetInfo,
            &attachment_views)) {
        WEBGPU_ReleaseRenderPassAttachmentViews(&attachment_views, numColorTargets);
        return;
    }

    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        WebGPUTexture *texture = colorTargetInfos[i].texture;

        color_attachments[i] = (WGPURenderPassColorAttachment)WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        color_attachments[i].view = attachment_views.color_views[i];
        color_attachments[i].resolveTarget = attachment_views.resolve_views[i];
        color_attachments[i].depthSlice = texture->header.info.type == SDL_GPU_TEXTURETYPE_3D ? colorTargetInfos[i].layer_or_depth_plane : WGPU_DEPTH_SLICE_UNDEFINED;
        color_attachments[i].loadOp = color_load_ops[i];
        color_attachments[i].storeOp = color_store_ops[i];
        color_attachments[i].clearValue.r = colorTargetInfos[i].clear_color.r;
        color_attachments[i].clearValue.g = colorTargetInfos[i].clear_color.g;
        color_attachments[i].clearValue.b = colorTargetInfos[i].clear_color.b;
        color_attachments[i].clearValue.a = colorTargetInfos[i].clear_color.a;
    }

    render_pass_desc.colorAttachmentCount = numColorTargets;
    render_pass_desc.colorAttachments = numColorTargets > 0 ? color_attachments : NULL;
    if (depth_texture) {
        depth_stencil_attachment.view = attachment_views.depth_view;
        depth_stencil_attachment.depthLoadOp = depth_load_op;
        depth_stencil_attachment.depthStoreOp = depth_store_op;
        depth_stencil_attachment.depthClearValue = depthStencilTargetInfo->clear_depth;
        if (WEBGPU_IsStencilFormat(depthStencilTargetInfo->view_description.format)) {
            depth_stencil_attachment.stencilLoadOp = stencil_load_op;
            depth_stencil_attachment.stencilStoreOp = stencil_store_op;
            depth_stencil_attachment.stencilClearValue = depthStencilTargetInfo->clear_stencil;
        }
        render_pass_desc.depthStencilAttachment = &depth_stencil_attachment;
    }

    command_buffer->render_pass = wgpuCommandEncoderBeginRenderPass(command_buffer->encoder, &render_pass_desc);
    WEBGPU_ReleaseRenderPassAttachmentViews(&attachment_views, numColorTargets);
    if (!command_buffer->render_pass) {
        WEBGPU_FailCommandBuffer(command_buffer, "BeginRenderPass failed");
        return;
    }
    command_buffer->current_graphics_pipeline = NULL;
    command_buffer->render_pass_color_target_count = numColorTargets;
    SDL_memcpy(
        command_buffer->render_pass_3d_resolve_copies,
        attachment_views.resolve_copies,
        attachment_views.resolve_copy_count * sizeof(attachment_views.resolve_copies[0]));
    command_buffer->render_pass_3d_resolve_copy_count = attachment_views.resolve_copy_count;
    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        command_buffer->render_pass_color_targets[i] = colorTargetInfos[i].texture;
        command_buffer->render_pass_resolve_targets[i] = colorTargetInfos[i].resolve_texture;
        command_buffer->render_pass_color_target_formats[i] = colorTargetInfos[i].view_description.format;
    }
    command_buffer->render_pass_depth_stencil_target = depth_texture;
    command_buffer->render_pass_sample_count = render_pass_sample_count;
    command_buffer->render_pass_attachment_width = render_pass_attachment_width;
    command_buffer->render_pass_attachment_height = render_pass_attachment_height;
    command_buffer->render_pass_has_depth_stencil_target = depth_texture != NULL;
    command_buffer->render_pass_depth_stencil_format = depthStencilTargetInfo ? depthStencilTargetInfo->view_description.format : SDL_GPU_TEXTUREFORMAT_INVALID;
    command_buffer->stencil_reference = 0;
    if (depth_texture && WEBGPU_IsStencilFormat(depthStencilTargetInfo->view_description.format)) {
        wgpuRenderPassEncoderSetStencilReference(command_buffer->render_pass, command_buffer->stencil_reference);
    }
    command_buffer->vertex_resource_bind_group_dirty = true;
    command_buffer->fragment_resource_bind_group_dirty = true;
    command_buffer->vertex_uniform_bind_group_dirty = true;
    command_buffer->fragment_uniform_bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_PASS);
    command_buffer->bound_vertex_buffer_mask = 0;
    SDL_zeroa(command_buffer->vertex_buffers);
    SDL_zeroa(command_buffer->vertex_buffer_offsets);
    SDL_zeroa(command_buffer->vertex_buffer_sizes);
    command_buffer->index_buffer_size = 0;
    command_buffer->index_element_size = 0;
    command_buffer->index_buffer_bound = false;
}

static bool WEBGPU_BeginRenderPass(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUColorTargetInfo *colorTargetInfos,
    Uint32 numColorTargets,
    const SDL_GPUDepthStencilTargetInfo *depthStencilTargetInfo)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderPassColorAttachmentInfo color_attachments[MAX_COLOR_TARGET_BINDINGS];
    WebGPURenderPassDepthStencilAttachmentInfo depth_attachment;
    const WebGPURenderPassDepthStencilAttachmentInfo *depth_attachment_ptr = NULL;
    const Uint32 max_color_targets = SDL_min(command_buffer->renderer->limits.maxColorAttachments, (Uint32)MAX_COLOR_TARGET_BINDINGS);

    if (command_buffer->failed || WEBGPU_FailIfAnyPassActive(command_buffer, "BeginRenderPass")) {
        return false;
    }
    if (numColorTargets > max_color_targets) {
        WEBGPU_FailCommandBuffer(command_buffer, "color target count exceeds WebGPU maxColorAttachments");
        return false;
    }
    if (numColorTargets > 0 && !colorTargetInfos) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing color target infos");
        return false;
    }

    SDL_zeroa(color_attachments);
    for (Uint32 i = 0; i < numColorTargets; i += 1) {
        if (!WEBGPU_RenderPassColorAttachmentInfoFromTarget(command_buffer, &colorTargetInfos[i], &color_attachments[i])) {
            return false;
        }
    }
    if (depthStencilTargetInfo) {
        if (!WEBGPU_RenderPassDepthStencilAttachmentInfoFromTarget(command_buffer, depthStencilTargetInfo, &depth_attachment)) {
            return false;
        }
        depth_attachment_ptr = &depth_attachment;
    }

    WEBGPU_INTERNAL_BeginRenderPass(
        commandBuffer,
        color_attachments,
        numColorTargets,
        depth_attachment_ptr);

    return !command_buffer->failed && command_buffer->render_pass != NULL;
}


static void WEBGPU_ApplyVertexBufferBinding(WebGPUCommandBuffer *command_buffer, Uint32 sdl_slot)
{
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;
    Uint32 wgpu_slot;

    if (!pipeline || sdl_slot >= MAX_VERTEX_BUFFERS ||
        (command_buffer->bound_vertex_buffer_mask & (1u << sdl_slot)) == 0) {
        return;
    }

    wgpu_slot = pipeline->vertex_buffer_wgpu_slots[sdl_slot];
    if (wgpu_slot >= MAX_VERTEX_BUFFERS) {
        return;
    }

    wgpuRenderPassEncoderSetVertexBuffer(
        command_buffer->render_pass,
        wgpu_slot,
        command_buffer->vertex_buffers[sdl_slot],
        command_buffer->vertex_buffer_offsets[sdl_slot],
        command_buffer->vertex_buffer_sizes[sdl_slot]);
}

static void WEBGPU_BindGraphicsPipeline(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUGraphicsPipeline *graphicsPipeline)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUGraphicsPipeline *pipeline = (WebGPUGraphicsPipeline *)graphicsPipeline;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindGraphicsPipeline")) {
        return;
    }
    if (!pipeline || pipeline->released || !pipeline->pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid graphics pipeline");
        return;
    }
    if (pipeline->color_target_count != command_buffer->render_pass_color_target_count) {
        WEBGPU_FailCommandBuffer(command_buffer, "graphics pipeline color target count does not match render pass");
        return;
    }
    for (Uint32 i = 0; i < pipeline->color_target_count; i += 1) {
        if (pipeline->color_target_formats[i] != command_buffer->render_pass_color_target_formats[i]) {
            WEBGPU_FailCommandBuffer(command_buffer, "graphics pipeline color target format does not match render pass");
            return;
        }
    }
    if (pipeline->has_depth_stencil_target != command_buffer->render_pass_has_depth_stencil_target ||
        (pipeline->has_depth_stencil_target && pipeline->depth_stencil_format != command_buffer->render_pass_depth_stencil_format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "graphics pipeline depth-stencil state does not match render pass");
        return;
    }
    if (pipeline->sample_count != command_buffer->render_pass_sample_count) {
        WEBGPU_FailCommandBuffer(command_buffer, "graphics pipeline sample count does not match render pass");
        return;
    }
    if (!WEBGPU_TrackCommandBufferGraphicsPipeline(command_buffer, pipeline)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track graphics pipeline");
        return;
    }
    wgpuRenderPassEncoderSetPipeline(command_buffer->render_pass, pipeline->pipeline);
    command_buffer->current_graphics_pipeline = pipeline;
    command_buffer->vertex_resource_bind_group_dirty = true;
    command_buffer->fragment_resource_bind_group_dirty = true;
    command_buffer->vertex_uniform_bind_group_dirty = true;
    command_buffer->fragment_uniform_bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_PIPELINE);

    for (Uint32 slot = 0; slot < MAX_VERTEX_BUFFERS; slot += 1) {
        WEBGPU_ApplyVertexBufferBinding(command_buffer, slot);
    }
}

static void WEBGPU_SetViewport(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUViewport *viewport)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "SetViewport")) {
        return;
    }
    if (SDL_isnanf(viewport->x) || SDL_isinff(viewport->x) ||
        SDL_isnanf(viewport->y) || SDL_isinff(viewport->y) ||
        SDL_isnanf(viewport->w) || SDL_isinff(viewport->w) ||
        SDL_isnanf(viewport->h) || SDL_isinff(viewport->h) ||
        SDL_isnanf(viewport->min_depth) || SDL_isinff(viewport->min_depth) ||
        SDL_isnanf(viewport->max_depth) || SDL_isinff(viewport->max_depth)) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid viewport values");
        return;
    }
    if (viewport->w < 0.0f || viewport->h < 0.0f) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid viewport dimensions");
        return;
    }
    if (!WEBGPU_ViewportFitsLimits(command_buffer->renderer, viewport)) {
        WEBGPU_FailCommandBuffer(command_buffer, "viewport exceeds WebGPU limits");
        return;
    }
    if (viewport->min_depth < 0.0f || viewport->min_depth > 1.0f ||
        viewport->max_depth < 0.0f || viewport->max_depth > 1.0f) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid viewport depth range");
        return;
    }
    if (viewport->min_depth > viewport->max_depth) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid viewport depth range");
        return;
    }
    wgpuRenderPassEncoderSetViewport(command_buffer->render_pass, viewport->x, viewport->y, viewport->w, viewport->h, viewport->min_depth, viewport->max_depth);
}

static void WEBGPU_SetScissor(SDL_GPUCommandBuffer *commandBuffer, const SDL_Rect *scissor)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    Uint64 x, y, w, h;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "SetScissor")) {
        return;
    }
    if (scissor->x < 0 || scissor->y < 0 || scissor->w < 0 || scissor->h < 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid scissor rectangle");
        return;
    }
    x = (Uint32)scissor->x;
    y = (Uint32)scissor->y;
    w = (Uint32)scissor->w;
    h = (Uint32)scissor->h;
    if (x + w > command_buffer->render_pass_attachment_width ||
        y + h > command_buffer->render_pass_attachment_height) {
        WEBGPU_FailCommandBuffer(command_buffer, "scissor rectangle exceeds render pass extent");
        return;
    }
    wgpuRenderPassEncoderSetScissorRect(command_buffer->render_pass, (uint32_t)scissor->x, (uint32_t)scissor->y, (uint32_t)scissor->w, (uint32_t)scissor->h);
}

static void WEBGPU_SetBlendConstants(SDL_GPUCommandBuffer *commandBuffer, SDL_FColor blendConstants)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WGPUColor color = WGPU_COLOR_INIT;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "SetBlendConstants")) {
        return;
    }
    if (!WEBGPU_FColorIsFinite(blendConstants)) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid blend constant values");
        return;
    }
    color.r = blendConstants.r;
    color.g = blendConstants.g;
    color.b = blendConstants.b;
    color.a = blendConstants.a;
    wgpuRenderPassEncoderSetBlendConstant(command_buffer->render_pass, &color);
}

static void WEBGPU_SetStencilReference(SDL_GPUCommandBuffer *commandBuffer, Uint8 reference)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "SetStencilReference")) {
        return;
    }
    command_buffer->stencil_reference = reference;
    if (command_buffer->render_pass_has_depth_stencil_target &&
        WEBGPU_IsStencilFormat(command_buffer->render_pass_depth_stencil_format)) {
        wgpuRenderPassEncoderSetStencilReference(command_buffer->render_pass, reference);
    }
}

static void WEBGPU_BindVertexBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUBufferBinding *bindings, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindVertexBuffers")) {
        return;
    }

    for (Uint32 i = 0; i < numBindings; i += 1) {
        WebGPUBuffer *buffer = (WebGPUBuffer *)bindings[i].buffer;
        Uint32 slot = firstSlot + i;
        if (!buffer || !buffer->buffer) {
            continue;
        }
        if (buffer->released) {
            WEBGPU_FailCommandBuffer(command_buffer, "vertex buffer has been released");
            return;
        }
        if (slot >= MAX_VERTEX_BUFFERS) {
            WEBGPU_FailCommandBuffer(command_buffer, "vertex buffer binding slot exceeds MAX_VERTEX_BUFFERS");
            return;
        }
        if ((buffer->usage & SDL_GPU_BUFFERUSAGE_VERTEX) == 0) {
            WEBGPU_FailCommandBuffer(command_buffer, "buffer was not created with SDL_GPU_BUFFERUSAGE_VERTEX");
            return;
        }
        if (bindings[i].offset > buffer->size) {
            WEBGPU_FailCommandBuffer(command_buffer, "vertex buffer binding offset exceeds buffer size");
            return;
        }
        if ((bindings[i].offset % 4) != 0) {
            WEBGPU_FailCommandBuffer(command_buffer, "vertex buffer binding offset must be 4-byte aligned");
            return;
        }
        if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, buffer->buffer)) {
            WEBGPU_FailCommandBuffer(command_buffer, "failed to track vertex buffer");
            return;
        }
        command_buffer->vertex_buffers[slot] = buffer->buffer;
        command_buffer->vertex_buffer_offsets[slot] = bindings[i].offset;
        command_buffer->vertex_buffer_sizes[slot] = (Uint64)buffer->size - bindings[i].offset;
        command_buffer->bound_vertex_buffer_mask |= 1u << slot;
        WEBGPU_ApplyVertexBufferBinding(command_buffer, slot);
    }
}

static void WEBGPU_BindIndexBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferBinding *binding, SDL_GPUIndexElementSize indexElementSize)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *buffer = (WebGPUBuffer *)binding->buffer;
    WGPUIndexFormat index_format;
    Uint32 index_element_size;

    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindIndexBuffer")) {
        return;
    }
    if (!buffer || !buffer->buffer) {
        return;
    }
    if (buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "index buffer has been released");
        return;
    }
    index_format = WEBGPU_ToIndexFormat(indexElementSize);
    if (index_format == WGPUIndexFormat_Undefined) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid index element size");
        return;
    }
    index_element_size = index_format == WGPUIndexFormat_Uint16 ? 2u : 4u;
    if ((buffer->usage & SDL_GPU_BUFFERUSAGE_INDEX) == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "buffer was not created with SDL_GPU_BUFFERUSAGE_INDEX");
        return;
    }
    if (binding->offset > buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "index buffer binding offset exceeds buffer size");
        return;
    }
    if ((binding->offset % index_element_size) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "index buffer binding offset is not aligned to index element size");
        return;
    }
    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track index buffer");
        return;
    }
    wgpuRenderPassEncoderSetIndexBuffer(command_buffer->render_pass, buffer->buffer, index_format, binding->offset, buffer->size - binding->offset);
    command_buffer->index_buffer_size = (Uint64)buffer->size - binding->offset;
    command_buffer->index_element_size = index_element_size;
    command_buffer->index_buffer_bound = true;
}

static bool WEBGPU_TextureIsCurrentRenderAttachment(WebGPUCommandBuffer *command_buffer, WebGPUTexture *texture)
{
    if (!texture) {
        return false;
    }
    if (texture == command_buffer->render_pass_depth_stencil_target) {
        return true;
    }

    for (Uint32 i = 0; i < command_buffer->render_pass_color_target_count; i += 1) {
        if (texture == command_buffer->render_pass_color_targets[i] ||
            texture == command_buffer->render_pass_resolve_targets[i]) {
            return true;
        }
    }
    return false;
}

typedef struct WebGPUSampledTextureBindGroupEntryErrors
{
    const char *missing_binding;
    const char *invalid_generation;
    const char *invalid_usage;
    const char *type_mismatch;
    const char *sample_count_mismatch;
    const char *sample_type_mismatch;
    const char *missing_sampler;
    const char *sampler_type_mismatch;
} WebGPUSampledTextureBindGroupEntryErrors;

static bool WEBGPU_AppendSampledTextureBindGroupEntries(
    WebGPUCommandBuffer *command_buffer,
    WGPUBindGroupEntry *entries,
    Uint32 *entry_count,
    const WebGPUSampledTextureBindingDescription *bindings,
    const WebGPUSampledTextureBindingLayout *layouts,
    Uint32 sampler_count,
    bool allow_blit_source,
    const WebGPUSampledTextureBindGroupEntryErrors *errors)
{
    for (Uint32 i = 0; i < sampler_count; i += 1) {
        const WebGPUSampledTextureBindingDescription *binding = &bindings[i];
        const WebGPUSampledTextureBindingLayout *layout = &layouts[i];
        WGPUBindGroupEntry *texture_entry = &entries[*entry_count];
        bool bound_multisampled;

        if (!binding->view) {
            WEBGPU_SetStringError(errors->missing_binding);
            return false;
        }
        if (binding->generation == 0) {
            WEBGPU_SetStringError(errors->invalid_generation);
            return false;
        }
        if (binding->view_usage != WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED &&
            !(allow_blit_source && binding->view_usage == WEBGPU_TEXTURE_VIEW_USAGE_BLIT_SOURCE)) {
            WEBGPU_SetStringError(errors->invalid_usage);
            return false;
        }
        if (binding->view_dimension == WGPUTextureViewDimension_Undefined) {
            WEBGPU_SetStringError("unsupported sampled texture type");
            return false;
        }
        if (binding->view_dimension != layout->view_dimension) {
            WEBGPU_SetStringError(errors->type_mismatch);
            return false;
        }
        bound_multisampled = binding->sample_count > SDL_GPU_SAMPLECOUNT_1;
        if (bound_multisampled != layout->multisampled) {
            WEBGPU_SetStringError(errors->sample_count_mismatch);
            return false;
        }
        if (!WEBGPU_TextureFormatMatchesSampledTextureLayout(
                command_buffer->renderer,
                binding->format,
                layout->sample_type,
                layout->multisampled)) {
            WEBGPU_SetStringError(errors->sample_type_mismatch);
            return false;
        }

        if (layout->has_sampler) {
            if (!binding->sampler) {
                WEBGPU_SetStringError(errors->missing_sampler);
                return false;
            }
            if (!WEBGPU_SamplerBindingTypeMatchesLayout(
                    layout->sampler_type,
                    binding->sampler_binding_type)) {
                WEBGPU_SetStringError(errors->sampler_type_mismatch);
                return false;
            }
        } else if (binding->sampler) {
            WEBGPU_SetStringError("samplerless sampled texture slot must not bind a sampler object");
            return false;
        }

        *texture_entry = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
        texture_entry->binding = i * 2;
        texture_entry->textureView = binding->view;
        *entry_count += 1;

        if (layout->has_sampler) {
            WGPUBindGroupEntry *sampler_entry = &entries[*entry_count];

            *sampler_entry = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
            sampler_entry->binding = i * 2 + 1;
            sampler_entry->sampler = binding->sampler;
            *entry_count += 1;
        }
    }

    return true;
}

static bool WEBGPU_AppendGraphicsStorageTextureBindGroupEntries(
    WGPUBindGroupEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    const WebGPUStorageTextureBindingDescription *bindings,
    const WebGPUStorageTextureBindingLayout *layouts,
    Uint32 storage_texture_count)
{
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        const WebGPUStorageTextureBindingDescription *binding = &bindings[i];
        const WebGPUStorageTextureBindingLayout *layout = &layouts[i];
        WGPUBindGroupEntry *storage_texture_entry = &entries[*entry_count];
        WGPUTextureViewDimension bound_view_dimension;
        WGPUTextureFormat bound_format;

        if (!binding->view) {
            WEBGPU_SetStringError("missing storage texture binding for graphics pipeline");
            return false;
        }
        if (binding->generation == 0) {
            WEBGPU_SetStringError("invalid storage texture binding generation");
            return false;
        }
        if (binding->view_usage != WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ) {
            WEBGPU_SetStringError("invalid storage texture binding usage");
            return false;
        }
        if (layout->access != WGPUStorageTextureAccess_ReadOnly) {
            WEBGPU_SetStringError("graphics storage texture layout must be read-only");
            return false;
        }
        if (binding->view_dimension == WGPUTextureViewDimension_Undefined) {
            WEBGPU_SetStringError("unsupported storage texture type");
            return false;
        }
        bound_view_dimension = binding->view_dimension;
        if (bound_view_dimension != layout->view_dimension) {
            WEBGPU_SetStringError("storage texture type does not match shader resource layout");
            return false;
        }
        bound_format = WEBGPU_ToWGPUTextureFormat(binding->format);
        if (bound_format != layout->format) {
            WEBGPU_SetStringError("storage texture format does not match shader resource layout");
            return false;
        }

        *storage_texture_entry = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
        storage_texture_entry->binding = first_binding + i;
        storage_texture_entry->textureView = binding->view;
        *entry_count += 1;
    }

    return true;
}

static bool WEBGPU_AppendComputeStorageTextureBindGroupEntries(
    WGPUBindGroupEntry *entries,
    Uint32 *entry_count,
    Uint32 first_binding,
    const WebGPUStorageTextureBindingDescription *bindings,
    const WebGPUStorageTextureBindingLayout *layouts,
    Uint32 storage_texture_count,
    WebGPUTextureViewUsage expected_usage,
    const char *missing_binding,
    const char *invalid_generation,
    const char *invalid_usage,
    const char *binding_context)
{
    for (Uint32 i = 0; i < storage_texture_count; i += 1) {
        const WebGPUStorageTextureBindingDescription *binding = &bindings[i];
        WGPUBindGroupEntry *storage_texture_entry = &entries[*entry_count];
        WGPUTextureViewDimension bound_view_dimension;

        if (!binding->view) {
            WEBGPU_SetStringError(missing_binding);
            return false;
        }
        if (binding->generation == 0) {
            WEBGPU_SetStringError(invalid_generation);
            return false;
        }
        if (binding->view_usage != expected_usage) {
            WEBGPU_SetStringError(invalid_usage);
            return false;
        }
        if (binding->view_dimension == WGPUTextureViewDimension_Undefined) {
            WEBGPU_SetStringError("unsupported storage texture type");
            return false;
        }
        bound_view_dimension = binding->view_dimension;
        if (!WEBGPU_ValidateComputeStorageTextureViewBindingLayout(
                &layouts[i],
                bound_view_dimension,
                binding->format,
                binding->usage,
                binding_context)) {
            return false;
        }

        *storage_texture_entry = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
        storage_texture_entry->binding = first_binding + i;
        storage_texture_entry->textureView = binding->view;
        *entry_count += 1;
    }

    return true;
}

static WebGPUBufferBindingDescription WEBGPU_WholeBufferBindingDescription(
    WebGPUBuffer *buffer,
    WebGPUBufferBindingUsage usage)
{
    WebGPUBufferBindingDescription description;

    description.buffer = buffer->buffer;
    description.generation = buffer->generation;
    description.offset = 0;
    description.size = buffer->size;
    description.usage = usage;
    return description;
}

static bool WEBGPU_BufferBindingDescriptionsMatch(
    const WebGPUBufferBindingDescription *a,
    const WebGPUBufferBindingDescription *b)
{
    return a->buffer == b->buffer &&
           a->generation == b->generation &&
           a->offset == b->offset &&
           a->size == b->size &&
           a->usage == b->usage;
}

static bool WEBGPU_BufferBindingUsageIsValid(WebGPUBufferBindingUsage usage)
{
    switch (usage) {
    case WEBGPU_BUFFER_BINDING_USAGE_GRAPHICS_STORAGE_READ:
    case WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ:
    case WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ_WRITE:
        return true;
    default:
        return false;
    }
}

static bool WEBGPU_InitBufferBindGroupEntry(
    WGPUBindGroupEntry *entry,
    Uint32 binding,
    const WebGPUBufferBindingDescription *description,
    const char *missing_error)
{
    if (!description->buffer || description->size == 0) {
        WEBGPU_SetStringError(missing_error);
        return false;
    }
    if (description->generation == 0) {
        WEBGPU_SetStringError("invalid buffer binding generation");
        return false;
    }
    if (!WEBGPU_BufferBindingUsageIsValid(description->usage)) {
        WEBGPU_SetStringError("invalid buffer binding usage");
        return false;
    }

    *entry = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
    entry->binding = binding;
    entry->buffer = description->buffer;
    entry->offset = description->offset;
    entry->size = description->size;
    return true;
}

static bool WEBGPU_AppendStorageBufferBindGroupEntries(
    WGPUBindGroupEntry *entries,
    Uint32 *entry_count,
    Uint32 base_binding,
    const WebGPUBufferBindingDescription *buffer_bindings,
    Uint32 buffer_count,
    const char *missing_error)
{
    for (Uint32 i = 0; i < buffer_count; i += 1) {
        if (!WEBGPU_InitBufferBindGroupEntry(
                &entries[*entry_count],
                base_binding + i,
                &buffer_bindings[i],
                missing_error)) {
            return false;
        }

        *entry_count += 1;
    }

    return true;
}

typedef struct WebGPUSampledTextureBindErrors
{
    const char *slot_exceeds_limit;
    const char *invalid_texture;
    const char *invalid_sampler;
    const char *swapchain_texture;
    const char *missing_usage;
    const char *render_attachment_alias;
    const char *unsupported_texture_type;
    const char *track_resources;
} WebGPUSampledTextureBindErrors;

static const WebGPUSampledTextureBindErrors WEBGPU_GraphicsSampledTextureBindErrors = {
    "sampler binding slot exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE",
    "invalid sampled texture binding",
    "invalid sampler binding",
    "swapchain texture sampling is not supported",
    "texture was not created with SDL_GPU_TEXTUREUSAGE_SAMPLER",
    "sampled texture is already used as a render attachment in this pass",
    "only 2D, 2D array, 3D, cube, and cube array sampled textures are supported",
    "failed to track sampler binding resources"
};

static const WebGPUSampledTextureBindErrors WEBGPU_ComputeSampledTextureBindErrors = {
    "compute sampler slot exceeds MAX_TEXTURE_SAMPLERS_PER_STAGE",
    "invalid compute sampled texture binding",
    "invalid compute sampler binding",
    "swapchain texture sampling is not supported",
    "compute sampled texture was not created with SDL_GPU_TEXTUREUSAGE_SAMPLER",
    NULL,
    NULL,
    "failed to track compute sampler binding resources"
};


static void WEBGPU_BindSampledTextures(
    WebGPUCommandBuffer *command_buffer,
    Uint32 first_slot,
    const SDL_GPUTextureSamplerBinding *texture_sampler_bindings,
    Uint32 num_bindings,
    WebGPUSampledTextureBindingDescription *bindings,
    bool *bind_group_dirty,
    bool reject_render_attachment_alias,
    bool validate_texture_type,
    const WebGPUSampledTextureBindErrors *errors)
{
    WebGPUSampledTextureBindingDescription new_bindings[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUTexture *tracked_textures[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    WebGPUSampler *tracked_samplers[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    bool changed_bindings[MAX_TEXTURE_SAMPLERS_PER_STAGE];
    bool changed = false;

    SDL_zeroa(changed_bindings);

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        WebGPUTexture *texture = (WebGPUTexture *)texture_sampler_bindings[i].texture;
        WebGPUSampler *sampler = (WebGPUSampler *)texture_sampler_bindings[i].sampler;
        WebGPUSampledTextureBindingDescription new_binding;
        WGPUTextureViewDimension view_dimension;
        Uint32 slot = first_slot + i;

        if (slot >= MAX_TEXTURE_SAMPLERS_PER_STAGE) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->slot_exceeds_limit);
            return;
        }
        if (!texture || texture->released) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_texture);
            return;
        }
        if (texture->from_surface) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->swapchain_texture);
            return;
        }
        if (!texture->texture || !texture->view) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_texture);
            return;
        }
        if (sampler && !sampler->sampler) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_sampler);
            return;
        }
        if (!(texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->missing_usage);
            return;
        }
        if (reject_render_attachment_alias &&
            WEBGPU_TextureIsCurrentRenderAttachment(command_buffer, texture)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->render_attachment_alias);
            return;
        }
        if (!WEBGPU_TrySampledTextureViewDimension(texture->header.info.type, &view_dimension) && validate_texture_type) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->unsupported_texture_type);
            return;
        }
        new_binding = WEBGPU_SampledTextureBindingDescriptionFromTexture(
            texture,
            texture->view,
            sampler,
            view_dimension,
            WEBGPU_TEXTURE_VIEW_USAGE_SAMPLED);
        new_bindings[slot] = new_binding;
        if (!WEBGPU_SampledTextureBindingDescriptionsMatch(&bindings[slot], &new_binding)) {
            tracked_textures[slot] = texture;
            tracked_samplers[slot] = sampler;
            changed_bindings[slot] = true;
            changed = true;
        }
    }

    if (!changed) {
        return;
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;
        WebGPUTexture *texture;
        WebGPUSampler *sampler;

        if (!changed_bindings[slot]) {
            continue;
        }

        texture = tracked_textures[slot];
        sampler = tracked_samplers[slot];
        if (!WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture) ||
            !WEBGPU_TrackCommandBufferTextureView(command_buffer, texture->view)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->track_resources);
            return;
        }
        if (sampler && !WEBGPU_TrackCommandBufferSampler(command_buffer, sampler->sampler)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->track_resources);
            return;
        }
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;

        if (changed_bindings[slot]) {
            bindings[slot] = new_bindings[slot];
        }
    }

    *bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_SAMPLED_RESOURCE);
}


typedef struct WebGPUStorageBufferBindErrors
{
    const char *slot_exceeds_limit;
    const char *invalid_buffer;
    const char *missing_usage;
    const char *track_buffer;
} WebGPUStorageBufferBindErrors;

static const WebGPUStorageBufferBindErrors WEBGPU_GraphicsStorageBufferBindErrors = {
    "storage buffer binding slot exceeds MAX_STORAGE_BUFFERS_PER_STAGE",
    "invalid storage buffer binding",
    "buffer was not created with SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ",
    "failed to track storage buffer"
};

static const WebGPUStorageBufferBindErrors WEBGPU_ComputeReadOnlyStorageBufferBindErrors = {
    "compute storage buffer slot exceeds MAX_STORAGE_BUFFERS_PER_STAGE",
    "missing read-only storage buffer for compute pass",
    "compute read-only storage buffer is missing COMPUTE_STORAGE_READ usage",
    "failed to track compute read-only storage buffer"
};

static void WEBGPU_BindStorageBufferBindings(
    WebGPUCommandBuffer *command_buffer,
    Uint32 first_slot,
    SDL_GPUBuffer *const *storage_buffers,
    Uint32 num_bindings,
    WebGPUBufferBindingDescription *buffer_bindings,
    bool *bind_group_dirty,
    SDL_GPUBufferUsageFlags required_usage,
    WebGPUBufferBindingUsage binding_usage,
    bool reject_zero_size,
    const WebGPUStorageBufferBindErrors *errors)
{
    WebGPUBufferBindingDescription new_bindings[MAX_STORAGE_BUFFERS_PER_STAGE];
    WebGPUBuffer *tracked_buffers[MAX_STORAGE_BUFFERS_PER_STAGE];
    bool changed_bindings[MAX_STORAGE_BUFFERS_PER_STAGE];
    bool changed = false;

    SDL_zeroa(changed_bindings);

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        WebGPUBuffer *buffer = (WebGPUBuffer *)storage_buffers[i];
        WebGPUBufferBindingDescription new_binding;
        Uint32 slot = first_slot + i;

        if (slot >= MAX_STORAGE_BUFFERS_PER_STAGE) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->slot_exceeds_limit);
            return;
        }
        if (!buffer || !buffer->buffer || (reject_zero_size && buffer->size == 0)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_buffer);
            return;
        }
        if (buffer->released) {
            WEBGPU_FailCommandBuffer(command_buffer, "storage buffer has been released");
            return;
        }
        if (!(buffer->usage & required_usage)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->missing_usage);
            return;
        }
        new_binding = WEBGPU_WholeBufferBindingDescription(buffer, binding_usage);
        new_bindings[slot] = new_binding;
        if (!WEBGPU_BufferBindingDescriptionsMatch(&buffer_bindings[slot], &new_binding)) {
            tracked_buffers[slot] = buffer;
            changed_bindings[slot] = true;
            changed = true;
        }
    }

    if (!changed) {
        return;
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;

        if (!changed_bindings[slot]) {
            continue;
        }
        if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, tracked_buffers[slot]->buffer)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->track_buffer);
            return;
        }
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;

        if (changed_bindings[slot]) {
            buffer_bindings[slot] = new_bindings[slot];
        }
    }

    *bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_STORAGE_RESOURCE);
}


typedef struct WebGPUStorageTextureBindErrors
{
    const char *slot_exceeds_limit;
    const char *invalid_texture;
    const char *missing_usage;
    const char *sample_mip_mismatch;
    const char *render_attachment_alias;
    const char *track_texture;
} WebGPUStorageTextureBindErrors;

static const WebGPUStorageTextureBindErrors WEBGPU_GraphicsStorageTextureBindErrors = {
    "storage texture binding slot exceeds MAX_STORAGE_TEXTURES_PER_STAGE",
    "invalid storage texture binding",
    "texture was not created with SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ",
    "graphics storage texture binding requires sample count 1 and one mip level",
    "storage texture is already used as a render attachment in this pass",
    "failed to track storage texture"
};

static const WebGPUStorageTextureBindErrors WEBGPU_ComputeReadOnlyStorageTextureBindErrors = {
    "compute storage texture slot exceeds MAX_STORAGE_TEXTURES_PER_STAGE",
    "missing read-only storage texture for compute pass",
    "compute read-only storage texture is missing standalone COMPUTE_STORAGE_READ usage",
    "compute read-only storage texture binding requires sample count 1 and one mip level",
    NULL,
    "failed to track compute read-only storage texture"
};

static void WEBGPU_BindStorageTextureBindings(
    WebGPUCommandBuffer *command_buffer,
    Uint32 first_slot,
    SDL_GPUTexture *const *storage_textures,
    Uint32 num_bindings,
    WebGPUStorageTextureBindingDescription *bindings,
    bool *bind_group_dirty,
    SDL_GPUTextureUsageFlags required_usage,
    bool require_exact_usage,
    bool check_view_before_usage,
    bool reject_render_attachment_alias,
    const WebGPUStorageTextureBindErrors *errors)
{
    WebGPUStorageTextureBindingDescription new_bindings[MAX_STORAGE_TEXTURES_PER_STAGE];
    WebGPUTexture *tracked_textures[MAX_STORAGE_TEXTURES_PER_STAGE];
    bool changed_bindings[MAX_STORAGE_TEXTURES_PER_STAGE];
    bool changed = false;

    SDL_zeroa(changed_bindings);

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        WebGPUTexture *texture = (WebGPUTexture *)storage_textures[i];
        WebGPUStorageTextureBindingDescription new_binding;
        WGPUTextureViewDimension view_dimension;
        Uint32 slot = first_slot + i;

        if (slot >= MAX_STORAGE_TEXTURES_PER_STAGE) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->slot_exceeds_limit);
            return;
        }
        if (texture && texture->from_surface) {
            WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture storage binding is not supported");
            return;
        }
        if (!texture || !texture->texture) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_texture);
            return;
        }
        if (texture->released) {
            WEBGPU_FailCommandBuffer(command_buffer, "storage texture has been released");
            return;
        }
        if (check_view_before_usage && !texture->storage_view) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_texture);
            return;
        }
        const bool usage_matches = require_exact_usage ?
            texture->header.info.usage == required_usage :
            (texture->header.info.usage & required_usage) != 0;
        if (!usage_matches) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->missing_usage);
            return;
        }
        if (!texture->storage_view) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->invalid_texture);
            return;
        }
        if (texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1 ||
            texture->header.info.num_levels != 1) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->sample_mip_mismatch);
            return;
        }
        if (reject_render_attachment_alias &&
            WEBGPU_TextureIsCurrentRenderAttachment(command_buffer, texture)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->render_attachment_alias);
            return;
        }
        if (!WEBGPU_ToStorageTextureViewDimension(texture->header.info.type, &view_dimension)) {
            WEBGPU_FailCommandBuffer(command_buffer, "unsupported storage texture type");
            return;
        }

        new_binding = WEBGPU_StorageTextureBindingDescription(
            texture,
            texture->storage_view,
            view_dimension,
            WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ);
        new_bindings[slot] = new_binding;
        if (!WEBGPU_StorageTextureBindingDescriptionsMatch(&bindings[slot], &new_binding)) {
            tracked_textures[slot] = texture;
            changed_bindings[slot] = true;
            changed = true;
        }
    }

    if (!changed) {
        return;
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;
        WebGPUTexture *texture;

        if (!changed_bindings[slot]) {
            continue;
        }

        texture = tracked_textures[slot];
        if (!WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture) ||
            !WEBGPU_TrackCommandBufferTextureView(command_buffer, texture->storage_view)) {
            WEBGPU_FailCommandBuffer(command_buffer, errors->track_texture);
            return;
        }
    }

    for (Uint32 i = 0; i < num_bindings; i += 1) {
        Uint32 slot = first_slot + i;

        if (changed_bindings[slot]) {
            bindings[slot] = new_bindings[slot];
        }
    }

    *bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_STORAGE_RESOURCE);
}


static bool WEBGPU_AcquirePreparedResourceBindGroup(
    WebGPUCommandBuffer *command_buffer,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    WGPUBindGroupLayout layout,
    WGPUBindGroupEntry *entries,
    Uint32 entry_count,
    const WebGPUSampledTextureBindingDescription *sampled_texture_bindings,
    Uint32 sampler_count,
    const WebGPUStorageTextureBindingDescription *storage_texture_bindings,
    Uint32 storage_texture_count,
    const WebGPUBufferBindingDescription *storage_buffer_bindings,
    Uint32 storage_buffer_count,
    const char *scoped_create_context,
    const char *unscoped_create_error,
    const char *track_error,
    WGPUBindGroup *bind_group)
{
    WGPUBindGroupDescriptor bind_group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    WebGPUResourceBindGroupCacheKey cache_key;

    bind_group_desc.layout = layout;
    bind_group_desc.entryCount = entry_count;
    bind_group_desc.entries = entries;
    WEBGPU_InitResourceBindGroupCacheKeyFromBindings(
        &cache_key,
        instrumentation_path,
        bind_group_desc.layout,
        entry_count,
        sampled_texture_bindings,
        sampler_count,
        storage_texture_bindings,
        storage_texture_count,
        storage_buffer_bindings,
        storage_buffer_count);

    return WEBGPU_AcquireResourceBindGroup(
        command_buffer,
        &bind_group_desc,
        &cache_key,
        instrumentation_path,
        scoped_create_context,
        unscoped_create_error,
        track_error,
        bind_group);
}

static bool WEBGPU_ApplyPreparedResourceBindGroup(
    WebGPUCommandBuffer *command_buffer,
    Uint32 group_index,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    WGPUBindGroupLayout layout,
    WGPUBindGroupEntry *entries,
    Uint32 entry_count,
    const WebGPUSampledTextureBindingDescription *sampled_texture_bindings,
    Uint32 sampler_count,
    const WebGPUStorageTextureBindingDescription *storage_texture_bindings,
    Uint32 storage_texture_count,
    const WebGPUBufferBindingDescription *storage_buffer_bindings,
    Uint32 storage_buffer_count,
    bool *bind_group_dirty,
    const char *scoped_create_context,
    const char *unscoped_create_error,
    const char *track_error,
    bool compute_pass)
{
    WGPUBindGroup bind_group;

    if (!WEBGPU_AcquirePreparedResourceBindGroup(
            command_buffer,
            instrumentation_path,
            layout,
            entries,
            entry_count,
            sampled_texture_bindings,
            sampler_count,
            storage_texture_bindings,
            storage_texture_count,
            storage_buffer_bindings,
            storage_buffer_count,
            scoped_create_context,
            unscoped_create_error,
            track_error,
            &bind_group)) {
        return false;
    }

    if (compute_pass) {
        wgpuComputePassEncoderSetBindGroup(command_buffer->compute_pass, group_index, bind_group, 0, NULL);
    } else {
        wgpuRenderPassEncoderSetBindGroup(command_buffer->render_pass, group_index, bind_group, 0, NULL);
    }
    WEBGPU_RecordBindGroupSet(command_buffer->renderer, instrumentation_path);
    *bind_group_dirty = false;
    return true;
}

static bool WEBGPU_ApplyResourceBindGroup(
    WebGPUCommandBuffer *command_buffer,
    Uint32 group_index,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    const WebGPUShaderResourceLayout *shader_layout,
    WebGPUSampledTextureBindingDescription *sampled_texture_bindings,
    WebGPUStorageTextureBindingDescription *storage_texture_bindings,
    const WebGPUBufferBindingDescription *storage_buffer_bindings,
    bool *bind_group_dirty)
{
    WGPUBindGroupEntry entries[MAX_TEXTURE_SAMPLERS_PER_STAGE * 2 + MAX_STORAGE_TEXTURES_PER_STAGE + MAX_STORAGE_BUFFERS_PER_STAGE];
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;
    WGPUBindGroupLayout layout;
    Uint32 sampler_count = shader_layout->sampler_count;
    Uint32 storage_texture_count = shader_layout->storage_texture_count;
    Uint32 storage_buffer_count = shader_layout->storage_buffer_count;
    Uint32 entry_count = 0;

    if (sampler_count == 0 && storage_texture_count == 0 && storage_buffer_count == 0) {
        return true;
    }
    if (!*bind_group_dirty) {
        WEBGPU_RecordBindGroupCleanApply(command_buffer->renderer, instrumentation_path);
        return true;
    }
    if (group_index >= pipeline->bind_group_layout_count || !pipeline->bind_group_layouts[group_index]) {
        WEBGPU_SetStringError("graphics pipeline is missing the required resource bind group layout");
        return false;
    }
    layout = pipeline->bind_group_layouts[group_index];

    {
        static const WebGPUSampledTextureBindGroupEntryErrors sampled_texture_errors = {
            .missing_binding = "missing sampled texture binding for graphics pipeline",
            .invalid_generation = "invalid sampled texture binding generation",
            .invalid_usage = "invalid sampled texture binding usage",
            .type_mismatch = "sampled texture type does not match shader resource layout",
            .sample_count_mismatch = "sampled texture sample count does not match shader resource layout",
            .sample_type_mismatch = "sampled texture sample type does not match shader resource layout",
            .missing_sampler = "missing sampler binding for graphics pipeline",
            .sampler_type_mismatch = "sampler binding type does not match shader resource layout"
        };

        if (!WEBGPU_AppendSampledTextureBindGroupEntries(
                command_buffer,
                entries,
                &entry_count,
                sampled_texture_bindings,
                shader_layout->samplers,
                sampler_count,
                true,
                &sampled_texture_errors)) {
            return false;
        }
    }

    if (!WEBGPU_AppendGraphicsStorageTextureBindGroupEntries(
            entries,
            &entry_count,
            sampler_count * 2,
            storage_texture_bindings,
            shader_layout->storage_textures,
            storage_texture_count)) {
        return false;
    }

    if (!WEBGPU_AppendStorageBufferBindGroupEntries(
            entries,
            &entry_count,
            sampler_count * 2 + storage_texture_count,
            storage_buffer_bindings,
            storage_buffer_count,
            "missing storage buffer binding for graphics pipeline")) {
        return false;
    }

    return WEBGPU_ApplyPreparedResourceBindGroup(
        command_buffer,
        group_index,
        instrumentation_path,
        layout,
        entries,
        entry_count,
        sampled_texture_bindings,
        sampler_count,
        storage_texture_bindings,
        storage_texture_count,
        storage_buffer_bindings,
        storage_buffer_count,
        bind_group_dirty,
        "CreateBindGroup for resources",
        "CreateBindGroup failed for resources",
        "failed to track resource bind group",
        false);
}

static bool WEBGPU_PrepareUniformBindGroup(
    WebGPUCommandBuffer *command_buffer,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    WGPUBindGroupLayout layout,
    Uint32 uniform_count,
    WGPUBuffer *buffers,
    Uint64 *buffer_offsets,
    Uint64 *buffer_sizes,
    WGPUBindGroup *current_bind_group,
    bool *bind_group_dirty,
    bool *bind_group_offsets_dirty,
    Uint32 *dynamic_offsets,
    WGPUBindGroup *bind_group_out,
    bool *set_bind_group_out,
    const char *missing_layout_error,
    const char *missing_uniform_error,
    const char *offset_range_error,
    const char *offset_alignment_error,
    const char *create_scope_name,
    const char *create_failed_error,
    const char *track_failed_error)
{
    WGPUBindGroupEntry entries[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WGPUBindGroupDescriptor bind_group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    WGPUBindGroup bind_group;
    bool create_bind_group;
    bool scoped_error;

    *bind_group_out = NULL;
    *set_bind_group_out = false;
    if (uniform_count == 0) {
        return true;
    }
    create_bind_group = *bind_group_dirty || !*current_bind_group;
    *set_bind_group_out = create_bind_group || *bind_group_offsets_dirty;
    if (!*set_bind_group_out) {
        WEBGPU_RecordBindGroupCleanApply(command_buffer->renderer, instrumentation_path);
        return true;
    }
    if (!layout) {
        WEBGPU_SetStringError(missing_layout_error);
        return false;
    }

    for (Uint32 i = 0; i < uniform_count; i += 1) {
        if (!buffers[i] || buffer_sizes[i] == 0) {
            WEBGPU_SetStringError(missing_uniform_error);
            return false;
        }
        if (buffer_offsets[i] > SDL_MAX_UINT32) {
            WEBGPU_SetStringError(offset_range_error);
            return false;
        }
        if ((buffer_offsets[i] % command_buffer->renderer->limits.minUniformBufferOffsetAlignment) != 0) {
            WEBGPU_SetStringError(offset_alignment_error);
            return false;
        }

        dynamic_offsets[i] = (Uint32)buffer_offsets[i];

        if (create_bind_group) {
            entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
            entries[i].binding = i;
            entries[i].buffer = buffers[i];
            entries[i].offset = 0;
            entries[i].size = buffer_sizes[i];
        }
    }

    if (create_bind_group) {
        bind_group_desc.layout = layout;
        bind_group_desc.entryCount = uniform_count;
        bind_group_desc.entries = entries;
        scoped_error = WEBGPU_CommandBufferCanWaitForErrorScope(command_buffer);
        if (scoped_error) {
            bind_group = WEBGPU_CreateBindGroup(command_buffer->renderer, &bind_group_desc, create_scope_name);
        } else {
            bind_group = wgpuDeviceCreateBindGroup(command_buffer->renderer->device, &bind_group_desc);
        }
        if (!bind_group) {
            if (!scoped_error) {
                WEBGPU_SetStringError(create_failed_error);
            }
            return false;
        }
        WEBGPU_RecordBindGroupCreate(command_buffer->renderer, instrumentation_path);
        if (!WEBGPU_TrackCommandBufferBindGroup(command_buffer, bind_group)) {
            wgpuBindGroupRelease(bind_group);
            WEBGPU_SetStringError(track_failed_error);
            return false;
        }
        *current_bind_group = bind_group;
        wgpuBindGroupRelease(bind_group);
    }

    *bind_group_out = *current_bind_group;
    return true;
}

static bool WEBGPU_ApplyUniformBindGroup(
    WebGPUCommandBuffer *command_buffer,
    Uint32 group_index,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    const WebGPUShaderResourceLayout *shader_layout,
    WGPUBuffer *buffers,
    Uint64 *buffer_offsets,
    Uint64 *buffer_sizes,
    WGPUBindGroup *current_bind_group,
    bool *bind_group_dirty,
    bool *bind_group_offsets_dirty)
{
    Uint32 dynamic_offsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout layout = group_index < pipeline->bind_group_layout_count ? pipeline->bind_group_layouts[group_index] : NULL;
    Uint32 uniform_count = shader_layout->uniform_buffer_count;
    bool set_bind_group;

    if (!WEBGPU_PrepareUniformBindGroup(
            command_buffer,
            instrumentation_path,
            layout,
            uniform_count,
            buffers,
            buffer_offsets,
            buffer_sizes,
            current_bind_group,
            bind_group_dirty,
            bind_group_offsets_dirty,
            dynamic_offsets,
            &bind_group,
            &set_bind_group,
            "graphics pipeline is missing the required uniform bind group layout",
            "missing uniform data for graphics pipeline",
            "graphics uniform offset exceeds WebGPU dynamic offset range",
            "graphics uniform offset is not aligned to minUniformBufferOffsetAlignment",
            "CreateBindGroup for uniforms",
            "CreateBindGroup failed for uniforms",
            "failed to track uniform bind group")) {
        return false;
    }
    if (!set_bind_group) {
        return true;
    }

    wgpuRenderPassEncoderSetBindGroup(command_buffer->render_pass, group_index, bind_group, uniform_count, dynamic_offsets);
    WEBGPU_RecordBindGroupSet(command_buffer->renderer, instrumentation_path);
    *bind_group_dirty = false;
    *bind_group_offsets_dirty = false;
    return true;
}

static bool WEBGPU_PrepareComputeResourceBindGroupApply(
    WebGPUCommandBuffer *command_buffer,
    Uint32 group_index,
    WebGPUBindGroupInstrumentationPath instrumentation_path,
    Uint32 sampler_count,
    Uint32 storage_texture_count,
    Uint32 storage_buffer_count,
    bool bind_group_dirty,
    const char *missing_layout_error,
    WGPUBindGroupLayout *layout,
    bool *apply_bind_group)
{
    WebGPUComputePipeline *pipeline = command_buffer->current_compute_pipeline;

    *layout = NULL;
    *apply_bind_group = false;

    if (sampler_count == 0 && storage_texture_count == 0 && storage_buffer_count == 0) {
        return true;
    }
    if (!bind_group_dirty) {
        WEBGPU_RecordBindGroupCleanApply(command_buffer->renderer, instrumentation_path);
        return true;
    }
    if (group_index >= pipeline->bind_group_layout_count ||
        !pipeline->bind_group_layouts[group_index]) {
        WEBGPU_SetStringError(missing_layout_error);
        return false;
    }

    *layout = pipeline->bind_group_layouts[group_index];
    *apply_bind_group = true;
    return true;
}

static bool WEBGPU_ApplyComputeReadOnlyBindGroup(WebGPUCommandBuffer *command_buffer)
{
    WGPUBindGroupEntry entries[MAX_TEXTURE_SAMPLERS_PER_STAGE * 2 + MAX_STORAGE_TEXTURES_PER_STAGE + MAX_STORAGE_BUFFERS_PER_STAGE];
    const WebGPUComputeResourceLayout *compute_layout = &command_buffer->current_compute_pipeline->resources;
    WGPUBindGroupLayout layout;
    Uint32 sampler_count = compute_layout->sampler_count;
    Uint32 storage_texture_count = compute_layout->readonly_storage_texture_count;
    Uint32 storage_buffer_count = compute_layout->readonly_storage_buffer_count;
    Uint32 entry_count = 0;
    bool apply_bind_group;

    if (!WEBGPU_PrepareComputeResourceBindGroupApply(
            command_buffer,
            WEBGPU_COMPUTE_READONLY_GROUP,
            WEBGPU_BIND_GROUP_PATH_COMPUTE_READONLY,
            sampler_count,
            storage_texture_count,
            storage_buffer_count,
            command_buffer->compute_readonly_bind_group_dirty,
            "compute pipeline is missing the required read-only bind group layout",
            &layout,
            &apply_bind_group)) {
        return false;
    }
    if (!apply_bind_group) {
        return true;
    }

    {
        static const WebGPUSampledTextureBindGroupEntryErrors sampled_texture_errors = {
            .missing_binding = "missing sampled texture binding for compute pipeline",
            .invalid_generation = "invalid sampled texture binding generation",
            .invalid_usage = "invalid sampled texture binding usage",
            .type_mismatch = "compute sampled texture type does not match shader resource layout",
            .sample_count_mismatch = "compute sampled texture sample count does not match shader resource layout",
            .sample_type_mismatch = "compute sampled texture sample type does not match shader resource layout",
            .missing_sampler = "missing sampler binding for compute pipeline",
            .sampler_type_mismatch = "compute sampler binding type does not match shader resource layout"
        };

        if (!WEBGPU_AppendSampledTextureBindGroupEntries(
                command_buffer,
                entries,
                &entry_count,
                command_buffer->compute_sampler_bindings,
                compute_layout->samplers,
                sampler_count,
                false,
                &sampled_texture_errors)) {
            return false;
        }
    }

    if (!WEBGPU_AppendComputeStorageTextureBindGroupEntries(
            entries,
            &entry_count,
            sampler_count * 2,
            command_buffer->compute_readonly_storage_texture_bindings,
            compute_layout->readonly_storage_textures,
            storage_texture_count,
            WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READ,
            "missing read-only storage texture binding for compute pipeline",
            "invalid read-only storage texture binding generation",
            "invalid read-only storage texture binding usage",
            "compute read-only")) {
        return false;
    }

    if (!WEBGPU_AppendStorageBufferBindGroupEntries(
            entries,
            &entry_count,
            sampler_count * 2 + storage_texture_count,
            command_buffer->compute_readonly_storage_buffer_bindings,
            storage_buffer_count,
            "missing read-only storage buffer binding for compute pipeline")) {
        return false;
    }

    return WEBGPU_ApplyPreparedResourceBindGroup(
        command_buffer,
        WEBGPU_COMPUTE_READONLY_GROUP,
        WEBGPU_BIND_GROUP_PATH_COMPUTE_READONLY,
        layout,
        entries,
        entry_count,
        command_buffer->compute_sampler_bindings,
        sampler_count,
        command_buffer->compute_readonly_storage_texture_bindings,
        storage_texture_count,
        command_buffer->compute_readonly_storage_buffer_bindings,
        storage_buffer_count,
        &command_buffer->compute_readonly_bind_group_dirty,
        "CreateBindGroup for compute read-only resources",
        "CreateBindGroup failed for compute read-only resources",
        "failed to track compute read-only bind group",
        true);
}

static bool WEBGPU_ApplyComputeReadWriteBindGroup(WebGPUCommandBuffer *command_buffer)
{
    WGPUBindGroupEntry entries[MAX_COMPUTE_WRITE_TEXTURES + MAX_COMPUTE_WRITE_BUFFERS];
    const WebGPUComputeResourceLayout *compute_layout = &command_buffer->current_compute_pipeline->resources;
    WGPUBindGroupLayout layout;
    Uint32 storage_texture_count = compute_layout->readwrite_storage_texture_count;
    Uint32 storage_buffer_count = compute_layout->readwrite_storage_buffer_count;
    Uint32 entry_count = 0;
    bool apply_bind_group;

    if (!WEBGPU_PrepareComputeResourceBindGroupApply(
            command_buffer,
            WEBGPU_COMPUTE_READWRITE_GROUP,
            WEBGPU_BIND_GROUP_PATH_COMPUTE_READWRITE,
            0,
            storage_texture_count,
            storage_buffer_count,
            command_buffer->compute_readwrite_bind_group_dirty,
            "compute pipeline is missing the required read-write bind group layout",
            &layout,
            &apply_bind_group)) {
        return false;
    }
    if (!apply_bind_group) {
        return true;
    }

    if (!WEBGPU_AppendComputeStorageTextureBindGroupEntries(
            entries,
            &entry_count,
            0,
            command_buffer->compute_readwrite_storage_texture_bindings,
            compute_layout->readwrite_storage_textures,
            storage_texture_count,
            WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READWRITE,
            "missing read-write storage texture binding for compute pipeline",
            "invalid read-write storage texture binding generation",
            "invalid read-write storage texture binding usage",
            "compute read-write")) {
        return false;
    }

    if (!WEBGPU_AppendStorageBufferBindGroupEntries(
            entries,
            &entry_count,
            storage_texture_count,
            command_buffer->compute_readwrite_storage_buffer_bindings,
            storage_buffer_count,
            "missing read-write storage buffer binding for compute pipeline")) {
        return false;
    }

    return WEBGPU_ApplyPreparedResourceBindGroup(
        command_buffer,
        WEBGPU_COMPUTE_READWRITE_GROUP,
        WEBGPU_BIND_GROUP_PATH_COMPUTE_READWRITE,
        layout,
        entries,
        entry_count,
        NULL,
        0,
        command_buffer->compute_readwrite_storage_texture_bindings,
        storage_texture_count,
        command_buffer->compute_readwrite_storage_buffer_bindings,
        storage_buffer_count,
        &command_buffer->compute_readwrite_bind_group_dirty,
        "CreateBindGroup for compute read-write storage resources",
        "CreateBindGroup failed for compute read-write storage resources",
        "failed to track compute read-write bind group",
        true);
}

static bool WEBGPU_ApplyComputeUniformBindGroup(WebGPUCommandBuffer *command_buffer)
{
    Uint32 dynamic_offsets[MAX_UNIFORM_BUFFERS_PER_STAGE];
    WebGPUComputePipeline *pipeline = command_buffer->current_compute_pipeline;
    WGPUBindGroup bind_group;
    WGPUBindGroupLayout layout = WEBGPU_COMPUTE_UNIFORM_GROUP < pipeline->bind_group_layout_count ? pipeline->bind_group_layouts[WEBGPU_COMPUTE_UNIFORM_GROUP] : NULL;
    Uint32 uniform_count = pipeline->header.numUniformBuffers;
    bool set_bind_group;

    if (!WEBGPU_PrepareUniformBindGroup(
            command_buffer,
            WEBGPU_BIND_GROUP_PATH_COMPUTE_UNIFORM,
            layout,
            uniform_count,
            command_buffer->compute_uniform_buffers,
            command_buffer->compute_uniform_buffer_offsets,
            command_buffer->compute_uniform_buffer_sizes,
            &command_buffer->compute_uniform_bind_group,
            &command_buffer->compute_uniform_bind_group_dirty,
            &command_buffer->compute_uniform_bind_group_offsets_dirty,
            dynamic_offsets,
            &bind_group,
            &set_bind_group,
            "compute pipeline is missing the required uniform bind group layout",
            "missing uniform data for compute pipeline",
            "compute uniform offset exceeds WebGPU dynamic offset range",
            "compute uniform offset is not aligned to minUniformBufferOffsetAlignment",
            "CreateBindGroup for compute uniforms",
            "CreateBindGroup failed for compute uniforms",
            "failed to track compute uniform bind group")) {
        return false;
    }
    if (!set_bind_group) {
        return true;
    }

    wgpuComputePassEncoderSetBindGroup(command_buffer->compute_pass, WEBGPU_COMPUTE_UNIFORM_GROUP, bind_group, uniform_count, dynamic_offsets);
    WEBGPU_RecordBindGroupSet(command_buffer->renderer, WEBGPU_BIND_GROUP_PATH_COMPUTE_UNIFORM);
    command_buffer->compute_uniform_bind_group_dirty = false;
    command_buffer->compute_uniform_bind_group_offsets_dirty = false;
    return true;
}

static bool WEBGPU_ApplyPipelineEmptyBindGroups(
    WebGPUCommandBuffer *command_buffer,
    const WGPUBindGroup *empty_bind_groups,
    Uint32 bind_group_layout_count,
    WebGPUBindGroupInstrumentationPath path,
    const char *track_error,
    bool compute_pass)
{
    /* Dense WebGPU pipeline layouts require matching bind groups even for empty slots. */
    for (Uint32 i = 0; i < bind_group_layout_count; i += 1) {
        WGPUBindGroup bind_group = empty_bind_groups[i];

        if (!bind_group) {
            continue;
        }
        if (!WEBGPU_TrackCommandBufferBindGroup(command_buffer, bind_group)) {
            WEBGPU_SetStringError(track_error);
            return false;
        }
        if (compute_pass) {
            wgpuComputePassEncoderSetBindGroup(command_buffer->compute_pass, i, bind_group, 0, NULL);
        } else {
            wgpuRenderPassEncoderSetBindGroup(command_buffer->render_pass, i, bind_group, 0, NULL);
        }
        WEBGPU_RecordBindGroupSet(command_buffer->renderer, path);
    }

    return true;
}

static bool WEBGPU_ApplyComputeEmptyBindGroups(WebGPUCommandBuffer *command_buffer)
{
    WebGPUComputePipeline *pipeline = command_buffer->current_compute_pipeline;

    return WEBGPU_ApplyPipelineEmptyBindGroups(
        command_buffer,
        pipeline->empty_bind_groups,
        pipeline->bind_group_layout_count,
        WEBGPU_BIND_GROUP_PATH_COMPUTE_EMPTY,
        "failed to track empty compute bind group",
        true);
}

static bool WEBGPU_ApplyComputeBindGroups(WebGPUCommandBuffer *command_buffer)
{
    if (!command_buffer->current_compute_pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "no compute pipeline is bound");
        return false;
    }
    if (!WEBGPU_ApplyComputeEmptyBindGroups(command_buffer)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyComputeReadOnlyBindGroup(command_buffer)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyComputeReadWriteBindGroup(command_buffer)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyComputeUniformBindGroup(command_buffer)) {
        command_buffer->failed = true;
        return false;
    }

    return true;
}

static bool WEBGPU_ApplyGraphicsEmptyBindGroups(WebGPUCommandBuffer *command_buffer)
{
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;

    return WEBGPU_ApplyPipelineEmptyBindGroups(
        command_buffer,
        pipeline->empty_bind_groups,
        pipeline->bind_group_layout_count,
        WEBGPU_BIND_GROUP_PATH_GRAPHICS_EMPTY,
        "failed to track empty graphics bind group",
        false);
}

static bool WEBGPU_ApplyGraphicsBindGroups(WebGPUCommandBuffer *command_buffer)
{
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;

    if (!pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "no graphics pipeline is bound");
        return false;
    }
    if (!WEBGPU_ApplyGraphicsEmptyBindGroups(command_buffer)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyResourceBindGroup(
            command_buffer,
            WEBGPU_VERTEX_RESOURCE_GROUP,
            WEBGPU_BIND_GROUP_PATH_GRAPHICS_VERTEX_RESOURCE,
            &pipeline->vertex_resources,
            command_buffer->vertex_sampler_bindings,
            command_buffer->vertex_storage_texture_bindings,
            command_buffer->vertex_storage_buffer_bindings,
            &command_buffer->vertex_resource_bind_group_dirty)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyUniformBindGroup(
            command_buffer,
            WEBGPU_VERTEX_UNIFORM_GROUP,
            WEBGPU_BIND_GROUP_PATH_GRAPHICS_VERTEX_UNIFORM,
            &pipeline->vertex_resources,
            command_buffer->vertex_uniform_buffers,
            command_buffer->vertex_uniform_buffer_offsets,
            command_buffer->vertex_uniform_buffer_sizes,
            &command_buffer->vertex_uniform_bind_group,
            &command_buffer->vertex_uniform_bind_group_dirty,
            &command_buffer->vertex_uniform_bind_group_offsets_dirty)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyResourceBindGroup(
            command_buffer,
            WEBGPU_FRAGMENT_RESOURCE_GROUP,
            WEBGPU_BIND_GROUP_PATH_GRAPHICS_FRAGMENT_RESOURCE,
            &pipeline->fragment_resources,
            command_buffer->fragment_sampler_bindings,
            command_buffer->fragment_storage_texture_bindings,
            command_buffer->fragment_storage_buffer_bindings,
            &command_buffer->fragment_resource_bind_group_dirty)) {
        command_buffer->failed = true;
        return false;
    }
    if (!WEBGPU_ApplyUniformBindGroup(
            command_buffer,
            WEBGPU_FRAGMENT_UNIFORM_GROUP,
            WEBGPU_BIND_GROUP_PATH_GRAPHICS_FRAGMENT_UNIFORM,
            &pipeline->fragment_resources,
            command_buffer->fragment_uniform_buffers,
            command_buffer->fragment_uniform_buffer_offsets,
            command_buffer->fragment_uniform_buffer_sizes,
            &command_buffer->fragment_uniform_bind_group,
            &command_buffer->fragment_uniform_bind_group_dirty,
            &command_buffer->fragment_uniform_bind_group_offsets_dirty)) {
        command_buffer->failed = true;
        return false;
    }

    return true;
}

static bool WEBGPU_ValidateVertexBufferDrawRange(WebGPUCommandBuffer *command_buffer, Uint32 slot, Uint64 stride_count)
{
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;
    Uint64 last_stride = pipeline->vertex_buffer_last_strides[slot];
    Uint64 stride = pipeline->vertex_buffer_strides[slot];
    Uint64 required_size;

    if (stride_count == 0) {
        return true;
    }

    if (stride != 0 && stride_count - 1 > (SDL_MAX_UINT64 - last_stride) / stride) {
        required_size = SDL_MAX_UINT64;
    } else {
        required_size = ((stride_count - 1) * stride) + last_stride;
    }

    if (required_size > command_buffer->vertex_buffer_sizes[slot]) {
        SDL_SetError(
            "WebGPU backend: vertex buffer binding for slot %u is too small for draw (need %" SDL_PRIu64 " bytes, have %" SDL_PRIu64 " bytes)",
            slot,
            required_size,
            command_buffer->vertex_buffer_sizes[slot]);
        command_buffer->failed = true;
        return false;
    }

    return true;
}

static bool WEBGPU_ValidateDrawBindings(WebGPUCommandBuffer *command_buffer, bool indexed, Uint32 element_count, Uint32 num_instances, Uint32 first_element, Uint32 first_instance)
{
    WebGPUGraphicsPipeline *pipeline = command_buffer->current_graphics_pipeline;
    Uint32 missing_vertex_buffer_mask;

    if (!pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "no graphics pipeline is bound");
        return false;
    }

    missing_vertex_buffer_mask = pipeline->required_vertex_buffer_mask & ~command_buffer->bound_vertex_buffer_mask;
    if (missing_vertex_buffer_mask) {
        Uint32 slot = 0;

        while ((missing_vertex_buffer_mask & (1u << slot)) == 0) {
            slot += 1;
        }

        SDL_SetError("WebGPU backend: missing vertex buffer binding for slot %u", slot);
        command_buffer->failed = true;
        return false;
    }

    if (indexed && !command_buffer->index_buffer_bound) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing index buffer binding for indexed draw");
        return false;
    }

    if (indexed) {
        Uint64 required_index_count = (Uint64)first_element + element_count;
        Uint64 required_index_size = required_index_count * command_buffer->index_element_size;

        if (required_index_size > command_buffer->index_buffer_size) {
            SDL_SetError(
                "WebGPU backend: index buffer binding is too small for indexed draw (need %" SDL_PRIu64 " bytes, have %" SDL_PRIu64 " bytes)",
                required_index_size,
                command_buffer->index_buffer_size);
            command_buffer->failed = true;
            return false;
        }
    }

    for (Uint32 slot = 0; slot < MAX_VERTEX_BUFFERS; slot += 1) {
        Uint64 stride_count;

        if ((pipeline->required_vertex_buffer_mask & (1u << slot)) == 0) {
            continue;
        }

        if (pipeline->vertex_buffer_instance_step[slot]) {
            stride_count = (Uint64)first_instance + num_instances;
        } else if (!indexed) {
            stride_count = (Uint64)first_element + element_count;
        } else {
            continue;
        }

        if (!WEBGPU_ValidateVertexBufferDrawRange(command_buffer, slot, stride_count)) {
            return false;
        }
    }

    return true;
}

static bool WEBGPU_AllocateUniformSlice(WebGPUCommandBuffer *command_buffer, Uint32 size, WGPUBuffer *buffer, Uint64 *offset)
{
    const Uint64 allocation_size = WEBGPU_AlignUp64(size, command_buffer->renderer->limits.minUniformBufferOffsetAlignment);

    if (!command_buffer->uniform_buffer_page ||
        command_buffer->uniform_buffer_page_offset + allocation_size > command_buffer->uniform_buffer_page_size) {
        WGPUBufferDescriptor buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
        WGPUBuffer page;
        Uint64 page_size = WEBGPU_UNIFORM_BUFFER_PAGE_SIZE;

        if (allocation_size > page_size) {
            page_size = allocation_size;
        }

        buffer_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        buffer_desc.size = page_size;
        page = wgpuDeviceCreateBuffer(command_buffer->renderer->device, &buffer_desc);
        if (!page) {
            WEBGPU_SetStringError("CreateBuffer failed for uniform page");
            return false;
        }
        if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, page)) {
            wgpuBufferRelease(page);
            WEBGPU_SetStringError("failed to track uniform page");
            return false;
        }

        command_buffer->uniform_buffer_page = page;
        command_buffer->uniform_buffer_page_offset = 0;
        command_buffer->uniform_buffer_page_size = page_size;
        wgpuBufferRelease(page);
    }

    *buffer = command_buffer->uniform_buffer_page;
    *offset = command_buffer->uniform_buffer_page_offset;
    command_buffer->uniform_buffer_page_offset += allocation_size;
    return true;
}

static void WEBGPU_BindVertexSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindVertexSamplers")) {
        return;
    }
    WEBGPU_BindSampledTextures(
        command_buffer,
        firstSlot,
        textureSamplerBindings,
        numBindings,
        command_buffer->vertex_sampler_bindings,
        &command_buffer->vertex_resource_bind_group_dirty,
        true,
        true,
        &WEBGPU_GraphicsSampledTextureBindErrors);
}


static void WEBGPU_BindVertexStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindVertexStorageTextures")) {
        return;
    }
    WEBGPU_BindStorageTextureBindings(
        command_buffer,
        firstSlot,
        storageTextures,
        numBindings,
        command_buffer->vertex_storage_texture_bindings,
        &command_buffer->vertex_resource_bind_group_dirty,
        SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
        false,
        false,
        true,
        &WEBGPU_GraphicsStorageTextureBindErrors);
}


static void WEBGPU_BindVertexStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindVertexStorageBuffers")) {
        return;
    }
    WEBGPU_BindStorageBufferBindings(
        command_buffer,
        firstSlot,
        storageBuffers,
        numBindings,
        command_buffer->vertex_storage_buffer_bindings,
        &command_buffer->vertex_resource_bind_group_dirty,
        SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        WEBGPU_BUFFER_BINDING_USAGE_GRAPHICS_STORAGE_READ,
        true,
        &WEBGPU_GraphicsStorageBufferBindErrors);
}


static void WEBGPU_BindFragmentSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindFragmentSamplers")) {
        return;
    }
    WEBGPU_BindSampledTextures(
        command_buffer,
        firstSlot,
        textureSamplerBindings,
        numBindings,
        command_buffer->fragment_sampler_bindings,
        &command_buffer->fragment_resource_bind_group_dirty,
        true,
        true,
        &WEBGPU_GraphicsSampledTextureBindErrors);
}


static void WEBGPU_BindFragmentStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindFragmentStorageTextures")) {
        return;
    }
    WEBGPU_BindStorageTextureBindings(
        command_buffer,
        firstSlot,
        storageTextures,
        numBindings,
        command_buffer->fragment_storage_texture_bindings,
        &command_buffer->fragment_resource_bind_group_dirty,
        SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ,
        false,
        false,
        true,
        &WEBGPU_GraphicsStorageTextureBindErrors);
}


static void WEBGPU_BindFragmentStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "BindFragmentStorageBuffers")) {
        return;
    }
    WEBGPU_BindStorageBufferBindings(
        command_buffer,
        firstSlot,
        storageBuffers,
        numBindings,
        command_buffer->fragment_storage_buffer_bindings,
        &command_buffer->fragment_resource_bind_group_dirty,
        SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        WEBGPU_BUFFER_BINDING_USAGE_GRAPHICS_STORAGE_READ,
        true,
        &WEBGPU_GraphicsStorageBufferBindErrors);
}


static void WEBGPU_PushUniformData(
    WebGPUCommandBuffer *command_buffer,
    Uint32 slot_index,
    const void *data,
    Uint32 length,
    WGPUBuffer *buffers,
    Uint64 *buffer_offsets,
    Uint64 *buffer_sizes,
    bool *bind_group_dirty,
    bool *bind_group_offsets_dirty)
{
    WGPUBuffer buffer;
    Uint64 buffer_offset;
    Uint32 buffer_size;
    Uint8 *write_data = NULL;
    bool binding_changed;

    if (command_buffer->failed) {
        return;
    }
    if (slot_index >= MAX_UNIFORM_BUFFERS_PER_STAGE) {
        WEBGPU_FailCommandBuffer(command_buffer, "uniform slot index exceeds MAX_UNIFORM_BUFFERS_PER_STAGE");
        return;
    }
    if (!data) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid uniform data");
        return;
    }
    if (length == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "uniform data length must be greater than zero");
        return;
    }
    if (length > UNIFORM_BUFFER_SIZE) {
        WEBGPU_FailCommandBuffer(command_buffer, "uniform data length exceeds UNIFORM_BUFFER_SIZE");
        return;
    }

    buffer_size = WEBGPU_AlignUp(length, 16);
    buffer_size = SDL_max(buffer_size, 16u);

    if (!WEBGPU_AllocateUniformSlice(command_buffer, buffer_size, &buffer, &buffer_offset)) {
        command_buffer->failed = true;
        return;
    }
    binding_changed = buffers[slot_index] != buffer || buffer_sizes[slot_index] != buffer_size;

    if (buffer_size == length) {
        wgpuQueueWriteBuffer(command_buffer->renderer->queue, buffer, buffer_offset, data, length);
    } else {
        write_data = SDL_stack_alloc(Uint8, buffer_size);
        if (!write_data) {
            WEBGPU_FailCommandBuffer(command_buffer, "failed to allocate padded uniform data");
            return;
        }
        SDL_memset(write_data, 0, buffer_size);
        SDL_memcpy(write_data, data, length);
        wgpuQueueWriteBuffer(command_buffer->renderer->queue, buffer, buffer_offset, write_data, buffer_size);
        SDL_stack_free(write_data);
    }

    buffers[slot_index] = buffer;
    buffer_offsets[slot_index] = buffer_offset;
    buffer_sizes[slot_index] = buffer_size;
    *bind_group_offsets_dirty = true;
    if (binding_changed) {
        *bind_group_dirty = true;
        WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_UNIFORM);
    }
}

static void WEBGPU_PushVertexUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WEBGPU_PushUniformData(
        command_buffer,
        slotIndex,
        data,
        length,
        command_buffer->vertex_uniform_buffers,
        command_buffer->vertex_uniform_buffer_offsets,
        command_buffer->vertex_uniform_buffer_sizes,
        &command_buffer->vertex_uniform_bind_group_dirty,
        &command_buffer->vertex_uniform_bind_group_offsets_dirty);
}

static void WEBGPU_PushFragmentUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WEBGPU_PushUniformData(
        command_buffer,
        slotIndex,
        data,
        length,
        command_buffer->fragment_uniform_buffers,
        command_buffer->fragment_uniform_buffer_offsets,
        command_buffer->fragment_uniform_buffer_sizes,
        &command_buffer->fragment_uniform_bind_group_dirty,
        &command_buffer->fragment_uniform_bind_group_offsets_dirty);
}

static void WEBGPU_DrawIndexedPrimitives(SDL_GPUCommandBuffer *commandBuffer, Uint32 numIndices, Uint32 numInstances, Uint32 firstIndex, Sint32 vertexOffset, Uint32 firstInstance)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "DrawIndexedPrimitives")) {
        return;
    }
    if (!WEBGPU_ValidateDrawBindings(command_buffer, true, numIndices, numInstances, firstIndex, firstInstance)) {
        return;
    }
    if (!WEBGPU_ApplyGraphicsBindGroups(command_buffer)) {
        return;
    }
    wgpuRenderPassEncoderDrawIndexed(command_buffer->render_pass, numIndices, numInstances, firstIndex, vertexOffset, firstInstance);
}

static void WEBGPU_DrawPrimitives(SDL_GPUCommandBuffer *commandBuffer, Uint32 numVertices, Uint32 numInstances, Uint32 firstVertex, Uint32 firstInstance)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "DrawPrimitives")) {
        return;
    }
    if (!WEBGPU_ValidateDrawBindings(command_buffer, false, numVertices, numInstances, firstVertex, firstInstance)) {
        return;
    }
    if (!WEBGPU_ApplyGraphicsBindGroups(command_buffer)) {
        return;
    }
    wgpuRenderPassEncoderDraw(command_buffer->render_pass, numVertices, numInstances, firstVertex, firstInstance);
}

static void WEBGPU_DrawPrimitivesIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset, Uint32 drawCount)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *webgpu_buffer = (WebGPUBuffer *)buffer;
    const Uint32 pitch = sizeof(SDL_GPUIndirectDrawCommand);

    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "DrawPrimitivesIndirect")) {
        return;
    }
    if (!WEBGPU_ValidateDrawBindings(command_buffer, false, 0, 0, 0, 0)) {
        return;
    }
    if (drawCount == 0) {
        return;
    }
    if (!webgpu_buffer || !webgpu_buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid indirect draw buffer");
        return;
    }
    if (webgpu_buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect draw buffer has been released");
        return;
    }
    if ((webgpu_buffer->usage & SDL_GPU_BUFFERUSAGE_INDIRECT) == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect draw buffer was not created with SDL_GPU_BUFFERUSAGE_INDIRECT");
        return;
    }
    if ((offset % 4) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect draw offset must be 4-byte aligned");
        return;
    }
    if ((Uint64)offset + ((Uint64)drawCount * pitch) > webgpu_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect draw range exceeds buffer size");
        return;
    }
    if (!WEBGPU_ApplyGraphicsBindGroups(command_buffer)) {
        return;
    }
    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, webgpu_buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track indirect draw buffer");
        return;
    }
    for (Uint32 i = 0; i < drawCount; i += 1) {
        wgpuRenderPassEncoderDrawIndirect(command_buffer->render_pass, webgpu_buffer->buffer, offset + (pitch * i));
    }
}

static void WEBGPU_DrawIndexedPrimitivesIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset, Uint32 drawCount)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *webgpu_buffer = (WebGPUBuffer *)buffer;
    const Uint32 pitch = sizeof(SDL_GPUIndexedIndirectDrawCommand);

    if (command_buffer->failed || WEBGPU_FailIfRenderPassInactive(command_buffer, "DrawIndexedPrimitivesIndirect")) {
        return;
    }
    if (!WEBGPU_ValidateDrawBindings(command_buffer, true, 0, 0, 0, 0)) {
        return;
    }
    if (drawCount == 0) {
        return;
    }
    if (!webgpu_buffer || !webgpu_buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid indexed indirect draw buffer");
        return;
    }
    if (webgpu_buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "indexed indirect draw buffer has been released");
        return;
    }
    if ((webgpu_buffer->usage & SDL_GPU_BUFFERUSAGE_INDIRECT) == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indexed indirect draw buffer was not created with SDL_GPU_BUFFERUSAGE_INDIRECT");
        return;
    }
    if ((offset % 4) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indexed indirect draw offset must be 4-byte aligned");
        return;
    }
    if ((Uint64)offset + ((Uint64)drawCount * pitch) > webgpu_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "indexed indirect draw range exceeds buffer size");
        return;
    }
    if (!WEBGPU_ApplyGraphicsBindGroups(command_buffer)) {
        return;
    }
    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, webgpu_buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track indexed indirect draw buffer");
        return;
    }
    for (Uint32 i = 0; i < drawCount; i += 1) {
        wgpuRenderPassEncoderDrawIndexedIndirect(command_buffer->render_pass, webgpu_buffer->buffer, offset + (pitch * i));
    }
}

static void WEBGPU_EndRenderPass(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (!command_buffer->render_pass) {
        if (!command_buffer->failed) {
            WEBGPU_FailCommandBuffer(command_buffer, "EndRenderPass requires an active render pass");
        }
        return;
    }
    wgpuRenderPassEncoderEnd(command_buffer->render_pass);
    wgpuRenderPassEncoderRelease(command_buffer->render_pass);
    command_buffer->render_pass = NULL;

    for (Uint32 i = 0; i < command_buffer->render_pass_3d_resolve_copy_count; i += 1) {
        const WebGPU3DResolveCopy *resolve_copy = &command_buffer->render_pass_3d_resolve_copies[i];
        WGPUTexelCopyTextureInfo source_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        WGPUTexelCopyTextureInfo destination_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        WGPUExtent3D copy_size = WGPU_EXTENT_3D_INIT;

        source_copy.texture = resolve_copy->source_2d_texture;
        source_copy.mipLevel = 0;
        source_copy.origin.x = 0;
        source_copy.origin.y = 0;
        source_copy.origin.z = 0;
        source_copy.aspect = WGPUTextureAspect_All;

        destination_copy.texture = resolve_copy->destination_3d_texture;
        destination_copy.mipLevel = resolve_copy->destination_mip_level;
        destination_copy.origin.x = 0;
        destination_copy.origin.y = 0;
        destination_copy.origin.z = resolve_copy->destination_depth_plane;
        destination_copy.aspect = WGPUTextureAspect_All;

        copy_size.width = resolve_copy->width;
        copy_size.height = resolve_copy->height;
        copy_size.depthOrArrayLayers = 1;

        wgpuCommandEncoderCopyTextureToTexture(command_buffer->encoder, &source_copy, &destination_copy, &copy_size);
    }

    command_buffer->current_graphics_pipeline = NULL;
    command_buffer->render_pass_color_target_count = 0;
    SDL_zeroa(command_buffer->render_pass_color_targets);
    SDL_zeroa(command_buffer->render_pass_resolve_targets);
    command_buffer->render_pass_depth_stencil_target = NULL;
    SDL_zeroa(command_buffer->render_pass_3d_resolve_copies);
    command_buffer->render_pass_3d_resolve_copy_count = 0;
    SDL_zeroa(command_buffer->render_pass_color_target_formats);
    command_buffer->render_pass_sample_count = SDL_GPU_SAMPLECOUNT_1;
    command_buffer->render_pass_attachment_width = 0;
    command_buffer->render_pass_attachment_height = 0;
    command_buffer->render_pass_has_depth_stencil_target = false;
    command_buffer->render_pass_depth_stencil_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    command_buffer->vertex_uniform_bind_group_offsets_dirty = false;
    command_buffer->fragment_uniform_bind_group_offsets_dirty = false;
}

static bool WEBGPU_ValidateComputeReadWriteStorageTextureBinding(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUStorageTextureReadWriteBinding *binding)
{
    WebGPUTexture *texture = (WebGPUTexture *)binding->texture;

    if (!texture || !texture->texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing read-write storage texture for compute pass");
        return false;
    }
    if (texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "read-write storage texture has been released");
        return false;
    }
    if (!(texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE) &&
        texture->header.info.usage != SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture is missing COMPUTE_STORAGE_WRITE or COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE usage");
        return false;
    }
    if (!WEBGPU_IsComputeReadWriteStorageTextureType(texture->header.info.type)) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture binding only supports 2D, 2D-array, and 3D textures");
        return false;
    }
    if (texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture binding requires sample count 1");
        return false;
    }
    if (binding->mip_level >= texture->header.info.num_levels) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture mip level exceeds texture levels");
        return false;
    }
    if (texture->header.info.type == SDL_GPU_TEXTURETYPE_3D && binding->layer != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture binding requires layer 0 for 3D textures");
        return false;
    }
    if (!WEBGPU_TextureRenderLayerInBounds(
            texture,
            binding->mip_level,
            binding->layer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture layer exceeds texture layer count");
        return false;
    }

    return true;
}

static bool WEBGPU_PrepareComputeReadWriteStorageTextureBinding(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUStorageTextureReadWriteBinding *binding,
    WebGPUStorageTextureBindingDescription *binding_description)
{
    WebGPUTexture *texture = (WebGPUTexture *)binding->texture;
    WGPUTextureViewDimension storage_view_dimension = texture->header.info.type == SDL_GPU_TEXTURETYPE_3D
        ? WGPUTextureViewDimension_3D
        : WGPUTextureViewDimension_2D;
    Uint32 storage_view_layer = texture->header.info.type == SDL_GPU_TEXTURETYPE_3D
        ? 0
        : binding->layer;
    WebGPUTextureViewDescription storage_view_description;
    WGPUTextureView storage_view;

    storage_view_description = WEBGPU_TextureViewDescriptionWithDimension(
        texture->header.info.format,
        storage_view_dimension,
        WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READWRITE,
        texture->generation,
        binding->mip_level,
        1,
        storage_view_layer,
        1);
    storage_view = WEBGPU_CreateTextureViewFromDescription(texture->texture, &storage_view_description);

    if (!storage_view) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for compute read-write storage texture");
        return false;
    }

    if (!WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture) ||
        !WEBGPU_TrackCommandBufferTextureView(command_buffer, storage_view)) {
        wgpuTextureViewRelease(storage_view);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track compute read-write storage texture");
        return false;
    }

    *binding_description = WEBGPU_StorageTextureBindingDescription(
        texture,
        storage_view,
        storage_view_dimension,
        WEBGPU_TEXTURE_VIEW_USAGE_STORAGE_READWRITE);
    wgpuTextureViewRelease(storage_view);

    return true;
}



static bool WEBGPU_ValidateComputeReadWriteStorageBufferBinding(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUStorageBufferReadWriteBinding *binding)
{
    WebGPUBuffer *buffer = (WebGPUBuffer *)binding->buffer;

    if (!buffer || !buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing read-write storage buffer for compute pass");
        return false;
    }
    if (buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "read-write storage buffer has been released");
        return false;
    }
    if (!(buffer->usage & SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE)) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage buffer is missing COMPUTE_STORAGE_WRITE usage");
        return false;
    }

    return true;
}

static bool WEBGPU_PrepareComputeReadWriteStorageBufferBinding(
    WebGPUCommandBuffer *command_buffer,
    const SDL_GPUStorageBufferReadWriteBinding *binding,
    WebGPUBufferBindingDescription *binding_description)
{
    WebGPUBuffer *buffer = (WebGPUBuffer *)binding->buffer;

    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track compute read-write storage buffer");
        return false;
    }

    *binding_description = WEBGPU_WholeBufferBindingDescription(
        buffer,
        WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ_WRITE);
    return true;
}



static void WEBGPU_BeginComputePass(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUStorageTextureReadWriteBinding *storageTextureBindings, Uint32 numStorageTextureBindings, const SDL_GPUStorageBufferReadWriteBinding *storageBufferBindings, Uint32 numStorageBufferBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;

    if (command_buffer->failed || WEBGPU_FailIfAnyPassActive(command_buffer, "BeginComputePass")) {
        return;
    }
    if (numStorageTextureBindings > MAX_COMPUTE_WRITE_TEXTURES) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage texture count exceeds MAX_COMPUTE_WRITE_TEXTURES");
        return;
    }
    if (numStorageBufferBindings > MAX_COMPUTE_WRITE_BUFFERS) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute read-write storage buffer count exceeds MAX_COMPUTE_WRITE_BUFFERS");
        return;
    }
    if (numStorageTextureBindings > 0 && !storageTextureBindings) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing compute read-write storage texture bindings");
        return;
    }
    if (numStorageBufferBindings > 0 && !storageBufferBindings) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing compute read-write storage buffer bindings");
        return;
    }
    for (Uint32 i = 0; i < numStorageTextureBindings; i += 1) {
        if (!WEBGPU_ValidateComputeReadWriteStorageTextureBinding(command_buffer, &storageTextureBindings[i])) {
            return;
        }
    }
    for (Uint32 i = 0; i < numStorageBufferBindings; i += 1) {
        if (!WEBGPU_ValidateComputeReadWriteStorageBufferBinding(command_buffer, &storageBufferBindings[i])) {
            return;
        }
    }
    for (Uint32 i = 0; i < numStorageTextureBindings; i += 1) {
        WebGPUTexture *texture = (WebGPUTexture *)storageTextureBindings[i].texture;
        if (!WEBGPU_CycleTextureIfBound(command_buffer, texture, storageTextureBindings[i].cycle, "failed to cycle compute read-write storage texture")) {
            return;
        }
    }
    for (Uint32 i = 0; i < numStorageBufferBindings; i += 1) {
        WebGPUBuffer *buffer = (WebGPUBuffer *)storageBufferBindings[i].buffer;
        if (!WEBGPU_CycleBufferIfBound(command_buffer, buffer, storageBufferBindings[i].cycle, "failed to cycle compute read-write storage buffer")) {
            return;
        }
    }

    command_buffer->compute_pass = wgpuCommandEncoderBeginComputePass(command_buffer->encoder, &pass_desc);
    if (!command_buffer->compute_pass) {
        WEBGPU_FailCommandBuffer(command_buffer, "BeginComputePass failed");
        return;
    }
    command_buffer->current_compute_pipeline = NULL;
    SDL_zeroa(command_buffer->compute_sampler_bindings);
    SDL_zeroa(command_buffer->compute_readonly_storage_texture_bindings);
    SDL_zeroa(command_buffer->compute_readonly_storage_buffer_bindings);
    SDL_zeroa(command_buffer->compute_readwrite_storage_texture_bindings);
    SDL_zeroa(command_buffer->compute_readwrite_storage_buffer_bindings);

    for (Uint32 i = 0; i < numStorageTextureBindings; i += 1) {
        if (!WEBGPU_PrepareComputeReadWriteStorageTextureBinding(
                command_buffer,
                &storageTextureBindings[i],
                &command_buffer->compute_readwrite_storage_texture_bindings[i])) {
            return;
        }
    }

    for (Uint32 i = 0; i < numStorageBufferBindings; i += 1) {
        if (!WEBGPU_PrepareComputeReadWriteStorageBufferBinding(
                command_buffer,
                &storageBufferBindings[i],
                &command_buffer->compute_readwrite_storage_buffer_bindings[i])) {
            return;
        }
    }
    command_buffer->compute_readonly_bind_group_dirty = true;
    command_buffer->compute_readwrite_bind_group_dirty = true;
    command_buffer->compute_uniform_bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_PASS);
    if (numStorageTextureBindings > 0 || numStorageBufferBindings > 0) {
        WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_STORAGE_RESOURCE);
    }
}


static void WEBGPU_BindComputePipeline(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUComputePipeline *computePipeline)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUComputePipeline *pipeline = (WebGPUComputePipeline *)computePipeline;

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "BindComputePipeline")) {
        return;
    }
    if (!pipeline || pipeline->released || !pipeline->pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid compute pipeline");
        return;
    }
    if (!WEBGPU_TrackCommandBufferComputePipeline(command_buffer, pipeline)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track compute pipeline");
        return;
    }
    wgpuComputePassEncoderSetPipeline(command_buffer->compute_pass, pipeline->pipeline);
    command_buffer->current_compute_pipeline = pipeline;
    command_buffer->compute_readonly_bind_group_dirty = true;
    command_buffer->compute_readwrite_bind_group_dirty = true;
    command_buffer->compute_uniform_bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_PIPELINE);
}

static void WEBGPU_BindComputeSamplers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, const SDL_GPUTextureSamplerBinding *textureSamplerBindings, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "BindComputeSamplers")) {
        return;
    }

    WEBGPU_BindSampledTextures(
        command_buffer,
        firstSlot,
        textureSamplerBindings,
        numBindings,
        command_buffer->compute_sampler_bindings,
        &command_buffer->compute_readonly_bind_group_dirty,
        false,
        false,
        &WEBGPU_ComputeSampledTextureBindErrors);
}


static void WEBGPU_BindComputeStorageTextures(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUTexture *const *storageTextures, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "BindComputeStorageTextures")) {
        return;
    }

    WEBGPU_BindStorageTextureBindings(
        command_buffer,
        firstSlot,
        storageTextures,
        numBindings,
        command_buffer->compute_readonly_storage_texture_bindings,
        &command_buffer->compute_readonly_bind_group_dirty,
        SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ,
        true,
        true,
        false,
        &WEBGPU_ComputeReadOnlyStorageTextureBindErrors);
}


static void WEBGPU_BindComputeStorageBuffers(SDL_GPUCommandBuffer *commandBuffer, Uint32 firstSlot, SDL_GPUBuffer *const *storageBuffers, Uint32 numBindings)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "BindComputeStorageBuffers")) {
        return;
    }

    WEBGPU_BindStorageBufferBindings(
        command_buffer,
        firstSlot,
        storageBuffers,
        numBindings,
        command_buffer->compute_readonly_storage_buffer_bindings,
        &command_buffer->compute_readonly_bind_group_dirty,
        SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        WEBGPU_BUFFER_BINDING_USAGE_COMPUTE_STORAGE_READ,
        false,
        &WEBGPU_ComputeReadOnlyStorageBufferBindErrors);
}


static void WEBGPU_PushComputeUniformData(SDL_GPUCommandBuffer *commandBuffer, Uint32 slotIndex, const void *data, Uint32 length)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WEBGPU_PushUniformData(
        command_buffer,
        slotIndex,
        data,
        length,
        command_buffer->compute_uniform_buffers,
        command_buffer->compute_uniform_buffer_offsets,
        command_buffer->compute_uniform_buffer_sizes,
        &command_buffer->compute_uniform_bind_group_dirty,
        &command_buffer->compute_uniform_bind_group_offsets_dirty);
}

static void WEBGPU_DispatchCompute(SDL_GPUCommandBuffer *commandBuffer, Uint32 groupcountX, Uint32 groupcountY, Uint32 groupcountZ)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "DispatchCompute")) {
        return;
    }
    if (groupcountX > command_buffer->renderer->limits.maxComputeWorkgroupsPerDimension ||
        groupcountY > command_buffer->renderer->limits.maxComputeWorkgroupsPerDimension ||
        groupcountZ > command_buffer->renderer->limits.maxComputeWorkgroupsPerDimension) {
        WEBGPU_FailCommandBuffer(command_buffer, "compute dispatch group count exceeds WebGPU limit");
        return;
    }
    if (!WEBGPU_ApplyComputeBindGroups(command_buffer)) {
        return;
    }
    wgpuComputePassEncoderDispatchWorkgroups(command_buffer->compute_pass, groupcountX, groupcountY, groupcountZ);
}

static void WEBGPU_DispatchComputeIndirect(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUBuffer *buffer, Uint32 offset)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *webgpu_buffer = (WebGPUBuffer *)buffer;
    const Uint32 command_size = sizeof(SDL_GPUIndirectDispatchCommand);

    if (command_buffer->failed || WEBGPU_FailIfComputePassInactive(command_buffer, "DispatchComputeIndirect")) {
        return;
    }
    if (!webgpu_buffer || !webgpu_buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid indirect compute dispatch buffer");
        return;
    }
    if (webgpu_buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect compute dispatch buffer has been released");
        return;
    }
    if ((webgpu_buffer->usage & SDL_GPU_BUFFERUSAGE_INDIRECT) == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect compute dispatch buffer was not created with SDL_GPU_BUFFERUSAGE_INDIRECT");
        return;
    }
    if ((offset % 4) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect compute dispatch offset must be 4-byte aligned");
        return;
    }
    if ((Uint64)offset + command_size > webgpu_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "indirect compute dispatch range exceeds buffer size");
        return;
    }
    if (!WEBGPU_ApplyComputeBindGroups(command_buffer)) {
        return;
    }
    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, webgpu_buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track indirect compute dispatch buffer");
        return;
    }
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(command_buffer->compute_pass, webgpu_buffer->buffer, offset);
}

static void WEBGPU_EndComputePass(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    if (!command_buffer->compute_pass) {
        if (!command_buffer->failed) {
            WEBGPU_FailCommandBuffer(command_buffer, "EndComputePass requires an active compute pass");
        }
        return;
    }
    wgpuComputePassEncoderEnd(command_buffer->compute_pass);
    wgpuComputePassEncoderRelease(command_buffer->compute_pass);
    command_buffer->compute_pass = NULL;
    command_buffer->current_compute_pipeline = NULL;
    SDL_zeroa(command_buffer->compute_sampler_bindings);
    SDL_zeroa(command_buffer->compute_readonly_storage_texture_bindings);
    SDL_zeroa(command_buffer->compute_readonly_storage_buffer_bindings);
    SDL_zeroa(command_buffer->compute_readwrite_storage_texture_bindings);
    SDL_zeroa(command_buffer->compute_readwrite_storage_buffer_bindings);
    command_buffer->compute_readonly_bind_group_dirty = false;
    command_buffer->compute_readwrite_bind_group_dirty = false;
    command_buffer->compute_uniform_bind_group_dirty = false;
    command_buffer->compute_uniform_bind_group_offsets_dirty = false;
}

static void *WEBGPU_MapTransferBuffer(SDL_GPURenderer *device, SDL_GPUTransferBuffer *transferBuffer, bool cycle)
{
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)transferBuffer;
    WebGPUTransferBufferGeneration *generation;
    (void)device;
    // Upload transfer data is copied into WebGPU staging buffers while encoding upload commands,
    // so cycling only needs to rotate CPU backing generations that are still receiving downloads.

    if (!transfer_buffer) {
        SDL_InvalidParamError("transferBuffer");
        return NULL;
    }

    generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
    if (!generation) {
        SDL_SetError("WebGPU backend: transfer buffer has no active generation");
        return NULL;
    }

    if (generation->pending_use_count > 0) {
        if (!cycle) {
            SDL_SetError("WebGPU backend: cannot map transfer buffer while GPU downloads are pending");
            return NULL;
        }
        if (!WEBGPU_CycleTransferBuffer(transfer_buffer)) {
            SDL_SetError("WebGPU backend: failed to cycle transfer buffer");
            return NULL;
        }

        generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
        if (!generation) {
            SDL_SetError("WebGPU backend: transfer buffer has no active generation after cycling");
            return NULL;
        }
    }

    return generation->data;
}

static void WEBGPU_UnmapTransferBuffer(SDL_GPURenderer *device, SDL_GPUTransferBuffer *transferBuffer)
{
    (void)device;
    (void)transferBuffer;
}

static void WEBGPU_BeginCopyPass(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->failed) {
        return;
    }
    if (WEBGPU_FailIfAnyPassActive(command_buffer, "BeginCopyPass")) {
        return;
    }
    command_buffer->copy_pass_active = true;
}

static void WEBGPU_UploadToTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureTransferInfo *source, const SDL_GPUTextureRegion *destination, bool cycle)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)source->transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    WebGPUTexture *texture = (WebGPUTexture *)destination->texture;
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WGPUBuffer staging_buffer;
    WGPUTexelCopyBufferInfo buffer_copy = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    WGPUTexelCopyTextureInfo texture_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    WGPUExtent3D copy_size = WGPU_EXTENT_3D_INIT;
    WebGPUTextureCopyEndpoint destination_endpoint;
    WebGPUSwapchainTextureDestination swapchain_destination;
    Uint32 block_width;
    Uint32 block_height;
    Uint32 block_size;
    Uint32 pixels_per_row;
    Uint32 rows_per_layer;
    Uint32 source_bytes_per_row;
    Uint32 copy_bytes_per_row;
    Uint32 staging_bytes_per_row;
    Uint64 source_blocks_per_row;
    Uint64 copy_blocks_per_row;
    Uint64 source_block_rows_per_layer64;
    Uint64 copy_block_rows64;
    Uint32 copy_block_rows;
    Uint64 source_bytes_per_layer;
    Uint64 required_source_size;
    Uint64 staging_size;
    Uint64 source_bytes_per_row64;
    Uint64 copy_bytes_per_row64;
    Uint64 staging_bytes_per_row64;
    Uint8 *mapped_data;
    Uint8 *source_data;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "UploadToTexture") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "UploadToTexture")) {
        return;
    }
    if (!transfer_buffer || !texture || (!texture->texture && !texture->from_surface)) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid UploadToTexture arguments");
        return;
    }
    if (texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture destination texture has been released");
        return;
    }
    if (transfer_buffer->usage != SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer is not an upload buffer");
        return;
    }
    if (!WEBGPU_ResolveSwapchainTextureDestination(
            command_buffer,
            texture,
            WGPUTextureUsage_CopyDst,
            "swapchain texture upload destination must be acquired by this command buffer",
            "swapchain texture upload destination is not supported by this surface",
            &swapchain_destination)) {
        return;
    }
    if (!WEBGPU_IsTransferTextureType(texture->header.info.type) ||
        destination->d == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture only supports 2D, 2D array, 3D, cube, and cube array texture regions");
        return;
    }
    if (texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture does not support multisample textures");
        return;
    }
    if (IsD24Format(texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture does not support WebGPU D24 depth formats");
        return;
    }
    if (destination->mip_level >= texture->header.info.num_levels) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture mip level exceeds texture levels");
        return;
    }
    WEBGPU_InitTextureCopyEndpointFromRegion(destination, &destination_endpoint);
    if (!WEBGPU_ResolveTextureCopyEndpointRegion(
            command_buffer,
            &destination_endpoint,
            destination->w,
            destination->h,
            "UploadToTexture region exceeds texture bounds")) {
        return;
    }
    if (destination->w == 0 || destination->h == 0) {
        return;
    }
    if (!WEBGPU_ValidateTextureCopyEndpointDepth(
            command_buffer,
            &destination_endpoint,
            destination->d,
            "UploadToTexture layer or depth range exceeds texture bounds")) {
        return;
    }

    block_width = WEBGPU_TextureFormatBlockWidth(texture->header.info.format);
    block_height = WEBGPU_TextureFormatBlockHeight(texture->header.info.format);
    block_size = WEBGPU_TextureTransferBlockSize(texture->header.info.format);
    if (block_size == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "unsupported UploadToTexture format");
        return;
    }

    pixels_per_row = source->pixels_per_row ? source->pixels_per_row : destination->w;
    rows_per_layer = source->rows_per_layer ? source->rows_per_layer : destination->h;
    if (pixels_per_row < destination->w || rows_per_layer < destination->h) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture source pitch is smaller than the destination region");
        return;
    }

    source_block_rows_per_layer64 = WEBGPU_TextureBlockCount(rows_per_layer, block_height);
    copy_block_rows64 = WEBGPU_TextureBlockCount(destination->h, block_height);
    source_blocks_per_row = WEBGPU_TextureBlockCount(pixels_per_row, block_width);
    copy_blocks_per_row = WEBGPU_TextureBlockCount(destination->w, block_width);
    source_bytes_per_row64 = source_blocks_per_row * block_size;
    copy_bytes_per_row64 = copy_blocks_per_row * block_size;
    staging_bytes_per_row64 = (copy_bytes_per_row64 + (WEBGPU_TEXTURE_COPY_BYTES_PER_ROW_ALIGNMENT - 1)) & ~(Uint64)(WEBGPU_TEXTURE_COPY_BYTES_PER_ROW_ALIGNMENT - 1);
    if (source_bytes_per_row64 > SDL_MAX_UINT32 ||
        copy_bytes_per_row64 > SDL_MAX_UINT32 ||
        staging_bytes_per_row64 > SDL_MAX_UINT32 ||
        source_block_rows_per_layer64 > SDL_MAX_UINT32 ||
        copy_block_rows64 > SDL_MAX_UINT32) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture row pitch is too large");
        return;
    }

    source_bytes_per_row = (Uint32)source_bytes_per_row64;
    copy_bytes_per_row = (Uint32)copy_bytes_per_row64;
    staging_bytes_per_row = (Uint32)staging_bytes_per_row64;
    copy_block_rows = (Uint32)copy_block_rows64;
    source_bytes_per_layer = source_block_rows_per_layer64 * source_bytes_per_row;
    required_source_size = (Uint64)source->offset + ((Uint64)destination->d - 1) * source_bytes_per_layer + (copy_block_rows64 - 1) * source_bytes_per_row + copy_bytes_per_row;
    staging_size = (Uint64)staging_bytes_per_row * copy_block_rows64 * destination->d;
    if (required_source_size > transfer_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture range exceeds transfer buffer size");
        return;
    }
    if (staging_size > SDL_SIZE_MAX) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToTexture staging size is too large");
        return;
    }
    if (!WEBGPU_InitTextureCopySize(
            command_buffer,
            texture->header.info.format,
            destination->w,
            destination->h,
            destination->d,
            "UploadToTexture copy size is too large",
            &copy_size)) {
        return;
    }
    if (!WEBGPU_CycleTextureIfBound(command_buffer, texture, cycle, "failed to cycle UploadToTexture destination")) {
        return;
    }
    if (!WEBGPU_MaterializeSwapchainTextureDestination(
            command_buffer,
            &swapchain_destination,
            "wgpuSurfaceGetCurrentTexture failed for texture upload destination")) {
        return;
    }
    transfer_generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
    if (!transfer_generation) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer has no active generation");
        return;
    }

    staging_desc.usage = WGPUBufferUsage_CopySrc;
    staging_desc.label = WEBGPU_StringView(transfer_buffer->debugName);
    staging_desc.size = staging_size;
    staging_desc.mappedAtCreation = true;
    staging_buffer = wgpuDeviceCreateBuffer(command_buffer->renderer->device, &staging_desc);
    if (!staging_buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateBuffer failed for texture upload staging buffer");
        return;
    }

    mapped_data = (Uint8 *)wgpuBufferGetMappedRange(staging_buffer, 0, (size_t)staging_size);
    if (!mapped_data) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "GetMappedRange failed for texture upload staging buffer");
        return;
    }

    source_data = (Uint8 *)transfer_generation->data + source->offset;
    for (Uint32 z = 0; z < destination->d; z += 1) {
        Uint8 *mapped_layer = mapped_data + (Uint64)z * staging_bytes_per_row * copy_block_rows;
        Uint8 *source_layer = source_data + (Uint64)z * source_bytes_per_layer;

        for (Uint32 y = 0; y < copy_block_rows; y += 1) {
            SDL_memcpy(mapped_layer + (Uint64)y * staging_bytes_per_row, source_layer + (Uint64)y * source_bytes_per_row, copy_bytes_per_row);
        }
    }
    wgpuBufferUnmap(staging_buffer);

    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, staging_buffer) ||
        !WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture)) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track UploadToTexture resources");
        return;
    }

    buffer_copy.buffer = staging_buffer;
    buffer_copy.layout.bytesPerRow = staging_bytes_per_row;
    buffer_copy.layout.rowsPerImage = copy_block_rows;

    texture_copy.texture = texture->texture;
    texture_copy.mipLevel = destination_endpoint.mip_level;
    texture_copy.origin.x = destination_endpoint.x;
    texture_copy.origin.y = destination_endpoint.y;
    texture_copy.origin.z = destination_endpoint.origin_z;
    texture_copy.aspect = WGPUTextureAspect_All;

    wgpuCommandEncoderCopyBufferToTexture(command_buffer->encoder, &buffer_copy, &texture_copy, &copy_size);
    wgpuBufferRelease(staging_buffer);
}

static void WEBGPU_UploadToBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTransferBufferLocation *source, const SDL_GPUBufferRegion *destination, bool cycle)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)source->transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    WebGPUBuffer *buffer = (WebGPUBuffer *)destination->buffer;
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WGPUBuffer staging_buffer;
    void *mapped_data;
    Uint64 staging_size;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "UploadToBuffer") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "UploadToBuffer")) {
        return;
    }
    if (!transfer_buffer || !buffer || !buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid UploadToBuffer arguments");
        return;
    }
    if (buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToBuffer destination buffer has been released");
        return;
    }
    if (transfer_buffer->usage != SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer is not an upload buffer");
        return;
    }
    if (source->offset > transfer_buffer->size || destination->size > transfer_buffer->size - source->offset ||
        destination->offset > buffer->size || destination->size > buffer->size - destination->offset) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToBuffer range exceeds buffer size");
        return;
    }
    if (destination->size == 0) {
        return;
    }
    if ((destination->offset % WEBGPU_BUFFER_COPY_ALIGNMENT) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToBuffer requires a 4-byte aligned destination offset");
        return;
    }
    if ((destination->size % WEBGPU_BUFFER_COPY_ALIGNMENT) != 0 &&
        destination->offset + destination->size != buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToBuffer requires a 4-byte aligned size unless the destination range ends at the buffer size");
        return;
    }
    if (!WEBGPU_CycleBufferIfBound(command_buffer, buffer, cycle, "failed to cycle UploadToBuffer destination")) {
        return;
    }
    transfer_generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
    if (!transfer_generation) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer has no active generation");
        return;
    }

    staging_size = WEBGPU_BufferAllocationSize(destination->size);
    if (staging_size > SDL_SIZE_MAX) {
        WEBGPU_FailCommandBuffer(command_buffer, "UploadToBuffer staging size is too large");
        return;
    }

    staging_desc.usage = WGPUBufferUsage_CopySrc;
    staging_desc.label = WEBGPU_StringView(transfer_buffer->debugName);
    staging_desc.size = staging_size;
    staging_desc.mappedAtCreation = true;
    staging_buffer = wgpuDeviceCreateBuffer(command_buffer->renderer->device, &staging_desc);
    if (!staging_buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateBuffer failed for upload staging buffer");
        return;
    }

    mapped_data = wgpuBufferGetMappedRange(staging_buffer, 0, (size_t)staging_size);
    if (!mapped_data) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "GetMappedRange failed for upload staging buffer");
        return;
    }
    SDL_memcpy(mapped_data, (Uint8 *)transfer_generation->data + source->offset, destination->size);
    if (staging_size > destination->size) {
        SDL_memset((Uint8 *)mapped_data + destination->size, 0, (size_t)(staging_size - destination->size));
    }
    wgpuBufferUnmap(staging_buffer);

    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, staging_buffer) ||
        !WEBGPU_TrackCommandBufferBuffer(command_buffer, buffer->buffer)) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track UploadToBuffer buffers");
        return;
    }

    wgpuCommandEncoderCopyBufferToBuffer(command_buffer->encoder, staging_buffer, 0, buffer->buffer, destination->offset, staging_size);
    wgpuBufferRelease(staging_buffer);
}

static void WEBGPU_DownloadFromTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureRegion *source, const SDL_GPUTextureTransferInfo *destination)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *texture = (WebGPUTexture *)source->texture;
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)destination->transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WGPUBuffer staging_buffer;
    WGPUTexelCopyTextureInfo texture_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    WGPUTexelCopyBufferInfo buffer_copy = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    WGPUExtent3D copy_size = WGPU_EXTENT_3D_INIT;
    WebGPUTextureDownload *download;
    WebGPUTextureCopyEndpoint source_endpoint;
    Uint32 block_width;
    Uint32 block_height;
    Uint32 block_size;
    Uint32 pixels_per_row;
    Uint32 rows_per_layer;
    Uint32 destination_bytes_per_row;
    Uint32 copy_bytes_per_row;
    Uint32 staging_bytes_per_row;
    Uint64 destination_blocks_per_row;
    Uint64 copy_blocks_per_row;
    Uint64 destination_block_rows_per_layer64;
    Uint64 copy_block_rows64;
    Uint32 copy_block_rows;
    Uint64 destination_bytes_per_layer;
    Uint64 required_destination_size;
    Uint64 staging_size;
    Uint64 destination_bytes_per_row64;
    Uint64 copy_bytes_per_row64;
    Uint64 staging_bytes_per_row64;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "DownloadFromTexture") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "DownloadFromTexture")) {
        return;
    }
    if (!transfer_buffer || !texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid DownloadFromTexture arguments");
        return;
    }
    if (texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture source texture has been released");
        return;
    }
    if (transfer_buffer->usage != SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer is not a download buffer");
        return;
    }
    if (texture->from_surface) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture download is not supported");
        return;
    }
    if (!texture->texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid DownloadFromTexture arguments");
        return;
    }
    if (!WEBGPU_IsTransferTextureType(texture->header.info.type) ||
        source->d == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture only supports 2D, 2D array, 3D, cube, and cube array texture regions");
        return;
    }
    if (texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture does not support multisample textures");
        return;
    }
    if (IsD24Format(texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture does not support WebGPU D24 depth formats");
        return;
    }
    if (source->mip_level >= texture->header.info.num_levels) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture mip level exceeds texture levels");
        return;
    }
    WEBGPU_InitTextureCopyEndpointFromRegion(source, &source_endpoint);
    if (!WEBGPU_ResolveTextureCopyEndpointRegion(
            command_buffer,
            &source_endpoint,
            source->w,
            source->h,
            "DownloadFromTexture region exceeds texture bounds")) {
        return;
    }
    if (source->w == 0 || source->h == 0) {
        return;
    }
    if (!WEBGPU_ValidateTextureCopyEndpointDepth(
            command_buffer,
            &source_endpoint,
            source->d,
            "DownloadFromTexture layer or depth range exceeds texture bounds")) {
        return;
    }

    block_width = WEBGPU_TextureFormatBlockWidth(texture->header.info.format);
    block_height = WEBGPU_TextureFormatBlockHeight(texture->header.info.format);
    block_size = WEBGPU_TextureTransferBlockSize(texture->header.info.format);
    if (block_size == 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "unsupported DownloadFromTexture format");
        return;
    }

    pixels_per_row = destination->pixels_per_row ? destination->pixels_per_row : source->w;
    rows_per_layer = destination->rows_per_layer ? destination->rows_per_layer : source->h;
    if (pixels_per_row < source->w || rows_per_layer < source->h) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture destination pitch is smaller than the source region");
        return;
    }

    destination_block_rows_per_layer64 = WEBGPU_TextureBlockCount(rows_per_layer, block_height);
    copy_block_rows64 = WEBGPU_TextureBlockCount(source->h, block_height);
    destination_blocks_per_row = WEBGPU_TextureBlockCount(pixels_per_row, block_width);
    copy_blocks_per_row = WEBGPU_TextureBlockCount(source->w, block_width);
    destination_bytes_per_row64 = destination_blocks_per_row * block_size;
    copy_bytes_per_row64 = copy_blocks_per_row * block_size;
    staging_bytes_per_row64 = (copy_bytes_per_row64 + (WEBGPU_TEXTURE_COPY_BYTES_PER_ROW_ALIGNMENT - 1)) & ~(Uint64)(WEBGPU_TEXTURE_COPY_BYTES_PER_ROW_ALIGNMENT - 1);
    if (destination_bytes_per_row64 > SDL_MAX_UINT32 ||
        copy_bytes_per_row64 > SDL_MAX_UINT32 ||
        staging_bytes_per_row64 > SDL_MAX_UINT32 ||
        destination_block_rows_per_layer64 > SDL_MAX_UINT32 ||
        copy_block_rows64 > SDL_MAX_UINT32) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture row pitch is too large");
        return;
    }

    destination_bytes_per_row = (Uint32)destination_bytes_per_row64;
    copy_bytes_per_row = (Uint32)copy_bytes_per_row64;
    staging_bytes_per_row = (Uint32)staging_bytes_per_row64;
    copy_block_rows = (Uint32)copy_block_rows64;
    destination_bytes_per_layer = destination_block_rows_per_layer64 * destination_bytes_per_row;
    required_destination_size = (Uint64)destination->offset + ((Uint64)source->d - 1) * destination_bytes_per_layer + (copy_block_rows64 - 1) * destination_bytes_per_row + copy_bytes_per_row;
    staging_size = (Uint64)staging_bytes_per_row * copy_block_rows64 * source->d;
    if (required_destination_size > transfer_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture range exceeds transfer buffer size");
        return;
    }
    if (staging_size > SDL_SIZE_MAX) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromTexture staging size is too large");
        return;
    }
    if (!WEBGPU_InitTextureCopySize(
            command_buffer,
            texture->header.info.format,
            source->w,
            source->h,
            source->d,
            "DownloadFromTexture copy size is too large",
            &copy_size)) {
        return;
    }
    transfer_generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
    if (!transfer_generation) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer has no active generation");
        return;
    }

    staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    staging_desc.label = WEBGPU_StringView(transfer_buffer->debugName);
    staging_desc.size = staging_size;
    staging_buffer = wgpuDeviceCreateBuffer(command_buffer->renderer->device, &staging_desc);
    if (!staging_buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateBuffer failed for texture download staging buffer");
        return;
    }

    download = (WebGPUTextureDownload *)SDL_calloc(1, sizeof(*download));
    if (!download) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to allocate texture download state");
        return;
    }
    download->staging_buffer = staging_buffer;
    download->transfer_buffer = transfer_buffer;
    download->transfer_generation = transfer_generation;
    download->destination_offset = destination->offset;
    download->row_count = copy_block_rows;
    download->depth = source->d;
    download->copy_bytes_per_row = copy_bytes_per_row;
    download->destination_bytes_per_row = destination_bytes_per_row;
    download->destination_bytes_per_layer = destination_bytes_per_layer;
    download->staging_bytes_per_row = staging_bytes_per_row;
    download->staging_size = staging_size;
    transfer_generation->pending_use_count += 1;
    transfer_buffer->pending_use_count += 1;

    if (!WEBGPU_TrackCommandBufferTexture(command_buffer, texture->texture) ||
        !WEBGPU_TrackCommandBufferTextureDownload(command_buffer, download)) {
        WEBGPU_DestroyTextureDownload(download);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track DownloadFromTexture resources");
        return;
    }

    texture_copy.texture = texture->texture;
    texture_copy.mipLevel = source_endpoint.mip_level;
    texture_copy.origin.x = source_endpoint.x;
    texture_copy.origin.y = source_endpoint.y;
    texture_copy.origin.z = source_endpoint.origin_z;
    texture_copy.aspect = WGPUTextureAspect_All;

    buffer_copy.buffer = staging_buffer;
    buffer_copy.layout.bytesPerRow = staging_bytes_per_row;
    buffer_copy.layout.rowsPerImage = copy_block_rows;

    wgpuCommandEncoderCopyTextureToBuffer(command_buffer->encoder, &texture_copy, &buffer_copy, &copy_size);
}

static void WEBGPU_DownloadFromBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferRegion *source, const SDL_GPUTransferBufferLocation *destination)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *buffer = (WebGPUBuffer *)source->buffer;
    WebGPUTransferBuffer *transfer_buffer = (WebGPUTransferBuffer *)destination->transfer_buffer;
    WebGPUTransferBufferGeneration *transfer_generation;
    WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    WGPUBuffer staging_buffer;
    WebGPUBufferDownload *download;
    Uint32 source_offset;
    Uint64 copy_source_offset;
    Uint64 staging_size;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "DownloadFromBuffer") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "DownloadFromBuffer")) {
        return;
    }
    if (!transfer_buffer || !buffer || !buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid DownloadFromBuffer arguments");
        return;
    }
    if (buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromBuffer source buffer has been released");
        return;
    }
    if (transfer_buffer->usage != SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer is not a download buffer");
        return;
    }
    if (source->offset > buffer->size || source->size > buffer->size - source->offset ||
        destination->offset > transfer_buffer->size || source->size > transfer_buffer->size - destination->offset) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromBuffer range exceeds buffer size");
        return;
    }
    if (source->size == 0) {
        return;
    }
    source_offset = source->offset % WEBGPU_BUFFER_COPY_ALIGNMENT;
    copy_source_offset = source->offset - source_offset;
    staging_size = WEBGPU_AlignUp64((Uint64)source_offset + source->size, WEBGPU_BUFFER_COPY_ALIGNMENT);
    if (copy_source_offset + staging_size > buffer->allocation_size) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromBuffer aligned source range exceeds buffer allocation");
        return;
    }
    if (staging_size > SDL_SIZE_MAX) {
        WEBGPU_FailCommandBuffer(command_buffer, "DownloadFromBuffer staging size is too large");
        return;
    }
    transfer_generation = WEBGPU_GetActiveTransferBufferGeneration(transfer_buffer);
    if (!transfer_generation) {
        WEBGPU_FailCommandBuffer(command_buffer, "transfer buffer has no active generation");
        return;
    }

    staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    staging_desc.label = WEBGPU_StringView(transfer_buffer->debugName);
    staging_desc.size = staging_size;
    staging_buffer = wgpuDeviceCreateBuffer(command_buffer->renderer->device, &staging_desc);
    if (!staging_buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateBuffer failed for buffer download staging buffer");
        return;
    }

    download = (WebGPUBufferDownload *)SDL_calloc(1, sizeof(*download));
    if (!download) {
        wgpuBufferRelease(staging_buffer);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to allocate buffer download state");
        return;
    }
    download->staging_buffer = staging_buffer;
    download->transfer_buffer = transfer_buffer;
    download->transfer_generation = transfer_generation;
    download->destination_offset = destination->offset;
    download->size = source->size;
    download->source_offset = source_offset;
    download->staging_size = staging_size;
    transfer_generation->pending_use_count += 1;
    transfer_buffer->pending_use_count += 1;

    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, buffer->buffer) ||
        !WEBGPU_TrackCommandBufferBufferDownload(command_buffer, download)) {
        WEBGPU_DestroyBufferDownload(download);
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track DownloadFromBuffer resources");
        return;
    }

    wgpuCommandEncoderCopyBufferToBuffer(command_buffer->encoder, buffer->buffer, copy_source_offset, staging_buffer, 0, staging_size);
}

static void WEBGPU_CopyTextureToTexture(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUTextureLocation *source, const SDL_GPUTextureLocation *destination, Uint32 w, Uint32 h, Uint32 d, bool cycle)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUTexture *source_texture = (WebGPUTexture *)source->texture;
    WebGPUTexture *destination_texture = (WebGPUTexture *)destination->texture;
    WebGPUTextureCopyEndpoint source_endpoint;
    WebGPUTextureCopyEndpoint destination_endpoint;
    WebGPUSwapchainTextureDestination swapchain_destination;
    WGPUTexelCopyTextureInfo source_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    WGPUTexelCopyTextureInfo destination_copy = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    WGPUExtent3D copy_size = WGPU_EXTENT_3D_INIT;
    WGPUTexture source_wgpu_texture;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "CopyTextureToTexture") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "CopyTextureToTexture")) {
        return;
    }
    if (!source_texture || (!source_texture->texture && !source_texture->from_surface) || !destination_texture || (!destination_texture->texture && !destination_texture->from_surface)) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid CopyTextureToTexture arguments");
        return;
    }
    if (source_texture->released || destination_texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture texture has been released");
        return;
    }
    if (source_texture->from_surface) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture copy source is not supported");
        return;
    }
    if (!WEBGPU_ResolveSwapchainTextureDestination(
            command_buffer,
            destination_texture,
            WGPUTextureUsage_CopyDst,
            "swapchain texture copy destination must be acquired by this command buffer",
            "swapchain texture copy destination is not supported by this surface",
            &swapchain_destination)) {
        return;
    }
    if (source_texture->header.info.format != destination_texture->header.info.format) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture requires matching texture formats");
        return;
    }
    if (IsD24Format(source_texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture does not support WebGPU D24 depth formats");
        return;
    }
    if (source_texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1 ||
        destination_texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture does not support multisample textures");
        return;
    }
    if (!WEBGPU_IsTransferTextureType(source_texture->header.info.type) ||
        !WEBGPU_IsTransferTextureType(destination_texture->header.info.type)) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture only supports 2D, 2D array, 3D, cube, and cube array texture regions");
        return;
    }
    WEBGPU_InitTextureCopyEndpointFromLocation(source, &source_endpoint);
    WEBGPU_InitTextureCopyEndpointFromLocation(destination, &destination_endpoint);

    if (source_endpoint.mip_level >= source_texture->header.info.num_levels ||
        destination_endpoint.mip_level >= destination_texture->header.info.num_levels) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyTextureToTexture mip level exceeds texture levels");
        return;
    }
    if (w == 0 || h == 0 || d == 0) {
        return;
    }
    if (!WEBGPU_ValidateTextureCopyEndpointDepth(
            command_buffer,
            &source_endpoint,
            d,
            "CopyTextureToTexture layer or depth range exceeds texture bounds") ||
        !WEBGPU_ValidateTextureCopyEndpointDepth(
            command_buffer,
            &destination_endpoint,
            d,
            "CopyTextureToTexture layer or depth range exceeds texture bounds")) {
        return;
    }
    if (!WEBGPU_ResolveTextureCopyEndpointRegion(
            command_buffer,
            &source_endpoint,
            w,
            h,
            "CopyTextureToTexture region exceeds texture bounds") ||
        !WEBGPU_ResolveTextureCopyEndpointRegion(
            command_buffer,
            &destination_endpoint,
            w,
            h,
            "CopyTextureToTexture region exceeds texture bounds")) {
        return;
    }
    if (WEBGPU_IsDepthStencilFormat(source_texture->header.info.format) &&
        (source_endpoint.x != 0 ||
         source_endpoint.y != 0 ||
         destination_endpoint.x != 0 ||
         destination_endpoint.y != 0 ||
         w != source_endpoint.mip_width ||
         h != source_endpoint.mip_height ||
         w != destination_endpoint.mip_width ||
         h != destination_endpoint.mip_height)) {
        WEBGPU_FailCommandBuffer(command_buffer, "depth-stencil CopyTextureToTexture only supports full-subresource copies");
        return;
    }
    if (!WEBGPU_InitTextureCopySize(
            command_buffer,
            source_texture->header.info.format,
            w,
            h,
            d,
            "CopyTextureToTexture copy size is too large",
            &copy_size)) {
        return;
    }

    if (!WEBGPU_MaterializeSwapchainTextureDestination(
            command_buffer,
            &swapchain_destination,
            "wgpuSurfaceGetCurrentTexture failed for texture copy destination")) {
        return;
    }

    source_wgpu_texture = source_texture->texture;
    if (!WEBGPU_CycleTextureIfBound(command_buffer, destination_texture, cycle, "failed to cycle CopyTextureToTexture destination")) {
        return;
    }
    if (source_wgpu_texture == destination_texture->texture &&
        !WEBGPU_TextureCopyEndpointsDisjoint(&source_endpoint, &destination_endpoint, d)) {
        WEBGPU_FailCommandBuffer(command_buffer, "same-texture CopyTextureToTexture requires disjoint subresources");
        return;
    }

    if (!WEBGPU_TrackCommandBufferTexture(command_buffer, source_wgpu_texture) ||
        !WEBGPU_TrackCommandBufferTexture(command_buffer, destination_texture->texture)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track CopyTextureToTexture textures");
        return;
    }

    source_copy.texture = source_wgpu_texture;
    source_copy.mipLevel = source_endpoint.mip_level;
    source_copy.origin.x = source_endpoint.x;
    source_copy.origin.y = source_endpoint.y;
    source_copy.origin.z = source_endpoint.origin_z;
    source_copy.aspect = WGPUTextureAspect_All;

    destination_copy.texture = destination_texture->texture;
    destination_copy.mipLevel = destination_endpoint.mip_level;
    destination_copy.origin.x = destination_endpoint.x;
    destination_copy.origin.y = destination_endpoint.y;
    destination_copy.origin.z = destination_endpoint.origin_z;
    destination_copy.aspect = WGPUTextureAspect_All;

    wgpuCommandEncoderCopyTextureToTexture(command_buffer->encoder, &source_copy, &destination_copy, &copy_size);
}

static void WEBGPU_CopyBufferToBuffer(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBufferLocation *source, const SDL_GPUBufferLocation *destination, Uint32 size, bool cycle)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUBuffer *source_buffer = (WebGPUBuffer *)source->buffer;
    WebGPUBuffer *destination_buffer = (WebGPUBuffer *)destination->buffer;
    WGPUBuffer source_wgpu_buffer;
    Uint64 copy_size;

    if (command_buffer->failed ||
        WEBGPU_FailIfRenderOrComputePassActive(command_buffer, "CopyBufferToBuffer") ||
        WEBGPU_FailIfCopyPassInactive(command_buffer, "CopyBufferToBuffer")) {
        return;
    }
    if (!source_buffer || !source_buffer->buffer || !destination_buffer || !destination_buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid CopyBufferToBuffer arguments");
        return;
    }
    if (source_buffer->released || destination_buffer->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyBufferToBuffer buffer has been released");
        return;
    }
    if (source->offset > source_buffer->size || size > source_buffer->size - source->offset ||
        destination->offset > destination_buffer->size || size > destination_buffer->size - destination->offset) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyBufferToBuffer range exceeds buffer size");
        return;
    }
    if ((source->offset % WEBGPU_BUFFER_COPY_ALIGNMENT) != 0 || (destination->offset % WEBGPU_BUFFER_COPY_ALIGNMENT) != 0) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyBufferToBuffer requires 4-byte aligned offsets");
        return;
    }
    if ((size % WEBGPU_BUFFER_COPY_ALIGNMENT) != 0 &&
        destination->offset + size != destination_buffer->size) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyBufferToBuffer requires a 4-byte aligned size unless the destination range ends at the buffer size");
        return;
    }
    copy_size = WEBGPU_BufferAllocationSize(size);
    if ((Uint64)source->offset + copy_size > source_buffer->allocation_size ||
        (Uint64)destination->offset + copy_size > destination_buffer->allocation_size) {
        WEBGPU_FailCommandBuffer(command_buffer, "CopyBufferToBuffer aligned range exceeds buffer allocation");
        return;
    }
    source_wgpu_buffer = source_buffer->buffer;
    if (!WEBGPU_CycleBufferIfBound(command_buffer, destination_buffer, cycle, "failed to cycle CopyBufferToBuffer destination")) {
        return;
    }
    if (source_wgpu_buffer == destination_buffer->buffer) {
        WEBGPU_FailCommandBuffer(command_buffer, "same-buffer CopyBufferToBuffer requires destination cycling");
        return;
    }
    if (!WEBGPU_TrackCommandBufferBuffer(command_buffer, source_wgpu_buffer) ||
        !WEBGPU_TrackCommandBufferBuffer(command_buffer, destination_buffer->buffer)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track CopyBufferToBuffer buffers");
        return;
    }

    wgpuCommandEncoderCopyBufferToBuffer(command_buffer->encoder, source_wgpu_buffer, source->offset, destination_buffer->buffer, destination->offset, copy_size);
}

static void WEBGPU_BlitInternal(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info,
    bool use_3d_source_depth_override,
    float source_3d_depth_override);

static void WEBGPU_GenerateMipmaps(SDL_GPUCommandBuffer *commandBuffer, SDL_GPUTexture *texture)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderer *renderer = command_buffer->renderer;
    WebGPUTexture *webgpu_texture = (WebGPUTexture *)texture;

    if (command_buffer->failed || WEBGPU_FailIfAnyPassActive(command_buffer, "GenerateMipmaps")) {
        return;
    }
    if (!webgpu_texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid GenerateMipmaps texture");
        return;
    }
    if (webgpu_texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps texture has been released");
        return;
    }
    if (webgpu_texture->from_surface) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture mipmap generation is not supported");
        return;
    }
    if (!webgpu_texture->texture) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid GenerateMipmaps texture");
        return;
    }
    if (!WEBGPU_IsBlitTextureType(webgpu_texture->header.info.type)) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps only supports 2D, 2D array, 3D, cube, and cube array textures");
        return;
    }
    if (webgpu_texture->header.info.num_levels <= 1) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps texture must have more than one mip level");
        return;
    }
    if (webgpu_texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps does not support multisample textures");
        return;
    }
    if (WEBGPU_IsDepthStencilFormat(webgpu_texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps does not support depth-stencil textures");
        return;
    }
    if (WEBGPU_TextureFormatIsCompressed(webgpu_texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps does not support compressed textures");
        return;
    }
    if ((webgpu_texture->header.info.usage & (SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) !=
        (SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps texture requires sampler and color target usage");
        return;
    }
    if (!WEBGPU_TextureFormatMatchesSampleType(renderer, webgpu_texture->header.info.format, WGPUTextureSampleType_Float)) {
        WEBGPU_FailCommandBuffer(command_buffer, "GenerateMipmaps requires a filterable sampled texture format");
        return;
    }

    if (webgpu_texture->header.info.type == SDL_GPU_TEXTURETYPE_3D) {
        for (Uint32 level = 1; level < webgpu_texture->header.info.num_levels; level += 1) {
            const Uint32 destination_mip_depth = WEBGPU_TextureMipLayerCountOrDepth(webgpu_texture, level);

            for (Uint32 depth_plane = 0; depth_plane < destination_mip_depth; depth_plane += 1) {
                SDL_GPUBlitInfo blit_info;
                const float source_depth = ((float)depth_plane + 0.5f) / (float)destination_mip_depth;

                SDL_zero(blit_info);
                blit_info.source.texture = texture;
                blit_info.source.mip_level = level - 1;
                blit_info.source.w = WEBGPU_TextureMipDimension(webgpu_texture->header.info.width, level - 1);
                blit_info.source.h = WEBGPU_TextureMipDimension(webgpu_texture->header.info.height, level - 1);
                blit_info.destination.texture = texture;
                blit_info.destination.mip_level = level;
                blit_info.destination.layer_or_depth_plane = depth_plane;
                blit_info.destination.w = WEBGPU_TextureMipDimension(webgpu_texture->header.info.width, level);
                blit_info.destination.h = WEBGPU_TextureMipDimension(webgpu_texture->header.info.height, level);
                blit_info.load_op = SDL_GPU_LOADOP_DONT_CARE;
                blit_info.filter = SDL_GPU_FILTER_LINEAR;

                WEBGPU_BlitInternal(commandBuffer, &blit_info, true, source_depth);
                if (command_buffer->failed) {
                    return;
                }
            }
        }
        return;
    }

    {
        const Uint32 layer_count = webgpu_texture->header.info.type == SDL_GPU_TEXTURETYPE_2D ? 1 : webgpu_texture->header.info.layer_count_or_depth;

        for (Uint32 layer = 0; layer < layer_count; layer += 1) {
            for (Uint32 level = 1; level < webgpu_texture->header.info.num_levels; level += 1) {
                SDL_GPUBlitInfo blit_info;

                SDL_zero(blit_info);
                blit_info.source.texture = texture;
                blit_info.source.mip_level = level - 1;
                blit_info.source.layer_or_depth_plane = layer;
                blit_info.source.w = WEBGPU_TextureMipDimension(webgpu_texture->header.info.width, level - 1);
                blit_info.source.h = WEBGPU_TextureMipDimension(webgpu_texture->header.info.height, level - 1);
                blit_info.destination.texture = texture;
                blit_info.destination.mip_level = level;
                blit_info.destination.layer_or_depth_plane = layer;
                blit_info.destination.w = WEBGPU_TextureMipDimension(webgpu_texture->header.info.width, level);
                blit_info.destination.h = WEBGPU_TextureMipDimension(webgpu_texture->header.info.height, level);
                blit_info.load_op = SDL_GPU_LOADOP_DONT_CARE;
                blit_info.filter = SDL_GPU_FILTER_LINEAR;

                SDL_BlitGPUTexture(commandBuffer, &blit_info);
                if (command_buffer->failed) {
                    return;
                }
            }
        }
    }
}

static void WEBGPU_EndCopyPass(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;

    if (command_buffer->copy_pass_active) {
        command_buffer->copy_pass_active = false;
        return;
    }
    if (command_buffer->failed) {
        return;
    }
    WEBGPU_FailCommandBuffer(command_buffer, "EndCopyPass requires an active copy pass");
}

static void WEBGPU_BlitInternal(
    SDL_GPUCommandBuffer *commandBuffer,
    const SDL_GPUBlitInfo *info,
    bool use_3d_source_depth_override,
    float source_3d_depth_override)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderer *renderer = command_buffer->renderer;
    WebGPUTextureBlitEndpoint source;
    WebGPUTextureBlitEndpoint destination;
    WebGPUSampler *sampler;
    WGPUTextureView source_view = NULL;
    SDL_GPUGraphicsPipeline *blit_pipeline = NULL;
    SDL_GPURenderPass *render_pass = NULL;
    SDL_GPUColorTargetInfo color_target_info;
    SDL_GPUViewport viewport;
    BlitFragmentUniforms blit_fragment_uniforms;
    WebGPUTextureViewDescription source_view_description;

    if (command_buffer->failed) {
        return;
    }
    if (WEBGPU_FailIfAnyPassActive(command_buffer, "Blit")) {
        return;
    }

    WEBGPU_InitTextureBlitEndpoint(&info->source, &source);
    WEBGPU_InitTextureBlitEndpoint(&info->destination, &destination);

    if (!source.texture || (!source.texture->texture && !source.texture->from_surface) || !destination.texture || (!destination.texture->texture && !destination.texture->from_surface)) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid Blit arguments");
        return;
    }
    if (source.texture->released || destination.texture->released) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit texture has been released");
        return;
    }
    if (source.texture->from_surface) {
        WEBGPU_FailCommandBuffer(command_buffer, "swapchain texture blit source is not supported");
        return;
    }
    if (!WEBGPU_ValidateSwapchainTextureDestination(
            command_buffer,
            destination.texture,
            WGPUTextureUsage_RenderAttachment,
            "swapchain blit destination must be acquired by this command buffer",
            "swapchain blit destination is not supported by this surface")) {
        return;
    }
    if (!WEBGPU_IsBlitTextureType(source.texture->header.info.type) ||
        !WEBGPU_IsBlitTextureType(destination.texture->header.info.type)) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit only supports 2D, 2D array, 3D, cube, and cube array textures");
        return;
    }
    if (source.texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1 ||
        destination.texture->header.info.sample_count != SDL_GPU_SAMPLECOUNT_1) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit does not support multisample textures");
        return;
    }
    if (!(source.texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_SAMPLER)) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit source texture was not created with SDL_GPU_TEXTUREUSAGE_SAMPLER");
        return;
    }
    if (!(destination.texture->header.info.usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit destination texture was not created with SDL_GPU_TEXTUREUSAGE_COLOR_TARGET");
        return;
    }
    if (WEBGPU_IsDepthStencilFormat(source.texture->header.info.format) ||
        WEBGPU_IsDepthStencilFormat(destination.texture->header.info.format)) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit does not support depth-stencil textures");
        return;
    }
    if (!WEBGPU_ValidateTextureBlitEndpointMip(command_buffer, &source, "Blit mip level exceeds texture levels") ||
        !WEBGPU_ValidateTextureBlitEndpointMip(command_buffer, &destination, "Blit mip level exceeds texture levels")) {
        return;
    }
    if (!WEBGPU_ValidateTextureBlitEndpointLayer(command_buffer, &source, "Blit layer exceeds texture layer count") ||
        !WEBGPU_ValidateTextureBlitEndpointLayer(command_buffer, &destination, "Blit layer exceeds texture layer count")) {
        return;
    }
    if (source.w == 0 || source.h == 0 ||
        destination.w == 0 || destination.h == 0) {
        return;
    }
    if (!WEBGPU_ResolveTextureBlitEndpointRegion(command_buffer, &source, "Blit region exceeds texture bounds") ||
        !WEBGPU_ResolveTextureBlitEndpointRegion(command_buffer, &destination, "Blit region exceeds texture bounds")) {
        return;
    }
    if (source.texture->texture == destination.texture->texture &&
        source.mip_level == destination.mip_level) {
        WEBGPU_FailCommandBuffer(command_buffer, "same-texture same-mip Blit is not supported");
        return;
    }
    if (WEBGPU_ToFilterMode(info->filter) == WGPUFilterMode_Undefined) {
        WEBGPU_FailCommandBuffer(command_buffer, "invalid Blit filter");
        return;
    }
    if (!WEBGPU_TextureFormatMatchesSampleType(renderer, source.texture->header.info.format, WGPUTextureSampleType_Float)) {
        WEBGPU_FailCommandBuffer(command_buffer, "Blit source texture format requires filterable sampling");
        return;
    }

    blit_pipeline = SDL_GPU_FetchBlitPipeline(
        command_buffer->header.device,
        source.view_type,
        destination.texture->header.info.format,
        renderer->blit_vertex_shader,
        renderer->blit_from_2d_shader,
        NULL,
        renderer->blit_from_3d_shader,
        NULL,
        NULL,
        &renderer->blit_pipelines,
        &renderer->blit_pipeline_count,
        &renderer->blit_pipeline_capacity);
    if (!blit_pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to fetch WebGPU blit pipeline");
        return;
    }

    source_view_description = WEBGPU_TextureViewDescription(
        source.texture->header.info.format,
        source.view_type,
        WEBGPU_TEXTURE_VIEW_USAGE_BLIT_SOURCE,
        source.texture->generation,
        source.mip_level,
        1,
        source.view_base_array_layer,
        1);
    source_view = WEBGPU_CreateTextureViewFromDescription(source.texture->texture, &source_view_description);
    if (!source_view) {
        WEBGPU_FailCommandBuffer(command_buffer, "CreateTextureView failed for blit source");
        return;
    }

    SDL_zero(color_target_info);
    color_target_info.texture = (SDL_GPUTexture *)destination.texture;
    color_target_info.mip_level = destination.mip_level;
    color_target_info.layer_or_depth_plane = destination.layer_or_depth_plane;
    color_target_info.load_op = info->load_op;
    color_target_info.store_op = SDL_GPU_STOREOP_STORE;
    color_target_info.clear_color = info->clear_color;
    color_target_info.cycle = info->cycle;

    render_pass = SDL_BeginGPURenderPass(commandBuffer, &color_target_info, 1, NULL);
    if (command_buffer->failed) {
        render_pass = NULL;
        goto cleanup;
    }
    if (!render_pass) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to begin blit render pass");
        goto cleanup;
    }

    viewport.x = (float)destination.x;
    viewport.y = (float)destination.y;
    viewport.w = (float)destination.w;
    viewport.h = (float)destination.h;
    viewport.min_depth = 0.0f;
    viewport.max_depth = 1.0f;
    SDL_SetGPUViewport(render_pass, &viewport);

    SDL_BindGPUGraphicsPipeline(render_pass, blit_pipeline);
    if (!command_buffer->current_graphics_pipeline) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to bind blit pipeline");
        goto cleanup;
    }

    sampler = (WebGPUSampler *)(info->filter == SDL_GPU_FILTER_NEAREST ? renderer->blit_nearest_sampler : renderer->blit_linear_sampler);
    if (!sampler || !sampler->sampler) {
        WEBGPU_FailCommandBuffer(command_buffer, "missing WebGPU blit sampler");
        goto cleanup;
    }
    if (!WEBGPU_TrackCommandBufferTexture(command_buffer, source.texture->texture) ||
        !WEBGPU_TrackCommandBufferTextureView(command_buffer, source_view) ||
        !WEBGPU_TrackCommandBufferSampler(command_buffer, sampler->sampler)) {
        WEBGPU_FailCommandBuffer(command_buffer, "failed to track blit resources");
        goto cleanup;
    }
    /* Blit validates source/destination subresources itself and tracks the
     * temporary source view directly in the command buffer. */
    command_buffer->fragment_sampler_bindings[0] = WEBGPU_SampledTextureBindingDescriptionFromTexture(
        source.texture,
        source_view,
        sampler,
        source_view_description.dimension,
        WEBGPU_TEXTURE_VIEW_USAGE_BLIT_SOURCE);
    command_buffer->fragment_resource_bind_group_dirty = true;
    WEBGPU_RecordBindGroupDirty(command_buffer->renderer, WEBGPU_BIND_GROUP_DIRTY_SAMPLED_RESOURCE);

    blit_fragment_uniforms.left = (float)source.x / source.mip_width;
    blit_fragment_uniforms.top = (float)source.y / source.mip_height;
    blit_fragment_uniforms.width = (float)source.w / source.mip_width;
    blit_fragment_uniforms.height = (float)source.h / source.mip_height;
    blit_fragment_uniforms.mip_level = 0;
    if (source.is_3d) {
        if (use_3d_source_depth_override) {
            blit_fragment_uniforms.layer_or_depth = source_3d_depth_override;
        } else {
            blit_fragment_uniforms.layer_or_depth = ((float)source.layer_or_depth_plane + 0.5f) / (float)source.mip_depth;
        }
    } else {
        blit_fragment_uniforms.layer_or_depth = 0.0f;
    }

    if (info->flip_mode & SDL_FLIP_HORIZONTAL) {
        blit_fragment_uniforms.left += blit_fragment_uniforms.width;
        blit_fragment_uniforms.width *= -1.0f;
    }
    if (info->flip_mode & SDL_FLIP_VERTICAL) {
        blit_fragment_uniforms.top += blit_fragment_uniforms.height;
        blit_fragment_uniforms.height *= -1.0f;
    }

    SDL_PushGPUFragmentUniformData(commandBuffer, 0, &blit_fragment_uniforms, sizeof(blit_fragment_uniforms));
    SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0);

cleanup:
    if (render_pass) {
        SDL_EndGPURenderPass(render_pass);
    }
    if (source_view) {
        wgpuTextureViewRelease(source_view);
    }
}

static void WEBGPU_Blit(SDL_GPUCommandBuffer *commandBuffer, const SDL_GPUBlitInfo *info)
{
    WEBGPU_BlitInternal(commandBuffer, info, false, 0.0f);
}

static bool WEBGPU_SupportsSwapchainComposition(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUSwapchainComposition swapchainComposition)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    WGPUCompositeAlphaMode alpha_mode;
    SDL_GPUTextureFormat sdl_format;

    if (!window_data || window_data->renderer != renderer) {
        return SDL_SetError("Cannot query WebGPU swapchain composition support on an unclaimed window");
    }

    if (!WEBGPU_ToCompositeAlphaMode(swapchainComposition, &alpha_mode)) {
        return false;
    }
    if (window_data->surface_format == WGPUTextureFormat_Undefined) {
        if (!WEBGPU_ChooseSurfaceFormat(window_data)) {
            return false;
        }
    }

    sdl_format = WEBGPU_ToSwapchainSDLFormat(window_data->surface_format, swapchainComposition);
    return sdl_format != SDL_GPU_TEXTUREFORMAT_INVALID;
}

static bool WEBGPU_SupportsPresentMode(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUPresentMode presentMode)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    WGPUPresentMode wgpu_present_mode;

    if (!window_data || window_data->renderer != renderer) {
        return SDL_SetError("Cannot query WebGPU present mode support on an unclaimed window");
    }

    return WEBGPU_ToPresentMode(presentMode, &wgpu_present_mode);
}

static bool WEBGPU_ClaimWindow(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    SDL_PropertiesID props;
    const char *selector;
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas_desc = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    WGPUSurfaceDescriptor surface_desc = WGPU_SURFACE_DESCRIPTOR_INIT;

    if (!WEBGPU_FailIfDeviceLost(renderer, "ClaimWindow")) {
        return false;
    }
    if (window_data) {
        if (window_data->renderer == renderer) {
            window_data->refcount += 1;
            return true;
        }
        return SDL_SetError("Window already claimed by another WebGPU device");
    }

    window_data = (WebGPUWindowData *)SDL_calloc(1, sizeof(*window_data));
    if (!window_data) {
        return false;
    }

    props = SDL_GetWindowProperties(window);
    if (!props) {
        SDL_free(window_data);
        return false;
    }
    selector = SDL_GetStringProperty(props, SDL_PROP_WINDOW_EMSCRIPTEN_CANVAS_ID_STRING, "#canvas");

    window_data->renderer = renderer;
    window_data->window = window;
    window_data->canvas_selector = SDL_strdup(selector ? selector : "#canvas");
    if (!window_data->canvas_selector) {
        SDL_free(window_data);
        return false;
    }
    window_data->surface_format = WGPUTextureFormat_Undefined;
    window_data->sdl_format = SDL_GPU_TEXTUREFORMAT_INVALID;
    window_data->refcount = 1;
    window_data->present_mode = SDL_GPU_PRESENTMODE_VSYNC;
    window_data->swapchain_composition = SDL_GPU_SWAPCHAINCOMPOSITION_SDR;

    canvas_desc.selector = WEBGPU_StringView(window_data->canvas_selector);
    surface_desc.nextInChain = &canvas_desc.chain;
    window_data->surface = wgpuInstanceCreateSurface(renderer->instance, &surface_desc);
    if (!window_data->surface) {
        SDL_free(window_data->canvas_selector);
        SDL_free(window_data);
        return SDL_SetError("wgpuInstanceCreateSurface failed");
    }

    if (!WEBGPU_ChooseSurfaceFormat(window_data) ||
        !WEBGPU_EnsureWindowConfigured(window_data)) {
        if (window_data->configured) {
            wgpuSurfaceUnconfigure(window_data->surface);
        }
        wgpuSurfaceRelease(window_data->surface);
        SDL_free(window_data->canvas_selector);
        SDL_free(window_data);
        return false;
    }

    if (renderer->claimed_window_count >= renderer->claimed_window_capacity) {
        Uint32 new_capacity = renderer->claimed_window_capacity ? renderer->claimed_window_capacity * 2 : 1;
        WebGPUWindowData **claimed_windows = (WebGPUWindowData **)SDL_realloc(renderer->claimed_windows, new_capacity * sizeof(*claimed_windows));
        if (!claimed_windows) {
            if (window_data->configured) {
                wgpuSurfaceUnconfigure(window_data->surface);
            }
            wgpuSurfaceRelease(window_data->surface);
            SDL_free(window_data->canvas_selector);
            SDL_free(window_data);
            return false;
        }
        renderer->claimed_windows = claimed_windows;
        renderer->claimed_window_capacity = new_capacity;
    }

    if (!SDL_SetPointerProperty(props, WINDOW_PROPERTY_DATA, window_data)) {
        if (window_data->configured) {
            wgpuSurfaceUnconfigure(window_data->surface);
        }
        wgpuSurfaceRelease(window_data->surface);
        SDL_free(window_data->canvas_selector);
        SDL_free(window_data);
        return false;
    }

    renderer->claimed_windows[renderer->claimed_window_count] = window_data;
    renderer->claimed_window_count += 1;
    SDL_AddWindowEventWatch(SDL_WINDOW_EVENT_WATCH_NORMAL, WEBGPU_OnWindowSurfaceStateChanged, window);
    return true;
}

static void WEBGPU_ReleaseWindow(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);

    if (!window_data || window_data->renderer != renderer) {
        return;
    }
    if (window_data->refcount > 1) {
        window_data->refcount -= 1;
        return;
    }

    for (Uint32 i = 0; i < renderer->claimed_window_count; i += 1) {
        if (renderer->claimed_windows[i] == window_data) {
            renderer->claimed_windows[i] = renderer->claimed_windows[renderer->claimed_window_count - 1];
            renderer->claimed_window_count -= 1;
            break;
        }
    }

    SDL_ClearProperty(SDL_GetWindowProperties(window), WINDOW_PROPERTY_DATA);
    SDL_RemoveWindowEventWatch(SDL_WINDOW_EVENT_WATCH_NORMAL, WEBGPU_OnWindowSurfaceStateChanged, window);
    WEBGPU_ClearWindowSwapchainSubmissions(renderer, window_data);
    if (window_data->surface) {
        if (window_data->configured) {
            wgpuSurfaceUnconfigure(window_data->surface);
            window_data->configured = false;
        }
        wgpuSurfaceRelease(window_data->surface);
        window_data->surface = NULL;
    }
    SDL_free(window_data->canvas_selector);
    window_data->canvas_selector = NULL;
    window_data->configured_usage = WGPUTextureUsage_None;
    window_data->width = 0;
    window_data->height = 0;
    window_data->needs_configure = true;
    window_data->refcount = 0;
    window_data->window = NULL;
    WEBGPU_MaybeDestroyWindowData(window_data);
}

static bool WEBGPU_SetSwapchainParameters(SDL_GPURenderer *driverData, SDL_Window *window, SDL_GPUSwapchainComposition swapchainComposition, SDL_GPUPresentMode presentMode)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    SDL_GPUSwapchainComposition previous_composition;
    SDL_GPUPresentMode previous_present_mode;
    SDL_GPUTextureFormat previous_sdl_format;
    SDL_GPUTextureFormat sdl_format;

    if (!WEBGPU_FailIfDeviceLost(renderer, "SetSwapchainParameters")) {
        return false;
    }
    if (!window_data || window_data->renderer != renderer) {
        return SDL_SetError("Cannot set swapchain parameters on an unclaimed window");
    }
    if (!WEBGPU_SupportsSwapchainComposition(driverData, window, swapchainComposition) ||
        !WEBGPU_SupportsPresentMode(driverData, window, presentMode)) {
        return SDL_SetError("Unsupported WebGPU swapchain parameters");
    }

    sdl_format = WEBGPU_ToSwapchainSDLFormat(window_data->surface_format, swapchainComposition);
    if (sdl_format == SDL_GPU_TEXTUREFORMAT_INVALID) {
        return SDL_SetError("Unsupported WebGPU swapchain parameters");
    }

    if (swapchainComposition == window_data->swapchain_composition &&
        presentMode == window_data->present_mode &&
        sdl_format == window_data->sdl_format &&
        !window_data->needs_configure) {
        return true;
    }

    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return SDL_SetError("Cannot set WebGPU swapchain parameters while a swapchain texture is awaiting submission");
    }

    previous_composition = window_data->swapchain_composition;
    previous_present_mode = window_data->present_mode;
    previous_sdl_format = window_data->sdl_format;
    window_data->swapchain_composition = swapchainComposition;
    window_data->present_mode = presentMode;
    window_data->sdl_format = sdl_format;
    window_data->needs_configure = true;
    if (!WEBGPU_EnsureWindowConfigured(window_data)) {
        window_data->swapchain_composition = previous_composition;
        window_data->present_mode = previous_present_mode;
        window_data->sdl_format = previous_sdl_format;
        window_data->needs_configure = true;
        return false;
    }
    return true;
}

static bool WEBGPU_SetAllowedFramesInFlight(SDL_GPURenderer *driverData, Uint32 allowedFramesInFlight)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;

    if (allowedFramesInFlight < 1 || allowedFramesInFlight > MAX_FRAMES_IN_FLIGHT) {
        return false;
    }
    if (!WEBGPU_WaitForAllPendingSubmissions(renderer)) {
        (void)WEBGPU_DrainDeviceLoss(renderer, "SetAllowedFramesInFlight");
        return false;
    }

    for (Uint32 i = 0; i < renderer->claimed_window_count; i += 1) {
        WEBGPU_ClearWindowSwapchainSubmissions(renderer, renderer->claimed_windows[i]);
    }
    renderer->allowed_frames_in_flight = allowedFramesInFlight;

    WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
    return WEBGPU_DrainRuntimeErrors(renderer, "SetAllowedFramesInFlight");
}

static SDL_GPUTextureFormat WEBGPU_GetSwapchainTextureFormat(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    (void)driverData;

    if (!window_data) {
        SDL_SetError("Cannot get swapchain format for an unclaimed window");
        return SDL_GPU_TEXTUREFORMAT_INVALID;
    }
    return window_data->sdl_format;
}

static SDL_GPUCommandBuffer *WEBGPU_AcquireCommandBuffer(SDL_GPURenderer *driverData)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUCommandBuffer *command_buffer;

    if (!WEBGPU_DrainRuntimeErrors(renderer, "AcquireCommandBuffer")) {
        return NULL;
    }

    command_buffer = (WebGPUCommandBuffer *)SDL_calloc(1, sizeof(*command_buffer));
    if (!command_buffer) {
        return NULL;
    }

    command_buffer->renderer = renderer;
    command_buffer->encoder = wgpuDeviceCreateCommandEncoder(renderer->device, NULL);
    if (!command_buffer->encoder) {
        SDL_free(command_buffer);
        SDL_SetError("wgpuDeviceCreateCommandEncoder failed");
        return NULL;
    }

    return (SDL_GPUCommandBuffer *)command_buffer;
}

static bool WEBGPU_INTERNAL_AcquireSwapchainTexture(
    bool block,
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchainTexture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    WebGPUTexture *texture;
    bool slot_available = true;

    *swapchainTexture = NULL;
    if (swapchainTextureWidth) {
        *swapchainTextureWidth = 0;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = 0;
    }

    if (command_buffer->failed || WEBGPU_FailIfAnyPassActive(command_buffer, "AcquireSwapchainTexture")) {
        return false;
    }

    if (!WEBGPU_DrainRuntimeErrors(command_buffer->renderer, "AcquireSwapchainTexture")) {
        command_buffer->failed = true;
        return false;
    }

    if (!window_data || window_data->renderer != command_buffer->renderer) {
        return SDL_SetError("Cannot acquire swapchain texture for an unclaimed window");
    }
    if (command_buffer->swapchain_texture) {
        return SDL_SetError("Command buffer already acquired a swapchain texture");
    }

    if (!WEBGPU_EnsureWindowConfigured(window_data)) {
        return false;
    }
    if (!window_data->configured) {
        return true;
    }

    if (!WEBGPU_WaitForSwapchainFrameSlot(
            command_buffer->renderer,
            window_data,
            block,
            true,
            &slot_available,
            block ? "WaitAndAcquireSwapchainTexture" : "AcquireSwapchainTexture")) {
        command_buffer->failed = true;
        return false;
    }
    if (!slot_available) {
        return true;
    }

    if (!WEBGPU_PrepareSwapchainBlitPipelines(command_buffer->header.device, command_buffer->renderer, window_data->sdl_format)) {
        return false;
    }

    texture = (WebGPUTexture *)SDL_calloc(1, sizeof(*texture));
    if (!texture) {
        return false;
    }
    if (!WEBGPU_RetainSwapchainWindowData(window_data)) {
        SDL_free(texture);
        return false;
    }

    texture->generation = WEBGPU_RESOURCE_GENERATION_INITIAL;
    texture->refcount = 1;
    texture->from_surface = true;
    texture->window_data = window_data;
    texture->window_configuration_generation = window_data->configuration_generation;
    texture->header.info.type = SDL_GPU_TEXTURETYPE_2D;
    texture->header.info.format = window_data->sdl_format;
    texture->header.info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET;
    texture->header.info.width = window_data->width;
    texture->header.info.height = window_data->height;
    texture->header.info.layer_count_or_depth = 1;
    texture->header.info.num_levels = 1;
    texture->header.info.sample_count = SDL_GPU_SAMPLECOUNT_1;

    command_buffer->swapchain_texture = texture;
    *swapchainTexture = (SDL_GPUTexture *)texture;
    if (swapchainTextureWidth) {
        *swapchainTextureWidth = window_data->width;
    }
    if (swapchainTextureHeight) {
        *swapchainTextureHeight = window_data->height;
    }
    return true;
}

static bool WEBGPU_AcquireSwapchainTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchainTexture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    return WEBGPU_INTERNAL_AcquireSwapchainTexture(false, commandBuffer, window, swapchainTexture, swapchainTextureWidth, swapchainTextureHeight);
}

static bool WEBGPU_WaitForSwapchain(SDL_GPURenderer *driverData, SDL_Window *window)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUWindowData *window_data = WEBGPU_FetchWindowData(window);
    bool slot_available = true;

    if (!WEBGPU_DrainRuntimeErrors(renderer, "WaitForSwapchain")) {
        return false;
    }
    if (!window_data || window_data->renderer != renderer) {
        return SDL_SetError("Cannot wait for swapchain texture for an unclaimed window");
    }
    return WEBGPU_WaitForSwapchainFrameSlot(renderer, window_data, true, false, &slot_available, "WaitForSwapchain");
}

static bool WEBGPU_WaitAndAcquireSwapchainTexture(
    SDL_GPUCommandBuffer *commandBuffer,
    SDL_Window *window,
    SDL_GPUTexture **swapchainTexture,
    Uint32 *swapchainTextureWidth,
    Uint32 *swapchainTextureHeight)
{
    return WEBGPU_INTERNAL_AcquireSwapchainTexture(true, commandBuffer, window, swapchainTexture, swapchainTextureWidth, swapchainTextureHeight);
}

static bool WEBGPU_SubmitInternal(SDL_GPUCommandBuffer *commandBuffer, WebGPUFence **out_fence)
{
    WebGPUCommandBuffer *command_buffer = (WebGPUCommandBuffer *)commandBuffer;
    WebGPURenderer *renderer = command_buffer->renderer;
    WGPUQueueWorkDoneCallbackInfo work_done_callback = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    WGPUCommandBuffer commands;
    WebGPUSubmission *submission = NULL;
    WebGPUFence *fence = NULL;
    const bool can_wait_before_submit = !WEBGPU_HasSwapchainSubmitWindow(renderer);

    // These drain helpers may poll/yield, so they must not run while any swapchain texture is awaiting submit.
    if (can_wait_before_submit) {
        WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
    }

    if (command_buffer->render_pass) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot submit command buffer with an active render pass");
    }
    if (command_buffer->compute_pass) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot submit command buffer with an active compute pass");
    }
    if (command_buffer->copy_pass_active) {
        WEBGPU_FailCommandBuffer(command_buffer, "cannot submit command buffer with an active copy pass");
    }
    if (command_buffer->failed) {
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return false;
    }

    if (can_wait_before_submit && !WEBGPU_DrainRuntimeErrors(renderer, "Submit")) {
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return false;
    }
    if (!can_wait_before_submit && !WEBGPU_FailIfDeviceLost(renderer, "Submit")) {
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return false;
    }

    if (!can_wait_before_submit) {
        // Do not yield while a swapchain texture is awaiting submit.
        commands = wgpuCommandEncoderFinish(command_buffer->encoder, NULL);
        if (!commands) {
            WEBGPU_DestroyCommandBuffer(command_buffer);
            return SDL_SetError("wgpuCommandEncoderFinish failed");
        }
    } else {
        commands = WEBGPU_FinishCommandEncoder(renderer, command_buffer->encoder);
        if (!commands) {
            WEBGPU_DestroyCommandBuffer(command_buffer);
            return false;
        }
    }

    submission = (WebGPUSubmission *)SDL_calloc(1, sizeof(*submission));
    if (!submission) {
        wgpuCommandBufferRelease(commands);
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return false;
    }
    submission->status = WGPUQueueWorkDoneStatus_Error;
    submission->refcount = 1;
    submission->renderer = renderer;

    if (out_fence) {
        fence = (WebGPUFence *)SDL_calloc(1, sizeof(*fence));
        if (!fence) {
            WEBGPU_DestroySubmission(submission, false);
            wgpuCommandBufferRelease(commands);
            WEBGPU_DestroyCommandBuffer(command_buffer);
            return false;
        }
    }

    WEBGPU_TransferCommandBufferResourcesToSubmission(command_buffer, submission);
    if (!WEBGPU_TrackPendingSubmission(renderer, submission)) {
        SDL_free(fence);
        WEBGPU_DestroySubmission(submission, false);
        wgpuCommandBufferRelease(commands);
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return false;
    }
    if (fence) {
        SDL_LockMutex(renderer->fence_lock);
        submission->handle_retained = true;
        WEBGPU_RetainSubmissionReferenceLocked(submission);
        SDL_UnlockMutex(renderer->fence_lock);
        fence->submission = submission;
    }

    wgpuQueueSubmit(renderer->queue, 1, &commands);
    submission->submitted = true;
    WEBGPU_EndSwapchainSubmitWindow(command_buffer);
    WEBGPU_RecordSubmittedSwapchainSubmission(command_buffer, submission);

    work_done_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    work_done_callback.callback = WEBGPU_OnQueueWorkDone;
    work_done_callback.userdata1 = submission;
    submission->future = wgpuQueueOnSubmittedWorkDone(renderer->queue, work_done_callback);
    if (submission->future.id == 0) {
        bool destroy = false;

        SDL_LockMutex(renderer->fence_lock);
        submission->completion_tracking_failed = true;
        if (fence && submission->handle_retained) {
            submission->handle_retained = false;
            destroy = WEBGPU_ReleaseSubmissionReferenceLocked(submission);
            fence->submission = NULL;
        }
        SDL_UnlockMutex(renderer->fence_lock);

        if (destroy) {
            WEBGPU_DestroySubmission(submission, false);
        }
        SDL_free(fence);
        wgpuCommandBufferRelease(commands);
        WEBGPU_RecordBackendDeviceLost(renderer, "wgpuQueueOnSubmittedWorkDone failed after WebGPU queue submission");
        WEBGPU_DestroyCommandBuffer(command_buffer);
        return SDL_SetError("wgpuQueueOnSubmittedWorkDone failed after WebGPU queue submission");
    }
    if (out_fence) {
        *out_fence = fence;
    }

    wgpuCommandBufferRelease(commands);
    WEBGPU_DestroyCommandBuffer(command_buffer);
    return true;
}

static bool WEBGPU_Submit(SDL_GPUCommandBuffer *commandBuffer)
{
    return WEBGPU_SubmitInternal(commandBuffer, NULL);
}

static SDL_GPUFence *WEBGPU_SubmitAndAcquireFence(SDL_GPUCommandBuffer *commandBuffer)
{
    WebGPUFence *fence = NULL;

    if (!WEBGPU_SubmitInternal(commandBuffer, &fence)) {
        return NULL;
    }
    return (SDL_GPUFence *)fence;
}

static bool WEBGPU_Cancel(SDL_GPUCommandBuffer *commandBuffer)
{
    WEBGPU_DestroyCommandBuffer((WebGPUCommandBuffer *)commandBuffer);
    return true;
}

static bool WEBGPU_Wait(SDL_GPURenderer *driverData)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;

    if (!WEBGPU_WaitForAllPendingSubmissions(renderer)) {
        (void)WEBGPU_DrainDeviceLoss(renderer, "Wait");
        return false;
    }

    WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
    return WEBGPU_DrainRuntimeErrors(renderer, "Wait");
}

static WebGPUSubmission *WEBGPU_RetainFenceSubmission(WebGPURenderer *renderer, SDL_GPUFence *fence)
{
    WebGPUFence *webgpu_fence = (WebGPUFence *)fence;
    WebGPUSubmission *submission;

    if (!webgpu_fence) {
        SDL_InvalidParamError("fence");
        return NULL;
    }

    SDL_LockMutex(renderer->fence_lock);
    submission = webgpu_fence->submission;
    if (!submission || !submission->handle_retained) {
        SDL_UnlockMutex(renderer->fence_lock);
        SDL_SetError("Invalid WebGPU fence");
        return NULL;
    }
    WEBGPU_RetainSubmissionReferenceLocked(submission);
    SDL_UnlockMutex(renderer->fence_lock);
    return submission;
}

static bool WEBGPU_WaitForFences(SDL_GPURenderer *driverData, bool waitAll, SDL_GPUFence *const *fences, Uint32 numFences)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUSubmission **submissions;
    WGPUFutureWaitInfo *wait_infos;
    bool result = false;

    if (numFences == 0) {
        return WEBGPU_DrainRuntimeErrors(renderer, "WaitForFences");
    }

    if (WEBGPU_HasSwapchainSubmitWindow(renderer)) {
        return SDL_SetError("Cannot wait for WebGPU fences while a swapchain texture is awaiting submission");
    }

    if (waitAll) {
        for (Uint32 i = 0; i < numFences; i += 1) {
            WebGPUSubmission *submission = WEBGPU_RetainFenceSubmission(renderer, fences[i]);
            bool completed = false;

            if (!submission) {
                return false;
            }
            result = WEBGPU_WaitForSubmission(renderer, submission, SDL_MAX_UINT64, &completed);
            WEBGPU_ReleaseSubmissionReference(renderer, submission);
            if (!result || !completed) {
                (void)WEBGPU_DrainDeviceLoss(renderer, "WaitForFences");
                return false;
            }
        }
        WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
        return WEBGPU_DrainRuntimeErrors(renderer, "WaitForFences");
    }

    if (numFences > renderer->wait_any_max_count) {
        return SDL_SetError("WebGPU wait-any fence count %u exceeds supported count %zu", numFences, renderer->wait_any_max_count);
    }

    submissions = SDL_stack_alloc(WebGPUSubmission *, numFences);
    wait_infos = SDL_stack_alloc(WGPUFutureWaitInfo, numFences);
    if (!submissions || !wait_infos) {
        SDL_stack_free(wait_infos);
        SDL_stack_free(submissions);
        return false;
    }
    SDL_memset(submissions, 0, numFences * sizeof(*submissions));

    for (Uint32 i = 0; i < numFences; i += 1) {
        submissions[i] = WEBGPU_RetainFenceSubmission(renderer, fences[i]);
        if (!submissions[i]) {
            goto wait_any_done;
        }
        if (submissions[i]->completion_tracking_failed) {
            SDL_SetError("WebGPU submission completion tracking failed");
            goto wait_any_done;
        }
        if (submissions[i]->future.id == 0) {
            SDL_SetError("WebGPU submission has no completion future");
            goto wait_any_done;
        }
        wait_infos[i] = (WGPUFutureWaitInfo)WGPU_FUTURE_WAIT_INFO_INIT;
        wait_infos[i].future = submissions[i]->future;
    }

    while (true) {
        WGPUWaitStatus status;

        for (Uint32 i = 0; i < numFences; i += 1) {
            if (SDL_GetAtomicInt(&submissions[i]->completed)) {
                if (submissions[i]->status != WGPUQueueWorkDoneStatus_Success) {
                    if (!WEBGPU_DrainDeviceLoss(renderer, "WaitForFences")) {
                        goto wait_any_done;
                    }
                    SDL_SetError("WebGPU queue work completed with status %d", (int)submissions[i]->status);
                    goto wait_any_done;
                }
                result = WEBGPU_ProcessSubmissionDownloads(renderer, submissions[i]);
                WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
                if (result) {
                    result = WEBGPU_DrainRuntimeErrors(renderer, "WaitForFences");
                }
                goto wait_any_done;
            }
        }

        status = wgpuInstanceWaitAny(renderer->instance, numFences, wait_infos, SDL_MAX_UINT64);
        if (status != WGPUWaitStatus_Success) {
            if (!WEBGPU_DrainDeviceLoss(renderer, "WaitForFences")) {
                goto wait_any_done;
            }
            SDL_SetError("wgpuInstanceWaitAny failed with status %d", (int)status);
            goto wait_any_done;
        }
        for (Uint32 i = 0; i < numFences; i += 1) {
            SDL_assert(!wait_infos[i].completed || SDL_GetAtomicInt(&submissions[i]->completed));
        }
    }

wait_any_done:
    for (Uint32 i = 0; i < numFences; i += 1) {
        WEBGPU_ReleaseSubmissionReference(renderer, submissions[i]);
    }
    SDL_stack_free(wait_infos);
    SDL_stack_free(submissions);
    return result;
}

static bool WEBGPU_QueryFence(SDL_GPURenderer *driverData, SDL_GPUFence *fence)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUSubmission *submission = WEBGPU_RetainFenceSubmission(renderer, fence);
    bool completed = false;
    bool result;

    if (!submission) {
        return false;
    }
    result = WEBGPU_WaitForSubmission(renderer, submission, 0, &completed);
    WEBGPU_ReleaseSubmissionReference(renderer, submission);
    if (!result) {
        (void)WEBGPU_DrainDeviceLoss(renderer, "QueryFence");
        return false;
    }
    WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
    if (!WEBGPU_DrainRuntimeErrors(renderer, "QueryFence")) {
        return false;
    }
    return completed;
}

static void WEBGPU_ReleaseFence(SDL_GPURenderer *driverData, SDL_GPUFence *fence)
{
    WebGPUFence *webgpu_fence = (WebGPUFence *)fence;
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;
    WebGPUSubmission *submission;
    bool destroy = false;

    if (!webgpu_fence) {
        return;
    }

    SDL_LockMutex(renderer->fence_lock);
    submission = webgpu_fence->submission;
    if (submission && submission->handle_retained) {
        submission->handle_retained = false;
        destroy = WEBGPU_ReleaseSubmissionReferenceLocked(submission);
    }
    webgpu_fence->submission = NULL;
    SDL_UnlockMutex(renderer->fence_lock);

    if (destroy) {
        WEBGPU_DestroySubmission(submission, false);
    }
    SDL_free(webgpu_fence);
    WEBGPU_DrainRetiredSubmissionsIfAllowed(renderer);
}

static bool WEBGPU_SupportsTextureFormat(SDL_GPURenderer *driverData, SDL_GPUTextureFormat format, SDL_GPUTextureType type, SDL_GPUTextureUsageFlags usage)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;

    if (!WEBGPU_IsTransferTextureType(type)) {
        return false;
    }
    if (WEBGPU_ToWGPUTextureFormat(format) == WGPUTextureFormat_Undefined) {
        return false;
    }
    if (!WEBGPU_TextureFormatSupportedByDevice(renderer, format)) {
        return false;
    }
    if (!WEBGPU_TextureFormatSupportedForType(renderer, format, type)) {
        return false;
    }
    if (IsD24Format(format)) {
        SDL_GPUTextureCreateInfo createinfo;

        SDL_zero(createinfo);
        createinfo.type = type;
        createinfo.format = format;
        createinfo.usage = usage;
        createinfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
        return IsD24AcceptedTextureCreateInfo(&createinfo);
    }
    if (!WEBGPU_Is2DTextureType(type)) {
        if (type == SDL_GPU_TEXTURETYPE_3D) {
            if (WEBGPU_IsDepthStencilFormat(format)) {
                return false;
            }
            if (WEBGPU_TextureFormatIsCompressed(format)) {
                if (usage == 0) {
                    return true;
                }
                if (usage & ~SDL_GPU_TEXTUREUSAGE_SAMPLER) {
                    return false;
                }
                return WEBGPU_TextureFormatSupportsTextureBindingForType(format, type);
            }
            if (usage == 0) {
                return true;
            }
            if (usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
                return false;
            }
            if ((usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
                !WEBGPU_TextureFormatSupportsColorTarget(renderer, format)) {
                return false;
            }
            if (!WEBGPU_ValidateStorageTexturePolicy(
                    renderer,
                    format,
                    type,
                    usage,
                    false)) {
                return false;
            }
            return !(usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) ||
                   WEBGPU_TextureFormatSupportsTextureBindingForType(format, type);
        }
        if (type == SDL_GPU_TEXTURETYPE_CUBE ||
            type == SDL_GPU_TEXTURETYPE_CUBE_ARRAY) {
            if (WEBGPU_IsDepthStencilFormat(format)) {
                if (!WEBGPU_TextureFormatIsAcceptedDepthCube(format)) {
                    return false;
                }
                if (usage == 0) {
                    return false;
                }
                if (usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
                    return false;
                }
                return !(usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) ||
                       WEBGPU_TextureFormatSupportsTextureBindingForType(format, type);
            }
            if (usage & ~(SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET)) {
                return false;
            }
            if ((usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
                !WEBGPU_TextureFormatSupportsColorTarget(renderer, format)) {
                return false;
            }
            return !(usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) ||
                   WEBGPU_TextureFormatSupportsTextureBindingForType(format, type);
        }
        return usage == 0 &&
               !WEBGPU_TextureFormatIsCompressed(format) &&
               WEBGPU_TextureBytesPerPixel(format) != 0;
    }
    if (usage & ~(SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE)) {
        return false;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) {
        if (WEBGPU_IsDepthStencilFormat(format) ||
            !WEBGPU_TextureFormatSupportsColorTarget(renderer, format)) {
            return false;
        }
    }
    if (!WEBGPU_ValidateStorageTexturePolicy(
            renderer,
            format,
            type,
            usage,
            false)) {
        return false;
    }
    if (usage & SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET) {
        return !(usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
               WEBGPU_IsDepthStencilFormat(format) &&
               (!(usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) || WEBGPU_TextureFormatSupportsTextureBindingForType(format, type));
    }
    if (WEBGPU_IsDepthStencilFormat(format)) {
        return (usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) &&
               !(usage & SDL_GPU_TEXTUREUSAGE_COLOR_TARGET) &&
               WEBGPU_TextureFormatSupportsTextureBindingForType(format, type);
    }
    if ((usage & SDL_GPU_TEXTUREUSAGE_SAMPLER) &&
        !WEBGPU_TextureFormatSupportsTextureBindingForType(format, type)) {
        return false;
    }
    return true;
}

static bool WEBGPU_SupportsSampleCount(SDL_GPURenderer *driverData, SDL_GPUTextureFormat format, SDL_GPUSampleCount desiredSampleCount)
{
    WebGPURenderer *renderer = (WebGPURenderer *)driverData;

    if (WEBGPU_ToWGPUTextureFormat(format) == WGPUTextureFormat_Undefined) {
        return false;
    }
    if (!WEBGPU_TextureFormatSupportedByDevice(renderer, format)) {
        return false;
    }
    if (desiredSampleCount == SDL_GPU_SAMPLECOUNT_1) {
        return true;
    }
    if (IsD24Format(format)) {
        return false;
    }
    if (desiredSampleCount == SDL_GPU_SAMPLECOUNT_4) {
        return WEBGPU_IsDepthStencilFormat(format) ||
               WEBGPU_TextureFormatSupportsMultisampleColorTarget(renderer, format);
    }
    return false;
}

static void WEBGPU_DestroyRenderer(WebGPURenderer *renderer)
{
    if (renderer->error_lock) {
        SDL_LockMutex(renderer->error_lock);
        renderer->device_shutting_down = true;
        SDL_UnlockMutex(renderer->error_lock);
    }
    if (renderer->device_lost_callback_state) {
        renderer->device_lost_callback_state->renderer_owned = false;
    }

    if (renderer->fence_lock) {
        (void)WEBGPU_WaitForAllPendingSubmissions(renderer);
        for (Uint32 i = 0; i < renderer->claimed_window_count; i += 1) {
            WEBGPU_ClearWindowSwapchainSubmissions(renderer, renderer->claimed_windows[i]);
        }
    }
    WEBGPU_DestroyPendingSubmissions(renderer);
    WEBGPU_DestroyBlitResources(renderer);

    while (renderer->claimed_window_count > 0) {
        WEBGPU_ReleaseWindow((SDL_GPURenderer *)renderer, renderer->claimed_windows[renderer->claimed_window_count - 1]->window);
    }

    SDL_free(renderer->claimed_windows);
    SDL_free(renderer->buffer_references);
    SDL_free(renderer->texture_references);
    if (renderer->fence_lock) {
        SDL_DestroyMutex(renderer->fence_lock);
    }
    if (renderer->resource_lock) {
        SDL_DestroyMutex(renderer->resource_lock);
    }
    if (renderer->queue) {
        wgpuQueueRelease(renderer->queue);
        renderer->queue = NULL;
    }
    if (renderer->device) {
        wgpuDeviceRelease(renderer->device);
        renderer->device = NULL;
        (void)WEBGPU_PollDeviceLossFuture(renderer, "DestroyRenderer", false);
    }
    if (renderer->adapter) {
        wgpuAdapterRelease(renderer->adapter);
        renderer->adapter = NULL;
    }
    if (renderer->instance) {
        wgpuInstanceRelease(renderer->instance);
        renderer->instance = NULL;
    }
    SDL_free(renderer->device_lost_callback_state);
    renderer->device_lost_callback_state = NULL;
    if (renderer->error_lock) {
        SDL_LockMutex(renderer->error_lock);
        SDL_free(renderer->last_uncaptured_error_message);
        renderer->last_uncaptured_error_message = NULL;
        renderer->uncaptured_error_count = 0;
        SDL_free(renderer->device_lost_message);
        renderer->device_lost_message = NULL;
        renderer->pending_device_loss = false;
        SDL_UnlockMutex(renderer->error_lock);
        SDL_DestroyMutex(renderer->error_lock);
    }
    if (renderer->props) {
        SDL_DestroyProperties(renderer->props);
    }
    SDL_free(renderer);
}

static void WEBGPU_DestroyDevice(SDL_GPUDevice *device)
{
    WebGPURenderer *renderer = (WebGPURenderer *)device->driverData;
    WEBGPU_DestroyRenderer(renderer);
    SDL_free(device);
}

static bool WEBGPU_QueryDeviceLimits(WebGPURenderer *renderer)
{
    renderer->limits = (WGPULimits)WGPU_LIMITS_INIT;

    if (wgpuDeviceGetLimits(renderer->device, &renderer->limits) != WGPUStatus_Success) {
        WEBGPU_SetStringError("DeviceGetLimits failed");
        return false;
    }
    if (renderer->limits.minUniformBufferOffsetAlignment == WGPU_LIMIT_U32_UNDEFINED ||
        !WEBGPU_IsPowerOfTwo32(renderer->limits.minUniformBufferOffsetAlignment)) {
        WEBGPU_SetStringError("invalid minUniformBufferOffsetAlignment");
        return false;
    }
    if (renderer->limits.maxUniformBufferBindingSize == WGPU_LIMIT_U64_UNDEFINED ||
        renderer->limits.maxUniformBufferBindingSize < UNIFORM_BUFFER_SIZE) {
        WEBGPU_SetStringError("maxUniformBufferBindingSize is too small for SDL_GPU uniforms");
        return false;
    }
    if (renderer->limits.minStorageBufferOffsetAlignment == WGPU_LIMIT_U32_UNDEFINED ||
        !WEBGPU_IsPowerOfTwo32(renderer->limits.minStorageBufferOffsetAlignment)) {
        WEBGPU_SetStringError("invalid minStorageBufferOffsetAlignment");
        return false;
    }
    if (renderer->limits.maxStorageBufferBindingSize == WGPU_LIMIT_U64_UNDEFINED ||
        renderer->limits.maxStorageBufferBindingSize == 0) {
        WEBGPU_SetStringError("invalid maxStorageBufferBindingSize");
        return false;
    }
    if (renderer->limits.maxDynamicUniformBuffersPerPipelineLayout == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxDynamicUniformBuffersPerPipelineLayout < MAX_UNIFORM_BUFFERS_PER_STAGE * 2) {
        WEBGPU_SetStringError("maxDynamicUniformBuffersPerPipelineLayout is too small for SDL_GPU uniforms");
        return false;
    }
    if (renderer->limits.maxVertexBuffers == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxVertexBuffers == 0) {
        WEBGPU_SetStringError("invalid maxVertexBuffers");
        return false;
    }
    if (renderer->limits.maxVertexAttributes == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxVertexAttributes == 0) {
        WEBGPU_SetStringError("invalid maxVertexAttributes");
        return false;
    }
    if (renderer->limits.maxVertexBufferArrayStride == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxVertexBufferArrayStride == 0) {
        WEBGPU_SetStringError("invalid maxVertexBufferArrayStride");
        return false;
    }
    if (renderer->limits.maxBindGroupsPlusVertexBuffers == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxBindGroupsPlusVertexBuffers == 0) {
        WEBGPU_SetStringError("invalid maxBindGroupsPlusVertexBuffers");
        return false;
    }
    if (renderer->limits.maxBindGroups == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxBindGroups == 0) {
        WEBGPU_SetStringError("invalid maxBindGroups");
        return false;
    }
    if (renderer->limits.maxColorAttachments == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxColorAttachments == 0) {
        WEBGPU_SetStringError("invalid maxColorAttachments");
        return false;
    }
    if (renderer->limits.maxTextureDimension2D == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxTextureDimension2D == 0) {
        WEBGPU_SetStringError("invalid maxTextureDimension2D");
        return false;
    }
    if (renderer->limits.maxTextureDimension3D == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxTextureDimension3D == 0) {
        WEBGPU_SetStringError("invalid maxTextureDimension3D");
        return false;
    }
    if (renderer->limits.maxTextureArrayLayers == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxTextureArrayLayers == 0) {
        WEBGPU_SetStringError("invalid maxTextureArrayLayers");
        return false;
    }
    if (renderer->limits.maxComputeWorkgroupsPerDimension == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxComputeWorkgroupsPerDimension == 0) {
        WEBGPU_SetStringError("invalid maxComputeWorkgroupsPerDimension");
        return false;
    }
    if (renderer->limits.maxStorageTexturesPerShaderStage == WGPU_LIMIT_U32_UNDEFINED ||
        renderer->limits.maxStorageTexturesPerShaderStage == 0) {
        WEBGPU_SetStringError("invalid maxStorageTexturesPerShaderStage");
        return false;
    }

    return true;
}

static void WEBGPU_SetDeviceProperties(WebGPURenderer *renderer)
{
    WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;

    SDL_SetStringProperty(renderer->props, SDL_PROP_GPU_DEVICE_DRIVER_NAME_STRING, "WebGPU");
    SDL_SetNumberProperty(renderer->props, WEBGPU_PROP_WAIT_ANY_MAX_COUNT, (Sint64)renderer->wait_any_max_count);

    if (wgpuAdapterGetInfo(renderer->adapter, &adapter_info) == WGPUStatus_Success) {
        if (!WEBGPU_SetStringViewProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING, adapter_info.description) &&
            !WEBGPU_SetStringViewProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING, adapter_info.device) &&
            !WEBGPU_SetStringViewProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING, adapter_info.vendor)) {
            SDL_SetStringProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING, "WebGPU device");
        }
        if (adapter_info.vendor.data || adapter_info.architecture.data || adapter_info.device.data || adapter_info.description.data) {
            char *vendor = WEBGPU_StringViewDup(adapter_info.vendor);
            char *architecture = WEBGPU_StringViewDup(adapter_info.architecture);
            char *device = WEBGPU_StringViewDup(adapter_info.device);
            char *description = WEBGPU_StringViewDup(adapter_info.description);
            char driver_info[256];

            (void)SDL_snprintf(
                driver_info,
                sizeof(driver_info),
                "backend=%s vendor=%s architecture=%s device=%s description=%s",
                WEBGPU_BackendTypeString(adapter_info.backendType),
                vendor ? vendor : "",
                architecture ? architecture : "",
                device ? device : "",
                description ? description : "");
            SDL_SetStringProperty(renderer->props, SDL_PROP_GPU_DEVICE_DRIVER_INFO_STRING, driver_info);

            SDL_free(vendor);
            SDL_free(architecture);
            SDL_free(device);
            SDL_free(description);
        }
        wgpuAdapterInfoFreeMembers(adapter_info);
    }

    if (!SDL_HasProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING)) {
        SDL_SetStringProperty(renderer->props, SDL_PROP_GPU_DEVICE_NAME_STRING, "WebGPU device");
    }
}

static SDL_GPUDevice *WEBGPU_CreateDevice(bool debug_mode, bool prefer_low_power, SDL_PropertiesID props)
{
    WGPUInstanceFeatureName instance_feature = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstanceLimits instance_limits = WGPU_INSTANCE_LIMITS_INIT;
    WGPUInstanceLimits reported_instance_limits = WGPU_INSTANCE_LIMITS_INIT;
    WGPURequestAdapterOptions adapter_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    WGPURequestAdapterCallbackInfo adapter_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
    WGPULimits adapter_limits = WGPU_LIMITS_INIT;
    WGPULimits required_limits = WGPU_LIMITS_INIT;
    WGPUFeatureName required_features[16];
    size_t required_feature_count = 0;
    bool has_required_limits = false;
    WGPURequestDeviceCallbackInfo device_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    WebGPUAdapterRequest adapter_request;
    WebGPUDeviceRequest device_request;
    WebGPURenderer *renderer;
    size_t wait_any_max_count = WEBGPU_WAIT_ANY_MAX_FENCES;
    bool enable_depth_clamping = SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_DEPTH_CLAMPING_BOOLEAN, true);
    SDL_GPUDevice *result;

    (void)debug_mode;
    renderer = (WebGPURenderer *)SDL_calloc(1, sizeof(*renderer));
    if (!renderer) {
        return NULL;
    }
    renderer->allowed_frames_in_flight = 2;
    renderer->fence_lock = SDL_CreateMutex();
    if (!renderer->fence_lock) {
        SDL_free(renderer);
        return NULL;
    }
    renderer->resource_lock = SDL_CreateMutex();
    if (!renderer->resource_lock) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }
    renderer->error_lock = SDL_CreateMutex();
    if (!renderer->error_lock) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }
    renderer->device_lost_callback_state = (WebGPUDeviceLostCallbackState *)SDL_calloc(1, sizeof(*renderer->device_lost_callback_state));
    if (!renderer->device_lost_callback_state) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }
    renderer->device_lost_callback_state->renderer = renderer;

    if (wgpuGetInstanceLimits(&reported_instance_limits) == WGPUStatus_Success) {
        if (reported_instance_limits.timedWaitAnyMaxCount == 0) {
            WEBGPU_DestroyRenderer(renderer);
            SDL_SetError("WebGPU TimedWaitAny reports zero max count");
            return NULL;
        }
        if (reported_instance_limits.timedWaitAnyMaxCount < wait_any_max_count) {
            wait_any_max_count = reported_instance_limits.timedWaitAnyMaxCount;
        }
    }

    instance_desc.requiredFeatureCount = 1;
    instance_desc.requiredFeatures = &instance_feature;
    instance_limits.timedWaitAnyMaxCount = wait_any_max_count;
    instance_desc.requiredLimits = &instance_limits;
    renderer->wait_any_max_count = instance_limits.timedWaitAnyMaxCount;
    renderer->instance = wgpuCreateInstance(&instance_desc);
    if (!renderer->instance) {
        WEBGPU_DestroyRenderer(renderer);
        SDL_SetError("wgpuCreateInstance failed");
        return NULL;
    }
    renderer->supports_readonly_and_readwrite_storage_textures = wgpuInstanceHasWGSLLanguageFeature(
        renderer->instance,
        WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures);

    SDL_zero(adapter_request);
    adapter_options.powerPreference = prefer_low_power ? WGPUPowerPreference_LowPower : WGPUPowerPreference_HighPerformance;
    adapter_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    adapter_callback.callback = WEBGPU_OnAdapter;
    adapter_callback.userdata1 = &adapter_request;
    if (!WEBGPU_WaitForFuture(renderer->instance, wgpuInstanceRequestAdapter(renderer->instance, &adapter_options, adapter_callback)) ||
        adapter_request.status != WGPURequestAdapterStatus_Success ||
        !adapter_request.adapter) {
        WEBGPU_DestroyRenderer(renderer);
        SDL_SetError("wgpuInstanceRequestAdapter failed");
        return NULL;
    }
    renderer->adapter = adapter_request.adapter;
    if (wgpuAdapterGetLimits(renderer->adapter, &adapter_limits) != WGPUStatus_Success) {
        WEBGPU_DestroyRenderer(renderer);
        SDL_SetError("wgpuAdapterGetLimits failed");
        return NULL;
    }
    if (adapter_limits.maxStorageTexturesPerShaderStage != WGPU_LIMIT_U32_UNDEFINED && adapter_limits.maxStorageTexturesPerShaderStage > 0) {
        required_limits.maxStorageTexturesPerShaderStage = SDL_min(adapter_limits.maxStorageTexturesPerShaderStage, (Uint32)MAX_STORAGE_TEXTURES_PER_STAGE);
        has_required_limits = true;
    }

    renderer->supports_depth32float_stencil8 = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_Depth32FloatStencil8);
    renderer->supports_rg11b10ufloat_renderable = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_RG11B10UfloatRenderable);
    renderer->supports_texture_formats_tier1 = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureFormatsTier1);
    renderer->supports_texture_formats_tier2 = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureFormatsTier2);
    renderer->supports_unorm16_texture_formats = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_Unorm16TextureFormats);
    renderer->supports_float32_filterable = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_Float32Filterable);
    renderer->supports_float32_blendable = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_Float32Blendable);
    renderer->supports_depth_clip_control = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_DepthClipControl);
    renderer->supports_texture_compression_bc = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureCompressionBC);
    renderer->supports_texture_compression_astc = wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureCompressionASTC);
    renderer->supports_texture_compression_bc_sliced_3d =
        renderer->supports_texture_compression_bc &&
        wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureCompressionBCSliced3D);
    renderer->supports_texture_compression_astc_sliced_3d =
        renderer->supports_texture_compression_astc &&
        wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_TextureCompressionASTCSliced3D);

    if (SDL_HasProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_CLIP_DISTANCE_BOOLEAN) &&
        SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_CLIP_DISTANCE_BOOLEAN, false)) {
        if (!wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_ClipDistances)) {
            WEBGPU_DestroyRenderer(renderer);
            SDL_SetError("WebGPU adapter does not support clip-distances");
            return NULL;
        }
        required_features[required_feature_count] = WGPUFeatureName_ClipDistances;
        required_feature_count += 1;
    }
    if (enable_depth_clamping) {
        if (!renderer->supports_depth_clip_control) {
            WEBGPU_DestroyRenderer(renderer);
            SDL_SetError("WebGPU adapter does not support depth-clip-control");
            return NULL;
        }
        required_features[required_feature_count] = WGPUFeatureName_DepthClipControl;
        required_feature_count += 1;
    }
    if (SDL_GetBooleanProperty(props, SDL_PROP_GPU_DEVICE_CREATE_FEATURE_INDIRECT_DRAW_FIRST_INSTANCE_BOOLEAN, true)) {
        if (!wgpuAdapterHasFeature(renderer->adapter, WGPUFeatureName_IndirectFirstInstance)) {
            WEBGPU_DestroyRenderer(renderer);
            SDL_SetError("WebGPU adapter does not support indirect-first-instance");
            return NULL;
        }
        required_features[required_feature_count] = WGPUFeatureName_IndirectFirstInstance;
        required_feature_count += 1;
    }
    if (renderer->supports_depth32float_stencil8) {
        required_features[required_feature_count] = WGPUFeatureName_Depth32FloatStencil8;
        required_feature_count += 1;
    }
    if (renderer->supports_rg11b10ufloat_renderable) {
        required_features[required_feature_count] = WGPUFeatureName_RG11B10UfloatRenderable;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_formats_tier1) {
        required_features[required_feature_count] = WGPUFeatureName_TextureFormatsTier1;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_formats_tier2) {
        required_features[required_feature_count] = WGPUFeatureName_TextureFormatsTier2;
        required_feature_count += 1;
    }
    if (renderer->supports_unorm16_texture_formats) {
        required_features[required_feature_count] = WGPUFeatureName_Unorm16TextureFormats;
        required_feature_count += 1;
    }
    if (renderer->supports_float32_filterable) {
        required_features[required_feature_count] = WGPUFeatureName_Float32Filterable;
        required_feature_count += 1;
    }
    if (renderer->supports_float32_blendable) {
        required_features[required_feature_count] = WGPUFeatureName_Float32Blendable;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_compression_bc) {
        required_features[required_feature_count] = WGPUFeatureName_TextureCompressionBC;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_compression_bc_sliced_3d) {
        required_features[required_feature_count] = WGPUFeatureName_TextureCompressionBCSliced3D;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_compression_astc) {
        required_features[required_feature_count] = WGPUFeatureName_TextureCompressionASTC;
        required_feature_count += 1;
    }
    if (renderer->supports_texture_compression_astc_sliced_3d) {
        required_features[required_feature_count] = WGPUFeatureName_TextureCompressionASTCSliced3D;
        required_feature_count += 1;
    }

    SDL_zero(device_request);
    device_desc.requiredFeatureCount = required_feature_count;
    device_desc.requiredFeatures = required_feature_count > 0 ? required_features : NULL;
    device_desc.requiredLimits = has_required_limits ? &required_limits : NULL;
    device_desc.uncapturedErrorCallbackInfo.callback = WEBGPU_OnUncapturedError;
    device_desc.uncapturedErrorCallbackInfo.userdata1 = renderer;
    device_desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    device_desc.deviceLostCallbackInfo.callback = WEBGPU_OnDeviceLost;
    device_desc.deviceLostCallbackInfo.userdata1 = renderer->device_lost_callback_state;
    device_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    device_callback.callback = WEBGPU_OnDevice;
    device_callback.userdata1 = &device_request;
    if (!WEBGPU_WaitForFuture(renderer->instance, wgpuAdapterRequestDevice(renderer->adapter, &device_desc, device_callback)) ||
        device_request.status != WGPURequestDeviceStatus_Success ||
        !device_request.device) {
        WEBGPU_DestroyRenderer(renderer);
        SDL_SetError("wgpuAdapterRequestDevice failed");
        return NULL;
    }
    renderer->device = device_request.device;
    renderer->device_lost_callback_state->renderer_owned = true;
    renderer->device_ready = true;
    renderer->device_lost_future = wgpuDeviceGetLostFuture(renderer->device);
    if (renderer->device_lost_future.id == 0) {
        WEBGPU_DestroyRenderer(renderer);
        SDL_SetError("wgpuDeviceGetLostFuture failed");
        return NULL;
    }
    renderer->device_lost_future_pending = true;
    renderer->queue = wgpuDeviceGetQueue(renderer->device);
    renderer->supports_core_features_and_limits = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_CoreFeaturesAndLimits);
    renderer->supports_rg11b10ufloat_renderable = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_RG11B10UfloatRenderable);
    renderer->supports_texture_formats_tier1 = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureFormatsTier1);
    renderer->supports_texture_formats_tier2 = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureFormatsTier2);
    renderer->supports_unorm16_texture_formats = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_Unorm16TextureFormats);
    renderer->supports_float32_filterable = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_Float32Filterable);
    renderer->supports_float32_blendable = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_Float32Blendable);
    renderer->supports_depth_clip_control = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_DepthClipControl);
    renderer->supports_texture_compression_bc = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureCompressionBC);
    renderer->supports_texture_compression_astc = wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureCompressionASTC);
    renderer->supports_texture_compression_bc_sliced_3d =
        renderer->supports_texture_compression_bc &&
        wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureCompressionBCSliced3D);
    renderer->supports_texture_compression_astc_sliced_3d =
        renderer->supports_texture_compression_astc &&
        wgpuDeviceHasFeature(renderer->device, WGPUFeatureName_TextureCompressionASTCSliced3D);
    if (!WEBGPU_QueryDeviceLimits(renderer)) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }

    renderer->props = SDL_CreateProperties();
    if (!renderer->props) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }
    WEBGPU_SetDeviceProperties(renderer);

    result = (SDL_GPUDevice *)SDL_calloc(1, sizeof(SDL_GPUDevice));
    if (!result) {
        WEBGPU_DestroyRenderer(renderer);
        return NULL;
    }

    ASSIGN_DRIVER(WEBGPU)
    result->driverData = (SDL_GPURenderer *)renderer;
    result->shader_formats = SDL_GPU_SHADERFORMAT_WGSL;

    if (!WEBGPU_CreateBlitResources(result)) {
        WEBGPU_DestroyDevice(result);
        return NULL;
    }

    return result;
}

SDL_GPUBootstrap WebGPUDriver = {
    "webgpu",
    WEBGPU_PrepareDriver,
    WEBGPU_CreateDevice
};

#endif // HAVE_GPU_WEBGPU

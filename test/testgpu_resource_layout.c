/*
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely.
*/

/* Tests for SDL_GPU shader resource layout facts. */

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_test.h>

static bool ErrorContains(const char *expected, const char *context)
{
    const char *error = SDL_GetError();

    if (!error || !SDL_strstr(error, expected)) {
        SDL_Log("%s failed with unexpected error: %s", context, error && *error ? error : "<empty>");
        return false;
    }

    return true;
}

static SDL_GPUShaderFormat GetAnySupportedShaderFormat(SDL_GPUDevice *device)
{
    const SDL_GPUShaderFormat formats = SDL_GetGPUShaderFormats(device);
    const SDL_GPUShaderFormat preferred_formats[] = {
        SDL_GPU_SHADERFORMAT_SPIRV,
        SDL_GPU_SHADERFORMAT_DXIL,
        SDL_GPU_SHADERFORMAT_DXBC,
        SDL_GPU_SHADERFORMAT_MSL,
        SDL_GPU_SHADERFORMAT_METALLIB,
        SDL_GPU_SHADERFORMAT_WGSL,
        SDL_GPU_SHADERFORMAT_PRIVATE,
    };
    int i;

    for (i = 0; i < SDL_arraysize(preferred_formats); i += 1) {
        if (formats & preferred_formats[i]) {
            return preferred_formats[i];
        }
    }

    return SDL_GPU_SHADERFORMAT_INVALID;
}

static bool ExpectShaderWithResourceLayoutFailure(
    SDL_GPUDevice *device,
    const SDL_GPUShaderResourceLayout *layout,
    const char *expected_error,
    const char *context)
{
    static const Uint8 shader_code[] = { 0 };
    SDL_GPUShaderWithResourceLayoutCreateInfo shader_info;
    SDL_GPUShader *shader;

    SDL_zero(shader_info);
    shader_info.code = shader_code;
    shader_info.code_size = sizeof(shader_code);
    shader_info.entrypoint = "main";
    shader_info.format = GetAnySupportedShaderFormat(device);
    shader_info.resource_layout = layout;

    SDL_ClearError();
    shader = SDL_CreateGPUShaderWithResourceLayout(device, &shader_info);
    if (shader) {
        SDL_ReleaseGPUShader(device, shader);
        SDL_Log("%s unexpectedly created a shader", context);
        return false;
    }

    return ErrorContains(expected_error, context);
}

static bool ExpectComputePipelineWithResourceLayoutFailure(
    SDL_GPUDevice *device,
    const SDL_GPUComputePipelineResourceLayout *layout,
    const char *expected_error,
    const char *context)
{
    static const Uint8 shader_code[] = { 0 };
    SDL_GPUComputePipelineWithResourceLayoutCreateInfo pipeline_info;
    SDL_GPUComputePipeline *pipeline;

    SDL_zero(pipeline_info);
    pipeline_info.code = shader_code;
    pipeline_info.code_size = sizeof(shader_code);
    pipeline_info.entrypoint = "main";
    pipeline_info.format = GetAnySupportedShaderFormat(device);
    pipeline_info.resource_layout = layout;
    pipeline_info.threadcount_x = 1;
    pipeline_info.threadcount_y = 1;
    pipeline_info.threadcount_z = 1;

    SDL_ClearError();
    pipeline = SDL_CreateGPUComputePipelineWithResourceLayout(device, &pipeline_info);
    if (pipeline) {
        SDL_ReleaseGPUComputePipeline(device, pipeline);
        SDL_Log("%s unexpectedly created a compute pipeline", context);
        return false;
    }

    return ErrorContains(expected_error, context);
}

static bool RunResourceLayoutChecks(SDL_GPUDevice *device)
{
    SDL_GPUSampledTextureSlotDescription sampled_texture_slot;
    SDL_GPUShaderResourceLayout shader_layout;
    SDL_GPUComputePipelineResourceLayout compute_layout;
    SDL_GPUStorageTextureSlotDescription compute_storage_slot;

    SDL_zero(sampled_texture_slot);
    sampled_texture_slot.texture_type = SDL_GPU_TEXTURETYPE_3D;
    sampled_texture_slot.sample_type = SDL_GPU_SHADERTEXTURESAMPLETYPE_DEPTH;
    sampled_texture_slot.sampler_type = SDL_GPU_SHADERSAMPLERTYPE_COMPARISON;

    SDL_zero(shader_layout);
    shader_layout.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
    shader_layout.num_samplers = 1;
    shader_layout.sampled_texture_slots = &sampled_texture_slot;
    if (!ExpectShaderWithResourceLayoutFailure(
            device,
            &shader_layout,
            "depth comparison layout only supports 2D, 2D-array, cube, and cube-array textures",
            "depth 3D comparison shader resource layout")) {
        return false;
    }

    sampled_texture_slot.texture_type = SDL_GPU_TEXTURETYPE_2D;
    sampled_texture_slot.sample_type = SDL_GPU_SHADERTEXTURESAMPLETYPE_FILTERABLE_FLOAT;
    sampled_texture_slot.sampler_type = SDL_GPU_SHADERSAMPLERTYPE_NONFILTERING;
    shader_layout.num_samplers = 0;
    if (!ExpectShaderWithResourceLayoutFailure(
            device,
            &shader_layout,
            "shader sampled texture layout array must be NULL when the resource count is zero",
            "sampled shader resource layout non-NULL array with zero count")) {
        return false;
    }

    shader_layout.num_samplers = 1;
    shader_layout.stage = (SDL_GPUShaderStage)999;
    if (!ExpectShaderWithResourceLayoutFailure(
            device,
            &shader_layout,
            "shader resource layout stage is invalid",
            "sampled shader resource layout invalid stage")) {
        return false;
    }

    SDL_zero(compute_storage_slot);
    compute_storage_slot.texture_type = SDL_GPU_TEXTURETYPE_2D;
    compute_storage_slot.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    compute_storage_slot.access = SDL_GPU_STORAGETEXTUREACCESS_READ_ONLY;

    SDL_zero(compute_layout);
    compute_layout.num_readwrite_storage_textures = 1;
    compute_layout.readwrite_storage_texture_slots = &compute_storage_slot;
    if (!ExpectComputePipelineWithResourceLayoutFailure(
            device,
            &compute_layout,
            "compute read-write storage texture layout does not accept read-only access",
            "compute read-write storage layout access mismatch")) {
        return false;
    }

    compute_storage_slot.texture_type = SDL_GPU_TEXTURETYPE_2D_ARRAY;
    compute_storage_slot.access = SDL_GPU_STORAGETEXTUREACCESS_WRITE_ONLY;
    if (!ExpectComputePipelineWithResourceLayoutFailure(
            device,
            &compute_layout,
            "compute read-write storage texture layout only supports 2D and 3D textures",
            "compute read-write storage layout 2D-array texture")) {
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{
    const char *gpu_driver = NULL;
    bool debug = false;
    SDLTest_CommonState *state;
    SDL_GPUDevice *device;
    bool ok;
    int i;

    state = SDLTest_CommonCreateState(argv, 0);
    if (!state) {
        return 1;
    }

    for (i = 1; i < argc;) {
        int consumed = SDLTest_CommonArg(state, i);
        if (consumed == 0) {
            consumed = -1;
            if (SDL_strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
                gpu_driver = argv[i + 1];
                consumed = 2;
            } else if (SDL_strcmp(argv[i], "--debug") == 0) {
                debug = true;
                consumed = 1;
            }
        }
        if (consumed < 0) {
            static const char *options[] = {
                "[--gpu DRIVER]",
                "[--debug]",
                NULL
            };
            SDLTest_CommonLogUsage(state, argv[0], options);
            SDLTest_CommonDestroyState(state);
            return 1;
        }
        i += consumed;
    }

    SDL_SetLogPriorities(SDL_LOG_PRIORITY_VERBOSE);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        SDLTest_CommonQuit(state);
        return 1;
    }

    device = SDL_CreateGPUDevice(
        SDL_GPU_SHADERFORMAT_DXBC |
            SDL_GPU_SHADERFORMAT_DXIL |
            SDL_GPU_SHADERFORMAT_SPIRV |
            SDL_GPU_SHADERFORMAT_MSL |
            SDL_GPU_SHADERFORMAT_METALLIB |
            SDL_GPU_SHADERFORMAT_WGSL,
        debug,
        gpu_driver);
    if (!device) {
        SDL_Log("No SDL_GPU device available for resource layout test: %s", SDL_GetError());
        SDLTest_CommonQuit(state);
        return gpu_driver ? 1 : 0;
    }

    ok = RunResourceLayoutChecks(device);

    SDL_DestroyGPUDevice(device);
    SDLTest_CommonQuit(state);

    return ok ? 0 : 1;
}

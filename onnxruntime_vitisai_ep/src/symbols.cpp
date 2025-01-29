/*
 *  Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc. All rights reserved.
 *  Licensed under the MIT License.
 */
typedef void* void_ptr_t;
extern "C" void* vart_dummy_runner_hook;
void_ptr_t reserved_symbols[] = {
    vart_dummy_runner_hook,
};

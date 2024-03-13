// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#pragma once

#include "synthetic-stream.h"

#ifdef RS2_USE_CUDA
#include "cuda/cuda-conversion.cuh"
#endif


namespace librealsense
{
    class LRS_EXTENSION_API y8i_to_y8y8 : public interleaved_functional_processing_block
    {
    public:
        y8i_to_y8y8(int left_idx = 1, int right_idx = 2);
        ~y8i_to_y8y8();

    protected:
        #ifdef RS2_USE_CUDA
        cudaStream_t stream_;
        #endif
        y8i_to_y8y8(const char* name, int left_idx, int right_idx);
        void process_function(byte * const dest[], const byte * source, int width, int height, int actual_size, int input_size) override;
    };
}

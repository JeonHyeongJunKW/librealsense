#pragma once
#ifdef RS2_USE_CUDA

#include <iostream>
#include <cuda_runtime.h>

#include "../../../include/librealsense2/rs.h"
#include <memory>
#include <stdint.h>

namespace librealsense
{
    class align_cuda_helper
    {
    public:
        align_cuda_helper()
        {
            auto result = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
            std::cout << "make : "<< cudaGetErrorName(result) << std::endl;
            result = cudaStreamSynchronize(stream_);
            std::cout << "sync : "<< cudaGetErrorName(result) << std::endl;
        }
        ~align_cuda_helper();

        void align_other_to_depth(unsigned char* h_aligned_out, const uint16_t* h_depth_in,
            float depth_scale, const rs2_intrinsics& h_depth_intrin, const rs2_extrinsics& h_depth_to_other,
            const rs2_intrinsics& h_other_intrin, const unsigned char* h_other_in, rs2_format other_format, int other_bytes_per_pixel);

        void align_depth_to_other(unsigned char* h_aligned_out, const uint16_t* h_depth_in,
            float depth_scale, const rs2_intrinsics& h_depth_intrin, const rs2_extrinsics& h_depth_to_other,
            const rs2_intrinsics& h_other_intrin);

    private:
        cudaStream_t stream_;
        uint16_t *       _d_depth_in = nullptr;
        unsigned char *  _d_other_in = nullptr;
        unsigned char *  _d_aligned_out = nullptr;
        int2 *           _d_pixel_map = nullptr;

        rs2_intrinsics * _d_other_intrinsics = nullptr;
        rs2_intrinsics * _d_depth_intrinsics = nullptr;
        rs2_extrinsics * _d_depth_other_extrinsics = nullptr;
    };
}
#endif // RS2_USE_CUDA

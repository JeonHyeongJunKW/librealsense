// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2019 Intel Corporation. All Rights Reserved.
#include "proc/cuda/cuda-pointcloud.h"

namespace librealsense
{
    pointcloud_cuda::pointcloud_cuda() : pointcloud("Pointcloud (CUDA)")
    {
        #ifdef RS2_USE_CUDA
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        cudaStreamSynchronize(stream_);
        #endif
    }

    const float3 * pointcloud_cuda::depth_to_points(
        rs2::points output,
        const rs2_intrinsics &depth_intrinsics,
        const rs2::depth_frame& depth_frame)
    {
        auto image = output.get_vertices();
        auto depth_data = (uint16_t*)depth_frame.get_data();
        auto depth_scale = depth_frame.get_units();
#ifdef RS2_USE_CUDA
        rscuda::deproject_depth_cuda(stream_, (float*)image, depth_intrinsics, depth_data, depth_scale);
#endif
        return (float3*)image;
    }
}

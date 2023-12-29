// part of the code is adapted from https://github.com/mit-han-lab/llm-awq/blob/main/awq/kernels/csrc/quantization/gemv_cuda.cu
#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define PACK_FACTOR 8


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}


/*
Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC];
  output: vector of shape [batch_size, OC];
*/

template<int BS>
__global__ void gemm_small_bs_kernel(
  const float4* _inputs, const float4* _weight, half* _outputs, const int IC, const int OC){
    const int oc_idx = 4 * blockIdx.y + threadIdx.y;
    const float4* weight = _weight + oc_idx * IC / PACK_FACTOR + threadIdx.x;

    half packed_weight[PACK_FACTOR];
    half packed_inputs[PACK_FACTOR * BS];
    float packed_psum[PACK_FACTOR * BS] = {0.};

    for (int ic_0 = 0; ic_0 < (IC / 256); ic_0++) {
      #pragma unroll
      for (int b = 0; b < BS; b++) {
        ((float4*)packed_inputs)[b] = \
            *(_inputs + b * IC / PACK_FACTOR + threadIdx.x + ic_0 * 32);
      }
      *((float4*)packed_weight) = *(weight + ic_0 * 32);
      #pragma unroll
      for (int ic_1 = 0; ic_1 < PACK_FACTOR * BS; ic_1++) {
        packed_psum[ic_1] += __half2float(packed_inputs[ic_1]) * \
                             __half2float(packed_weight[ic_1 % PACK_FACTOR]);
      }
    }

    float psum[BS] = {0.};

    #pragma unroll
    for (int ic_1 = 0; ic_1 < PACK_FACTOR * BS; ic_1++) {
      psum[ic_1 / PACK_FACTOR] += packed_psum[ic_1];
    }

    #pragma unroll
    for (int b = 0; b < BS; b++) {
      psum[b] = warp_reduce_sum(psum[b]);
    }

    if (threadIdx.x == 0) {
      #pragma unroll
      for (int b = 0; b < BS; b++) {
        _outputs[b * OC + oc_idx] = __float2half(psum[b]);
      }
    }
}


template<int BS>
__global__ void gemm_small_bs_kernel_tmp(
  const float4* _inputs, const float4* _weight, half* _outputs, const int IC, const int OC){
    const int oc_idx = 4 * blockIdx.y + threadIdx.y;
    const float4* weight = _weight + oc_idx * IC / PACK_FACTOR + threadIdx.x;

    half packed_weight[PACK_FACTOR];
    half packed_inputs[PACK_FACTOR * BS];
    float packed_psum[PACK_FACTOR * BS] = {0.};

    for (int ic_0 = 0; ic_0 < (IC / 256); ic_0++) {
      #pragma unroll
      for (int b = 0; b < BS; b++) {
        ((float4*)packed_inputs)[b] = \
            *(_inputs + b * IC / PACK_FACTOR + threadIdx.x + ic_0 * 32);
      }
      *((float4*)packed_weight) = *(weight + ic_0 * 32);
      #pragma unroll
      for (int ic_1 = 0; ic_1 < PACK_FACTOR * BS; ic_1++) {
        packed_psum[ic_1] += __half2float(packed_inputs[ic_1]) * \
                             __half2float(packed_weight[ic_1 % PACK_FACTOR]);
      }
    }

    float psum[BS] = {0.};

    #pragma unroll
    for (int ic_1 = 0; ic_1 < PACK_FACTOR * BS; ic_1++) {
      psum[ic_1 / PACK_FACTOR] += packed_psum[ic_1];
    }

    #pragma unroll
    for (int b = 0; b < BS; b++) {
      psum[b] = warp_reduce_sum(psum[b]);
    }

    half* inputs_half = (half*)_inputs;
    half* weight_half = (half*)_weight;
    for (int ic_0 = (IC / 256) * 256 + threadIdx.x; ic_0 < IC; ic_0++) {
      #pragma unroll
      for (int b = 0; b < BS; b++) {
        psum[b] += __half2float(*(inputs_half + b * IC + ic_0)) * \
                   __half2float(*(weight_half + oc_idx * IC + ic_0));
      }
    }

    if (threadIdx.x == 0) {
      #pragma unroll
      for (int b = 0; b < BS; b++) {
        _outputs[b * OC + oc_idx] = __float2half(psum[b]);
      }
    }
}


torch::Tensor gemm_small_bs(
    torch::Tensor& _in_feats,
    torch::Tensor& _kernel)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<float4*>(_kernel.data_ptr<at::Half>());

    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(0)}, options);

    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);

    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    dim3 num_blocks(1, num_out_channels / 4);
    dim3 num_threads(32, 4);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (num_out_channels % 4 != 0) {
      std::cout << "Not supported" << std::endl;
      return _out_feats;
    }

    if (num_in_channels % 256 != 0) {
      if (num_out_feats == 1) {
        gemm_small_bs_kernel_tmp<1><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 2) {
        gemm_small_bs_kernel_tmp<2><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 3) {
        gemm_small_bs_kernel_tmp<3><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 4) {
        gemm_small_bs_kernel_tmp<4><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else {
        std::cout << "Not supported." << std::endl;
      }
    } else {
      if (num_out_feats == 1) {
        gemm_small_bs_kernel<1><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 2) {
        gemm_small_bs_kernel<2><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 3) {
        gemm_small_bs_kernel<3><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 4) {
        gemm_small_bs_kernel<4><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 5) {
        gemm_small_bs_kernel<5><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 6) {
        gemm_small_bs_kernel<6><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 7) {
        gemm_small_bs_kernel<7><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else if (num_out_feats == 8) {
        gemm_small_bs_kernel<8><<<num_blocks, num_threads, 0, stream>>>(
          in_feats, kernel, out_feats,
          num_in_channels, num_out_channels
        );
      } else {
        std::cout << "Not supported." << std::endl;
      }
    }
    return _out_feats;
}


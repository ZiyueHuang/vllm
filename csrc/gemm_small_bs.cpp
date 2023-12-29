#include <torch/extension.h>

torch::Tensor gemm_small_bs(
  torch::Tensor& inputs,
  torch::Tensor& weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "gemm_small_bs",
    &gemm_small_bs,
    "GEMM kernel for small batch size");
}

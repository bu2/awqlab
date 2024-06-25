#include <torch/extension.h>

torch::Tensor gemm_forward_cuda_quick(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters);

void layernorm_forward_cuda(
    torch::Tensor _input, 
    torch::Tensor _gamma, 
    torch::Tensor _out, 
    float eps);

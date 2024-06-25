#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("layernorm_forward_cuda", &layernorm_forward_cuda, "FasterTransformer layernorm kernel");
    m.def("gemm_forward_cuda_quick", &gemm_forward_cuda_quick, "QUICK AWQ GEMM kernel.");
}

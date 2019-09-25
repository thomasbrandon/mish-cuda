#include <torch/types.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

namespace kernel {
using at::cuda::CUDA_tensor_apply2;
using at::cuda::CUDA_tensor_apply3;
using at::cuda::TensorArgType;

template <typename scalar_t>
void
mish_forward(
  torch::Tensor &output,
  const torch::Tensor &input
) {
  CUDA_tensor_apply2<scalar_t,scalar_t>(
    output, input,
    [=] __host__ __device__ (scalar_t &out, const scalar_t &inp) {
      out = inp * tanh(log(exp(inp) + 1));
    },
    TensorArgType::ReadWrite, TensorArgType::ReadOnly
  );
}

template <typename scalar_t>
void
mish_backward(
  torch::Tensor &grad_inp,
  const torch::Tensor &input,
  const torch::Tensor &grad_out
) {
  CUDA_tensor_apply3<scalar_t,scalar_t,scalar_t>(
    grad_inp, input, grad_out,
    [=] __host__ __device__ (scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
      scalar_t w, d;
      // TODO: Test performance using exp(x*inp) -> exp(inp)**x
      w = 4*(inp+1) + 4*exp(2*inp) + exp(3*inp) + exp(inp) * (4*inp+6);
      d = 2*exp(inp) + exp(2*inp) + 2;
      grad_inp = grad_out * exp(inp) * w / pow(d, 2);
    },
    TensorArgType::ReadWrite, TensorArgType::ReadOnly, TensorArgType::ReadOnly
  );
}

} // namespace kernel

void
mish_forward_cuda(
    torch::Tensor &output, const torch::Tensor &input
) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "mish_forward_cuda", [&] {
      kernel::mish_forward<scalar_t>(output, input);
  });
}

void
mish_backward_cuda(
  torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out
) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "mish_backward_cuda", [&] {
      kernel::mish_backward<scalar_t>(grad_inp, input, grad_out);
  });
}

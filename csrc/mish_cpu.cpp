#include "mish.h"
#include <torch/types.h>
#include <ATen/CPUApplyUtils.h>

namespace cpu {

template <typename scalar_t>
void
mish_forward(
  torch::Tensor &output,
  const torch::Tensor &input
) {
  at::CPU_tensor_apply2<scalar_t,scalar_t>(
    output, input,
    [=] (scalar_t &out, const scalar_t &inp) {
      mish_fwd_func(out, inp);
    }
  );
}

template <typename scalar_t>
void
mish_backward(
  torch::Tensor &grad_inp,
  const torch::Tensor &input,
  const torch::Tensor &grad_out
) {
  at::CPU_tensor_apply3<scalar_t,scalar_t,scalar_t>(
    grad_inp, input, grad_out,
    [=] (scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
      mish_bwd_func(grad_inp, inp, grad_out);
    }
  );
}

}

void
mish_forward_cpu(
    torch::Tensor &output, const torch::Tensor &input
) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "mish_forward_cpu", [&] {
      cpu::mish_forward<scalar_t>(output, input);
  });
}

void
mish_backward_cpu(
    torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out
) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_inp.scalar_type(), "mish_backward_cpu", [&] {
      cpu::mish_backward<scalar_t>(grad_inp, input, grad_out);
  });
}
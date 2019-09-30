#include "mish_kernels.h"

#include <torch/extension.h>
using namespace pybind11::literals;

//TODO: Only return intermediate when requires_grad

std::pair<torch::Tensor,torch::Tensor>
mish_forward(const torch::Tensor &input, at::optional<torch::Tensor> opt_out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  if (opt_out) {
    auto out_arg = torch::TensorArg(*opt_out, "out", 1);
    torch::checkSameType("mish_forward", input_arg, out_arg);
    torch::checkSameSize("mish_forward", input_arg, out_arg);
  }
  auto out = opt_out.value_or(torch::empty_like(input));
  auto inter = torch::empty_like(input);
  switch (input.device().type()) {
    case c10::kCUDA:
      mish_forward_cuda(out, inter, input);
      break;
    case c10::kCPU:
      mish_forward_cpu(out, inter, input);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", input.device().type());
  }
  return {out,inter};
}


torch::Tensor
mish_backward(const torch::Tensor &input, const torch::Tensor &inter, torch::Tensor &grad_out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  auto inter_arg = torch::TensorArg(grad_out, "inter", 1);
  auto grad_out_arg = torch::TensorArg(grad_out, "grad_out", 2);
  torch::checkAllSameType("mish_backward", {input_arg, inter_arg, grad_out_arg});

  auto grad_inp = torch::empty_like(input);
  switch (input.device().type()) {
    case c10::kCUDA:
      mish_backward_cuda(grad_inp, input, inter, grad_out);
      break;
    case c10::kCPU:
      mish_backward_cpu(grad_inp, input, inter, grad_out);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", input.device().type());
  }
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mish_forward", &mish_forward, "Mish activation forward", "input"_a, "out"_a = nullptr);
  m.def("mish_backward", &mish_backward, "Mish activation backward", "input"_a, "inter"_a, "grad_out"_a);
}

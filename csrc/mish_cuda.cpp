
#include <torch/extension.h>
using namespace pybind11::literals;

// Forward declaration of kernels
void mish_forward_cuda(torch::Tensor &output, const torch::Tensor &input);
void mish_backward_cuda(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out);
void mish_forward_cpu(torch::Tensor &output, const torch::Tensor &input);
void mish_backward_cpu(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out);

torch::Tensor
mish_forward(const torch::Tensor &input, const at::optional<torch::Tensor> out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  if (out) {
    auto out_arg = torch::TensorArg(*out, "out", 1);
    torch::checkSameType("mish_forward", input_arg, out_arg);
    torch::checkSameSize("mish_forward", input_arg, out_arg);
  }
  auto o = out.value_or(torch::empty_like(input));
  switch (input.device().type()) {
    case c10::kCUDA:
      mish_forward_cuda(o, input);
      break;
    case c10::kCPU:
      mish_forward_cpu(o, input);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", input.device().type());
  }
  return o;
}

torch::Tensor
mish_backward(const torch::Tensor &input, const torch::Tensor &grad_out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  auto grad_out_arg = torch::TensorArg(grad_out, "grad_out", 1);
  torch::checkSameType("mish_backward", input_arg, grad_out_arg);

  auto grad_inp = torch::empty_like(input);
  switch (input.device().type()) {
    case c10::kCUDA:
      mish_backward_cuda(grad_inp, input, grad_out);
      break;
    case c10::kCPU:
      mish_backward_cpu(grad_inp, input, grad_out);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", input.device().type());
  }
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mish_forward", &mish_forward, "Mish activation forward", "input"_a, "out"_a = nullptr);
  m.def("mish_backward", &mish_backward, "Mish activation backward", "input"_a, "grad_out"_a);
}

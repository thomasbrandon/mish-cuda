
#include <torch/extension.h>

using namespace pybind11::literals;

// Forward declaration of kernel
void mish_backward_cuda(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out);
void mish_forward_cuda(torch::Tensor &output, const torch::Tensor &input);

torch::Tensor
mish_forward(const torch::Tensor &input, const at::optional<torch::Tensor> out) {
  torch::checkBackend("mish_forward", input, at::Backend::CUDA);
  auto input_arg = torch::TensorArg(input, "input", 0);
  if (out) {
    auto out_arg = torch::TensorArg(*out, "out", 1);
    torch::checkSameGPU("mish_forward", input_arg, out_arg);
    torch::checkSameType("mish_forward", input_arg, out_arg);
    torch::checkSameSize("mish_forward", input_arg, out_arg);
  }
  auto o = out.value_or(torch::empty_like(input));
  mish_forward_cuda(o, input);
  return o;
}


torch::Tensor
mish_backward(const torch::Tensor &input, const torch::Tensor &grad_out) {
  torch::checkDeviceType("mish_backward", {input, grad_out}, torch::kCUDA);
  auto input_arg = torch::TensorArg(input, "input", 0);
  auto grad_out_arg = torch::TensorArg(grad_out, "grad_out", 1);
  torch::checkSameGPU("mish_backward", input_arg, grad_out_arg);
  torch::checkSameType("mish_backward", input_arg, grad_out_arg);

  torch::Tensor grad_inp = torch::empty_like(input);
  mish_backward_cuda(grad_inp, input, grad_out);
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mish_forward_cuda", &mish_forward, "Mish forward (CUDA)", "input"_a, "out"_a = nullptr);
  m.def("mish_backward_cuda", &mish_backward, "Mish backward (CUDA)", "input"_a, "grad_out"_a);
}

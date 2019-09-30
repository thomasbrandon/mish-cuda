#pragma once

#include <torch/types.h>

void mish_forward_cuda(torch::Tensor &output, torch::Tensor &inter, const torch::Tensor &input);
void mish_backward_cuda(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &inter, const torch::Tensor &grad_out);
void mish_forward_cpu(torch::Tensor &output, torch::Tensor &inter, const torch::Tensor &input);
void mish_backward_cpu(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &inter, const torch::Tensor &grad_out);
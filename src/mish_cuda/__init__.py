ALL = ['MishCuda','MishCudaFunction','mish_forward_cuda','mish_backward_cuda']

import torch # Must import torch before C extension
from ._C import mish_forward_cuda, mish_backward_cuda

class MishCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return mish_forward_cuda(inp)
    
    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        return mish_backward_cuda(inp, grad_out)

class MishCuda(torch.nn.Module):
    def forward(self, inp): return MishCudaFunction.apply(inp)

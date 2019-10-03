# Mish-Cuda: Self Regularized Non-Monotonic Activation Function

This is a PyTorch CUDA implementation of the Mish activation by Diganta Misra (https://github.com/digantamisra98/).

## Installation
It is currently distributed as a source only PyTorch extension. So you need a propely set up toolchain and CUDA compilers to install.
1) _Toolchain_ - In conda the `cxx_linux-64` package provides an appropriate toolchain. However there can still be compatbility issues with this depending on system. You can also try with the system toolchian.
2) _CUDA Toolkit_ - The [nVidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is required in addition to drivers to provide needed headers and tools. Get the appropriate version for your Linux distro from nVidia or check for distro specific instructions otherwise.

_It is important your CUDA Toolkit matches the version PyTorch is built for or errors can occur. Currently PyTorch builds for v10.0 and v9.2._

## Performance
The CUDA implementation seems to mirror the learning perfomance of the original implementation and no stability issues have been observed. In terms of speed of the function it is fairly comparable with other PyTorch activation functions and significantly faster than the pure PyTorch implementation:
```
Profiling over 100 runs after 10 warmup runs.
Profiling on GeForce RTX 2070
Testing on torch.float16:
 relu_fwd:      223.7µs ± 1.026µs (221.6µs - 229.2µs)
 relu_bwd:      312.1µs ± 2.308µs (307.8µs - 317.4µs)
 softplus_fwd:  342.2µs ± 38.08µs (282.4µs - 370.6µs)
 softplus_bwd:  488.5µs ± 53.75µs (406.0µs - 528.4µs)
 mish_pt_fwd:   658.8µs ± 1.467µs (655.9µs - 661.9µs)
 mish_pt_bwd:   1.135ms ± 4.785µs (1.127ms - 1.145ms)
 mish_cuda_fwd: 267.3µs ± 1.852µs (264.5µs - 274.2µs)
 mish_cuda_bwd: 345.6µs ± 1.875µs (341.9µs - 349.8µs)

Testing on torch.float32:
 relu_fwd:      234.2µs ± 621.8ns (233.2µs - 235.7µs)
 relu_bwd:      419.3µs ± 1.238µs (417.8µs - 426.0µs)
 softplus_fwd:  255.1µs ± 753.6ns (252.4µs - 256.5µs)
 softplus_bwd:  420.2µs ± 631.4ns (418.2µs - 421.9µs)
 mish_pt_fwd:   797.4µs ± 1.094µs (795.4µs - 802.8µs)
 mish_pt_bwd:   1.689ms ± 1.222µs (1.686ms - 1.696ms)
 mish_cuda_fwd: 282.9µs ± 876.1ns (281.1µs - 287.8µs)
 mish_cuda_bwd: 496.3µs ± 1.781µs (493.6µs - 503.0µs)

Testing on torch.float64:
 relu_fwd:      450.4µs ± 879.7ns (448.8µs - 456.4µs)
 relu_bwd:      834.2µs ± 925.8ns (832.3µs - 838.8µs)
 softplus_fwd:  6.370ms ± 2.348µs (6.362ms - 6.375ms)
 softplus_bwd:  2.359ms ± 1.276µs (2.356ms - 2.365ms)
 mish_pt_fwd:   10.11ms ± 2.806µs (10.10ms - 10.12ms)
 mish_pt_bwd:   4.897ms ± 1.312µs (4.893ms - 4.901ms)
 mish_cuda_fwd: 8.989ms ± 3.646µs (8.980ms - 9.007ms)
 mish_cuda_bwd: 10.92ms ± 3.966µs (10.91ms - 10.93ms)
 ```
(Collected with `test/perftest.py -b`)

Note that double precision performance is very low. Some optimisation might be possible but this does not seem to be a common usage so is not a priority. Raise an issue if you have a use-case for it.
# Mish-Cuda: Self Regularized Non-Monotonic Activation Function

This is a PyTorch CUDA implementation of the Mish activation by Diganta Misra (https://github.com/digantamisra98/).

## Installation
It is distributed as a source only PyTorch extension. So you need a propely set up toolchain CUDA compilers to install.
1) _Toolchain_ - In conda the `cxx_linux-64` package provides an appropriate toolchain. However there can still be compatbility issues with this depending on system. You can also try with the system toolchian.
2) _CUDA Toolkit_ - The [nVidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is required in addition to drivers to provide needed headers and tools. Get the appropriate version for your Linux distro from nVidia or check for distro specific instructions otherwise.

_It is important your CUDA Toolkit matches the version PyTorch is built for or errors can occur. Currently PyTorch builds for v10.0 and v9.2._

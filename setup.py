from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_SRCS = [
    'csrc/mish_cuda.cpp',
    'csrc/mish_kernel.cu',
]

setup(
    name='mish_cuda',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            'mish_cuda._C',
            EXT_SRCS,
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--expt-extended-lambda']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
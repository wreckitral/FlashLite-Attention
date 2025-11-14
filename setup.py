import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# disable CUDA version check
os.environ['TORCH_CUDA_VERSION_CHECK'] = '0'


def _patched_check_cuda_version(*args, **kwargs):
    pass


try:
    from torch.utils import cpp_extension
    cpp_extension._check_cuda_version = _patched_check_cuda_version
except ImportError:
    pass

setup(
    name='cuda_attention',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='cuda_attention',
            sources=[
                'src/cpp/pytorch_bindings.cpp',
                'src/cuda/naive_qk.cu',
                'src/cuda/naive_softmax.cu',
                'src/cuda/naive_av.cu',
            ],
            include_dirs=['src/cuda'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)

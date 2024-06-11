import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12'


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension



setup(
    name='FasterOps',
    packages=find_packages(),
    version='0.0.0',
    ext_modules=[
        CUDAExtension(
            'faster_ops', # operator name
            ['./cpp/faster.cpp',
             './cpp/faster_cuda.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
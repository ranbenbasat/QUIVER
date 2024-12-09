from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

ext_modules = []

extension = CppExtension(
    'quiver_cpp', ['python_bindings.cpp'],
    extra_compile_args={'cxx': ['-O2']})
ext_modules.append(extension)

setup(
    name='quiver_cpp',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})

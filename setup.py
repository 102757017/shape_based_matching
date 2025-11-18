import os
import sys
import platform
import subprocess
# 导入构建依赖
import numpy
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # 直接从包中获取路径，这是最可靠的方式
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        numpy_include_dir = numpy.get_include()

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            # 明确告诉 CMake pybind11 的位置
            f'-Dpybind11_DIR={pybind11_cmake_dir}',
            # 明确告诉 CMake numpy 头文件的位置
            f'-DNUMPY_INCLUDE_DIR={numpy_include_dir}',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        if platform.system() == "Windows":
            # 使用正确的 Visual Studio 生成器
            # 在 GitHub Actions 中，使用较新版本的 Visual Studio
            cmake_args += [
                '-G', 'Visual Studio 17 2022',
                '-A', 'x64'
            ]
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=Release']
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        print("CMake arguments:", cmake_args)
        print("Build directory:", self.build_temp)
        
        # 运行 CMake 配置
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        # 运行 CMake 构建
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# setup(...) 部分保持不变
setup(
    name='shape_based_matching_py',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python wrapper for shape based matching',
    long_description='',
    ext_modules=[CMakeExtension('shape_based_matching_py')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    )

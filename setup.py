import os
import sys
import platform
import subprocess
import sysconfig
import numpy
import pybind11
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        # 确保扩展名称正确
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # 直接从包中获取路径
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        numpy_include_dir = numpy.get_include()

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-Dpybind11_DIR={pybind11_cmake_dir}',
            f'-DNUMPY_INCLUDE_DIR={numpy_include_dir}',
        ]

        # 显式传递 Python 信息
        cmake_args.append(f'-DPython_EXECUTABLE={sys.executable}')
        
        # 显式传递 OpenCV_DIR
        opencv_dir = os.environ.get('OpenCV_DIR', 'C:/opencv/opencv/build')
        cmake_args.append(f'-DOpenCV_DIR={opencv_dir}')
        print(f"Setting OpenCV_DIR: {opencv_dir}")

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        if platform.system() == "Windows":
            cmake_args += [
                '-G', 'Visual Studio 17 2022',
                '-A', 'x64'
            ]
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            build_args += ['--', '/m']
            
            # 在 Windows 上，确保生成 .pyd 文件
            cmake_args += ['-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=Release']
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        print("CMake arguments:", cmake_args)
        print("Extension name:", ext.name)
        print("Extension fullname:", self.get_ext_fullname(ext.name))
        print("Extension fullpath:", self.get_ext_fullpath(ext.name))
        
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
            
            # 构建完成后检查生成的文件
            print("Build completed. Checking generated files:")
            for root, dirs, files in os.walk(self.build_temp):
                for file in files:
                    if file.endswith(('.pyd', '.so', '.dll')):
                        print(f"Found built module: {os.path.join(root, file)}")
                        
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            raise

# 确保包结构正确
setup(
    name='shape_based_matching_py',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python wrapper for shape based matching',
    long_description='',
    # 使用正确的扩展模块名称
    ext_modules=[CMakeExtension('shape_based_matching_py')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    # 明确指定包
    packages=[],  # 如果没有纯 Python 包，设为空列表
    include_package_data=True,
)

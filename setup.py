# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import pybind11
import sys
import os

# 强制 setuptools 使用当前编译环境
os.environ.setdefault('DISTUTILS_USE_SDK', '1')
os.environ.setdefault('MSSdk', '1')

# ===== OpenCV 4.13.0 路径（预编译包解压位置）=====
OPENCV_ROOT = r"E:\programing\opencv-4.13.0\opencv\build"
OPENCV_INCLUDE_DIR = os.path.join(OPENCV_ROOT, "include")
OPENCV_LIB_DIR = os.path.join(OPENCV_ROOT, "x64", "vc16", "lib")
# ========================================================

extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/O2', '/openmp']
    extra_link_args = []

    # OpenCV 4.13.0 预编译包为 world 单库模式
    opencv_libs = ['opencv_world4130']
else:
    extra_compile_args = ['-std=c++17', '-O3', '-Wall', '-fopenmp']
    extra_link_args = []
    opencv_libs = ['opencv_core', 'opencv_imgproc', 'opencv_highgui']

ext_modules = [
    Extension(
        'shape_based_matching_py',
        ['pybind11/pybind11.cpp',
         'pybind11/np2mat/ndarray_converter.cpp',
         'line2Dup.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '.',
            OPENCV_INCLUDE_DIR,        # OpenCV 头文件路径
            './MIPP',                  #  mipp.h 在此路径下
            './pybind11',
        ],
        library_dirs=[OPENCV_LIB_DIR], # OpenCV 库文件路径
        libraries=opencv_libs,         # 要链接的库名（不含 .lib 后缀）
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        runtime_library_dirs=[os.path.join(OPENCV_ROOT, "x64", "vc16", "bin")] if sys.platform != 'win32' else [],
    ),
]

setup(
    name='shape_based_matching',
    version='0.1.0',
    author='sunny',
    ext_modules=ext_modules,
    zip_safe=False,
)

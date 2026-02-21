# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import pybind11
import sys
import os

# 强制 setuptools 使用当前编译环境
os.environ.setdefault('DISTUTILS_USE_SDK', '1')
os.environ.setdefault('MSSdk', '1')

# ===== 手动指定 OpenCV 3 路径（请根据实际情况修改）=====
OPENCV_ROOT = r"C:\Users\sdf63\Downloads\opencv3\build"
OPENCV_INCLUDE_DIR = os.path.join(OPENCV_ROOT, "include")
OPENCV_LIB_DIR = os.path.join(OPENCV_ROOT, "x64", "vc15", "lib")  # 如果vc15不存在，改为vc14
# ========================================================

extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++14', '/O2']
    extra_link_args = []
    
    # 添加 OpenCV 库（根据你实际的文件名修改）
    # 如果是 world 模块（单个库）：
    opencv_libs = ['opencv_world346']  # 例如 opencv_world3413.lib -> 写 opencv_world3413
    
    # 如果是多个模块（core, imgproc, highgui等）：
    # opencv_libs = ['opencv_core3413', 'opencv_imgproc3413', 'opencv_highgui3413']
else:
    extra_compile_args = ['-std=c++14', '-O3', '-Wall']
    extra_link_args = []
    opencv_libs = ['opencv_core', 'opencv_imgproc', 'opencv_highgui']  # Linux 下名称不同

ext_modules = [
    Extension(
        'shape_based_matching_py',
        ['pybind11/pybind11.cpp', 'line2Dup.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '.',
            OPENCV_INCLUDE_DIR,        # OpenCV 头文件路径
            './MIPP',         #  mipp.h 在此路径下
        ],
        library_dirs=[OPENCV_LIB_DIR], # OpenCV 库文件路径
        libraries=opencv_libs,         # 要链接的库名（不含 .lib 后缀）
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='shape_based_matching',
    version='0.1.0',
    author='sunny',
    ext_modules=ext_modules,
    zip_safe=False,
)
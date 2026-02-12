# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from setuptools import setup, Extension
import pybind11
import sys

# 根据平台设置编译器参数
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++14', '/O2']
    extra_link_args = []
else:
    extra_compile_args = ['-std=c++14', '-O3', '-Wall']
    extra_link_args = []

ext_modules = [
    Extension(
        'shape_based_matching_py',
        ['pybind11/pybind11.cpp', 'line2Dup.cpp'],
        include_dirs=[pybind11.get_include(), '.'],
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
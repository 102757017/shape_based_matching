import os
import sys
import platform
import subprocess
import sysconfig
import glob
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install_lib import install_lib

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(f"Extension output directory: {extdir}")

        import pybind11
        import numpy
        
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        numpy_include_dir = numpy.get_include()

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-Dpybind11_DIR={pybind11_cmake_dir}',
            f'-DNUMPY_INCLUDE_DIR={numpy_include_dir}',
            f'-DPython_EXECUTABLE={sys.executable}',
            f'-DOpenCV_DIR=C:/opencv/opencv/build',
            # 强制使用Release模式
            '-DCMAKE_BUILD_TYPE=Release',
            # 开启OpenMP
            '-DOPENMP_ENABLE=ON',
        ]

        build_args = ['--config', 'Release']

        if platform.system() == "Windows":
            cmake_args += [
                '-G', 'Visual Studio 17 2022',
                '-A', 'x64',
                '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                # Windows特定优化选项
                '-DUSE_MSVC_OPTIMIZATIONS=ON'
            ]
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={extdir}']
            build_args += ['--', '/m', '/p:UseMultiToolTask=true', '/p:CL_MPCount=8']
        else:
            build_args += ['--', '-j8']  # 使用8个线程并行编译

        env = os.environ.copy()
        
        # 设置环境变量以优化编译
        if platform.system() == "Windows":
            env['CL'] = '/O2 /GL /arch:AVX2 /openmp'
            env['_CL_'] = '/O2 /GL /arch:AVX2 /openmp'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        try:
            print("Running CMake with args:", cmake_args)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            print("Building with args:", build_args)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
            
            # 复制 OpenCV DLL 到输出目录
            self.copy_opencv_dlls(extdir)
            
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            raise

    def copy_opencv_dlls(self, extdir):
        """复制所需的 OpenCV DLL 到扩展目录"""
        opencv_bin_path = "C:/opencv/opencv/build/x64/vc16/bin"
        if os.path.exists(opencv_bin_path):
            print(f"Copying OpenCV DLLs from {opencv_bin_path}")
            
            # 查找所需的 OpenCV DLL
            dll_patterns = [
                "opencv_world4*.dll",
                "opencv_core4*.dll", 
                "opencv_imgproc4*.dll",
                "opencv_highgui4*.dll"
            ]
            
            for pattern in dll_patterns:
                for dll_path in glob.glob(os.path.join(opencv_bin_path, pattern)):
                    dll_name = os.path.basename(dll_path)
                    dest_path = os.path.join(extdir, dll_name)
                    print(f"Copying {dll_name} to {extdir}")
                    shutil.copy2(dll_path, dest_path)
        else:
            print(f"Warning: OpenCV bin path not found: {opencv_bin_path}")

class CustomInstallLib(install_lib):
    """确保 DLL 文件被安装"""
    def run(self):
        super().run()
        # 复制 DLL 文件到安装目录
        for ext in self.distribution.ext_modules:
            if hasattr(ext, '_needs_stub'):
                continue
            build_ext_cmd = self.get_finalized_command('build_ext')
            ext_path = build_ext_cmd.get_ext_fullpath(ext.name)
            ext_dir = os.path.dirname(ext_path)
            
            # 复制 DLL 文件
            for file in os.listdir(ext_dir):
                if file.endswith('.dll') and 'opencv' in file.lower():
                    src = os.path.join(ext_dir, file)
                    dst = os.path.join(self.install_dir, file)
                    print(f"Installing {file} to {self.install_dir}")
                    shutil.copy2(src, dst)

setup(
    name='shape_based_matching_py',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python wrapper for shape based matching',
    ext_modules=[CMakeExtension('shape_based_matching_py')],
    cmdclass={
        'build_ext': CMakeBuild,
        'install_lib': CustomInstallLib,
    },
    zip_safe=False,
    python_requires=">=3.8",
    # 包含数据文件
    include_package_data=True,
    package_data={
        '': ['*.dll'],  # 包含所有 DLL 文件
    },
)

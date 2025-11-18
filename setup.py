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
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        if platform.system() == "Windows":
            cmake_args += [
                '-G', 'Visual Studio 17 2022',
                '-A', 'x64',
                '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE'
            ]
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            build_args += ['--', '/m']
        else:
            build_args += ['--', '-j2']

        env = os.environ.copy()
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
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

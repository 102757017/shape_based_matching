#ifndef __NDARRAY_CONVERTER_H__
#define __NDARRAY_CONVERTER_H__

#include <Python.h>
#include <opencv2/core/core.hpp>

// 这部分是从 cpp 文件移动过来的，并修正了函数签名
// This is a private class that is not exposed to Python
class NumpyAllocator : public cv::MatAllocator
{
public:
    NumpyAllocator() { stdAllocator = cv::Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const;

    // 关键修复：这里的两个 allocate 函数签名必须和 OpenCV 4.x 匹配
    cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step,
                         cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const override; // <-- 修改1

    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const override; // <-- 修改2

    void deallocate(cv::UMatData* u) const override; // <-- 修改3: 加上 override

    const cv::MatAllocator* stdAllocator;
};


class NDArrayConverter {
public:
    // must call this first, or the other routines don't work!
    static bool init_numpy();
    
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};

//
// Define the type converter
//

#include <pybind11/pybind11.h>

namespace pybind11 { namespace detail {
    
template <> struct type_caster<cv::Mat> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};
    
    
}} // namespace pybind11::detail

#endif

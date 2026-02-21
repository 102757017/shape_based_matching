#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "../line2Dup.h"
namespace py = pybind11;

PYBIND11_MODULE(shape_based_matching_py, m) {
    NDArrayConverter::init_numpy();

    // 1. 绑定结果结构体
    py::class_<line2Dup::RegistrationResult>(m, "RegistrationResult")
        .def(py::init<>())
        .def_readwrite("transformation", &line2Dup::RegistrationResult::transformation)
        .def_readwrite("fitness", &line2Dup::RegistrationResult::fitness)
        .def_readwrite("inlier_rmse", &line2Dup::RegistrationResult::inlier_rmse);

    py::class_<line2Dup::Match>(m, "Match")
        .def(py::init<>())
        .def_readwrite("x", &line2Dup::Match::x)
        .def_readwrite("y", &line2Dup::Match::y)
        .def_readwrite("similarity", &line2Dup::Match::similarity)
        .def_readwrite("class_id", &line2Dup::Match::class_id)
        .def_readwrite("template_id", &line2Dup::Match::template_id);

    py::class_<line2Dup::Feature>(m, "Feature")
        .def(py::init<>())
        .def_readwrite("x", &line2Dup::Feature::x)
        .def_readwrite("y", &line2Dup::Feature::y)
        .def_readwrite("label", &line2Dup::Feature::label);

    py::class_<line2Dup::Template>(m, "Template")
        .def(py::init<>())
        .def_readwrite("width", &line2Dup::Template::width)
        .def_readwrite("height", &line2Dup::Template::height)
        .def_readwrite("tl_x", &line2Dup::Template::tl_x)
        .def_readwrite("tl_y", &line2Dup::Template::tl_y)
        .def_readwrite("pyramid_level", &line2Dup::Template::pyramid_level)
        .def_readwrite("features", &line2Dup::Template::features);

    py::class_<cv::Point2f>(m, "CV_Point2f")
        .def(py::init<>())
        .def(py::init<float, float>());

    py::class_<line2Dup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<int, std::vector<int>, float, float>(),
            py::arg("num_features"), py::arg("spread_and_pyr"),
            py::arg("min_det_contrast") = 30, py::arg("min_train_contrast") = 60)
        .def("addTemplate", &line2Dup::Detector::addTemplate, py::arg("sources"), py::arg("class_id"),
            py::arg("object_mask") = cv::Mat(), py::arg("num_features") = 0)
        .def("addTemplate_rotate", &line2Dup::Detector::addTemplate_rotate,
            py::arg("class_id"), py::arg("zero_id"), py::arg("theta"), py::arg("center"))
        .def("writeClasses", &line2Dup::Detector::writeClasses, py::arg("format") = "templates_%s.yml.gz")
        // 修复 clear_classes 的绑定
        .def("clear_classes", static_cast<void (line2Dup::Detector::*)()>(&line2Dup::Detector::clear_classes))
        .def("readClasses", &line2Dup::Detector::readClasses,
            py::arg("class_ids") = std::vector<std::string>(), py::arg("format") = "templates_%s.yml.gz")
        .def("match", &line2Dup::Detector::match, py::arg("sources"),
            py::arg("threshold"), py::arg("class_ids") = std::vector<std::string>(), py::arg("masks") = cv::Mat())
        .def("getTemplates", &line2Dup::Detector::getTemplates, py::arg("class_id"), py::arg("template_id"))
        .def("numTemplates", static_cast<int (line2Dup::Detector::*)() const>(&line2Dup::Detector::numTemplates))
        .def("numTemplates", static_cast<int (line2Dup::Detector::*)(const std::string&) const>(&line2Dup::Detector::numTemplates), py::arg("class_id"))
        .def("classIds", &line2Dup::Detector::classIds)
        .def_readwrite("dx_", &line2Dup::Detector::dx_)
        .def_readwrite("dy_", &line2Dup::Detector::dy_)
        .def("refine", &line2Dup::Detector::refine, py::arg("match"));

    // Info是shapeInfo_producer的嵌套类
    py::class_<shape_based_matching::shapeInfo_producer::Info>(m, "Info")
        .def(py::init<>())
        .def(py::init<float, float>())
        .def_readwrite("angle", &shape_based_matching::shapeInfo_producer::Info::angle)
        .def_readwrite("scale", &shape_based_matching::shapeInfo_producer::Info::scale);

    py::class_<shape_based_matching::shapeInfo_producer>(m, "shapeInfo_producer")
        .def(py::init<>())
        .def(py::init<cv::Mat, cv::Mat>(), py::arg("src"), py::arg("mask") = cv::Mat())
        .def_readwrite("infos", &shape_based_matching::shapeInfo_producer::infos)
        .def_readwrite("angle_range", &shape_based_matching::shapeInfo_producer::angle_range)
        .def_readwrite("angle_step", &shape_based_matching::shapeInfo_producer::angle_step)
        .def_readwrite("scale_range", &shape_based_matching::shapeInfo_producer::scale_range)
        .def_readwrite("scale_step", &shape_based_matching::shapeInfo_producer::scale_step)
        .def("produce_infos", &shape_based_matching::shapeInfo_producer::produce_infos)
        .def_static("save_infos", &shape_based_matching::shapeInfo_producer::save_infos,
            py::arg("infos"), py::arg("path") = "infos.yaml")
        .def_static("load_infos", &shape_based_matching::shapeInfo_producer::load_infos,
            py::arg("path") = "info.yaml")
        .def("src_of", &shape_based_matching::shapeInfo_producer::src_of,
            py::arg("info"))
        .def("mask_of", &shape_based_matching::shapeInfo_producer::mask_of,
            py::arg("info"));
}
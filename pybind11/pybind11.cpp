#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "../line2Dup.h"
#include "py_matcher.h"
namespace py = pybind11;

PYBIND11_MODULE(shape_based_matching_py, m) {
    NDArrayConverter::init_numpy();

    // 1. 绑定结果结构体
    py::class_<line2Dup::RegistrationResult>(m, "RegistrationResult")
        .def(py::init<>())
        .def_readwrite("transformation", &line2Dup::RegistrationResult::transformation)
        .def_readwrite("fitness", &line2Dup::RegistrationResult::fitness)
        .def_readwrite("inlier_rmse", &line2Dup::RegistrationResult::inlier_rmse);

    py::class_<line2Dup::OverlapResult>(m, "OverlapResult")
        .def(py::init<>())
        .def_readwrite("iou", &line2Dup::OverlapResult::iou)
        .def_readwrite("inter_area", &line2Dup::OverlapResult::inter_area)
        .def_readwrite("union_area", &line2Dup::OverlapResult::union_area)
        .def_readwrite("pred_area", &line2Dup::OverlapResult::pred_area)
        .def_readwrite("gt_area", &line2Dup::OverlapResult::gt_area)
        .def_property_readonly("precision", &line2Dup::OverlapResult::precision)
        .def_property_readonly("recall", &line2Dup::OverlapResult::recall);

    py::class_<line2Dup::Match>(m, "Match")
        .def(py::init<>())
        .def_readwrite("x", &line2Dup::Match::x)
        .def_readwrite("y", &line2Dup::Match::y)
        .def_readwrite("width", &line2Dup::Match::width)
        .def_readwrite("height", &line2Dup::Match::height)
        .def_readwrite("similarity", &line2Dup::Match::similarity)
        .def_readwrite("class_id", &line2Dup::Match::class_id)
        .def_readwrite("template_id", &line2Dup::Match::template_id)
        .def_readwrite("confidence", &line2Dup::Match::confidence)
        .def_readwrite("overlap", &line2Dup::Match::overlap)
        .def_readwrite("transform", &line2Dup::Match::transform)
        .def_readwrite("angle", &line2Dup::Match::angle)
        .def_readwrite("scale", &line2Dup::Match::scale)
        .def_readwrite("fitness", &line2Dup::Match::fitness)
        .def_readwrite("inlier_rmse", &line2Dup::Match::inlier_rmse)
        .def("__repr__", [](const line2Dup::Match& m) {
            return "<Match class='" + m.class_id + "' conf=" + std::to_string(m.confidence).substr(0, 5) +
                " overlap=" + std::to_string(m.overlap).substr(0, 5) +
                " pos=(" + std::to_string(m.x) + "," + std::to_string(m.y) + ")>";
        });

    py::class_<line2Dup::MatchParams>(m, "MatchParams")
        .def(py::init<>())
        .def_readwrite("class_ids", &line2Dup::MatchParams::class_ids)
        .def_readwrite("min_confidence", &line2Dup::MatchParams::min_confidence)
        .def_readwrite("max_matches", &line2Dup::MatchParams::max_matches)
        .def_readwrite("use_refine", &line2Dup::MatchParams::use_refine)
        .def_readwrite("min_fitness", &line2Dup::MatchParams::min_fitness)
        .def_readwrite("max_overlap", &line2Dup::MatchParams::max_overlap)
        .def_readwrite("nms", &line2Dup::MatchParams::nms)
        .def_readwrite("nms_overlap", &line2Dup::MatchParams::nms_overlap)
        .def_readwrite("fill_overlap", &line2Dup::MatchParams::fill_overlap)
        .def_readwrite("masks", &line2Dup::MatchParams::masks);

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
        // 两个 match 重载: 简化版(阈值) 与 完整参数版(MatchParams), 必须显式消歧
        .def("match", py::overload_cast<cv::Mat, float, const std::vector<std::string>&, const cv::Mat>(
                &line2Dup::Detector::match),
            py::arg("sources"), py::arg("threshold"),
            py::arg("class_ids") = std::vector<std::string>(), py::arg("masks") = cv::Mat())
        .def("match", py::overload_cast<cv::Mat, const line2Dup::MatchParams&>(
                &line2Dup::Detector::match),
            py::arg("sources"), py::arg("params"))
        .def("matchRect", &line2Dup::Detector::matchRect, py::arg("match"))
        .def("getTemplates", &line2Dup::Detector::getTemplates, py::arg("class_id"), py::arg("template_id"))
        .def("numTemplates", static_cast<int (line2Dup::Detector::*)() const>(&line2Dup::Detector::numTemplates))
        .def("numTemplates", static_cast<int (line2Dup::Detector::*)(const std::string&) const>(&line2Dup::Detector::numTemplates), py::arg("class_id"))
        .def("classIds", &line2Dup::Detector::classIds)
        .def_readwrite("dx_", &line2Dup::Detector::dx_)
        .def_readwrite("dy_", &line2Dup::Detector::dy_)
        .def("refine", &line2Dup::Detector::refine, py::arg("match"))
        .def("getTemplateMask", &line2Dup::Detector::getTemplateMask,
            py::arg("class_id"), py::arg("template_id"),
            py::return_value_policy::reference_internal)
        .def("setTemplateMask", &line2Dup::Detector::setTemplateMask,
            py::arg("class_id"), py::arg("template_id"), py::arg("mask"))
        .def("computeIoU", &line2Dup::Detector::computeIoU,
            py::arg("match"), py::arg("gt_mask"),
            py::arg("templ_mask") = cv::Mat(), py::arg("use_refine") = true);

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

    // ============ 高层门面: 训练/加载/匹配一站式封装 (原 matcher.py 逻辑) ============
    py::class_<PyMatcher>(m, "PyMatcher")
        .def(py::init<>())
        .def("clear", &PyMatcher::clear)
        .def("train", &PyMatcher::train,
            py::arg("train_image"), py::arg("roi"), py::arg("class_id"),
            py::arg("train_params"), py::arg("save_dir") = ".",
            py::arg("exclusion_zones") = py::none(),
            py::arg("positive_mask") = py::none(),
            py::arg("negative_mask") = py::none())
        .def("add_template_class", &PyMatcher::add_template_class,
            py::arg("path"), py::arg("override_params") = py::none())
        .def("get_loaded_class_ids", &PyMatcher::get_loaded_class_ids)
        .def("get_base_template_features", &PyMatcher::get_base_template_features,
            py::arg("class_id"))
        .def("match", &PyMatcher::match,
            py::arg("image"), py::arg("score_threshold"),
            py::arg("class_ids_to_match") = py::none(),
            py::arg("use_nms") = true, py::arg("nms_threshold") = 0.5,
            py::arg("grasp_points_config") = py::none(),
            py::arg("max_matches") = 0, py::arg("min_fitness") = 0.0,
            py::arg("use_refine") = true, py::arg("max_overlap") = 1.0,
            py::arg("masks") = py::none(), py::arg("fill_overlap") = true);
}
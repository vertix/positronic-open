#include <pybind11/pybind11.h>
#include "franka_control.h"

namespace py = pybind11;

PYBIND11_MODULE(_franka_control, m) {
    m.doc() = "Positronic's Franka control module";

    m.attr("VERSION") = positronic::hardware::franka_control::kVersion;
}

#include <torch/extension.h>
#include "ClarificationDataset.h"
#include "ClarificationLz4Dataset.h"
#include "ClarificationOpusDataset.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Base class (abstract, not directly instantiable from Python)
    py::class_<ClarificationDataset>(m, "ClarificationDataset")
        .def("next", &ClarificationDataset::next)
        .def("reset", &ClarificationDataset::reset)
        .def("total_files", &ClarificationDataset::total_files)
        .def_readonly("sample_size", &ClarificationDataset::sample_size)
        .def_readonly("sample_rate", &ClarificationDataset::sample_rate)
        .def_readonly("overlap_size", &ClarificationDataset::overlap_size)
        .def_readonly("file_idx", &ClarificationDataset::file_idx);

    // LZ4 compressed raw audio loader
    py::class_<ClarificationLz4Dataset, ClarificationDataset>(m, "ClarificationLz4Dataset")
        .def(py::init<torch::Device, std::string, std::string, int, int, int>(),
             py::arg("device"),
             py::arg("base_dir"),
             py::arg("csv_filename"),
             py::arg("num_preload_batches") = 16,
             py::arg("batch_size") = 16,
             py::arg("num_threads") = 0);

    // Opus audio loader
    py::class_<ClarificationOpusDataset, ClarificationDataset>(m, "ClarificationOpusDataset")
        .def(py::init<torch::Device, std::string, std::string, int, int, int>(),
             py::arg("device"),
             py::arg("base_dir"),
             py::arg("csv_filename"),
             py::arg("num_preload_batches") = 16,
             py::arg("batch_size") = 16,
             py::arg("num_threads") = 0);
}

#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include "../ExactQUIVER.h"
#include "../ApproxQUIVER.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_exact(torch::Tensor svec, int s) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ExactQUIVER<false, false> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

torch::Tensor quiver_exact_accelerated(torch::Tensor svec, int s) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ExactQUIVER<false, true> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

torch::Tensor quiver_exact_weighted(torch::Tensor svec, torch::Tensor svec_weights, int s) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ExactQUIVER<true, false> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), svec_weights.data_ptr<double>());
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

torch::Tensor quiver_approx(torch::Tensor svec, int s, int m) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ApproxQUIVER<false> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), nullptr, m);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

torch::Tensor quiver_approx_weighted(torch::Tensor svec, torch::Tensor svec_weights, int s, int m) {

    TORCH_CHECK(svec.device().type() == torch::kCPU, "the input vector must be a kCPU tensor");
    TORCH_CHECK(svec.dtype() == torch::kFloat64, "the input vector must be a double (kFloat64)");
    TORCH_CHECK(svec.size(-1) == svec.numel(), "the input vector must be 1D");
    TORCH_CHECK(svec.is_contiguous(), "the input vector must be contiguous in memory");

    ApproxQUIVER<true> A(svec.data_ptr<double>(), (uint32_t)svec.numel(), svec_weights.data_ptr<double>(), m);
    auto sqv = A.calcQuantizationValues(s);
    auto tsqv = torch::zeros(sqv.size(), torch::kFloat64);
    memcpy(tsqv.data_ptr(), sqv.data(), sizeof(double) * sqv.size());

    return tsqv;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quiver_exact", &quiver_exact, "quiver_exact");
    m.def("quiver_exact_accelerated", &quiver_exact_accelerated, "quiver_exact_accelerated");
    m.def("quiver_exact_weighted", &quiver_exact_weighted, "quiver_exact_weighted");
    m.def("quiver_approx", &quiver_approx, "quiver_approx");
    m.def("quiver_approx_weighted", &quiver_approx_weighted, "quiver_approx_weighted");
}


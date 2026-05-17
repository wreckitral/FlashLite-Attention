#include <cmath>
#include <torch/extension.h>

namespace py = pybind11;

void launch_naive_qk(const float* A, const float* B, float* C,
                     int M, int N, int K, float scale);

void launch_naive_softmax(const float* input, float* output,
                          int num_rows, int num_cols,
                          bool use_causal_mask);

void launch_naive_av(const float* A, const float* V,
                     float* O, int M, int N, int K);

void launch_tiled_qk(const float* A, const float* B, float* C,
                     int M, int N, int K, float scale);

void launch_tiled_av(const float* A, const float* V, float* O,
                     int M, int N, int K);

void launch_online_softmax(
    const float* Q,
    const float* K,
    float* A,
    int M,
    int N,
    int d_k,
    float scale,
    bool use_causal_mask
);

void launch_flashLite_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int M,
    int N,
    int d_k,
    float scale,
    bool use_causal_mask,
    int q_offset
);

torch::Tensor naive_qk(torch::Tensor Q, torch::Tensor K, float scale) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    int M = Q.size(0);
    int d_k = Q.size(1);
    int N = K.size(0);
    TORCH_CHECK(K.size(1) == d_k, "Q and K must have same d_k");
    auto S = torch::empty({M, N}, Q.options());
    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    float* S_ptr = S.data_ptr<float>();
    launch_naive_qk(Q_ptr, K_ptr, S_ptr, M, N, d_k, scale);
    return S;
}

torch::Tensor naive_softmax(torch::Tensor S, bool use_causal_mask) {
    TORCH_CHECK(S.is_cuda(), "S must be a CUDA tensor");
    int M = S.size(0);
    int N = S.size(1);
    auto A = torch::empty({M, N}, S.options());
    const float* S_ptr = S.data_ptr<float>();
    float* A_ptr = A.data_ptr<float>();
    launch_naive_softmax(S_ptr, A_ptr, M, N, use_causal_mask);
    return A;
}

torch::Tensor naive_av(torch::Tensor A, torch::Tensor V) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    int M = A.size(0);
    int N = V.size(0);
    int K = V.size(1);
    TORCH_CHECK(V.size(0) == A.size(1), "A cols and V rows must have same size");
    auto O = torch::empty({M, K}, A.options());
    const float* A_ptr = A.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    launch_naive_av(A_ptr, V_ptr, O_ptr, M, N, K);
    return O;
}

torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool use_causal_mask) {
    int d_k = Q.size(1);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    auto S = naive_qk(Q, K, scale);
    auto A = naive_softmax(S, use_causal_mask);
    auto O = naive_av(A, V);
    return O;
}

torch::Tensor tiled_qk(torch::Tensor Q, torch::Tensor K, float scale) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    int M = Q.size(0);
    int d_k = Q.size(1);
    int N = K.size(0);
    TORCH_CHECK(K.size(1) == d_k, "Q and K must have same d_k");
    auto S = torch::empty({M, N}, Q.options());
    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    float* S_ptr = S.data_ptr<float>();
    launch_tiled_qk(Q_ptr, K_ptr, S_ptr, M, N, d_k, scale);
    return S;
}

torch::Tensor tiled_av(torch::Tensor A, torch::Tensor V) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    int M = A.size(0);
    int N = V.size(0);
    int K = V.size(1);
    TORCH_CHECK(A.size(1) == N, "A cols and V rows must match");
    auto O = torch::empty({M, K}, A.options());
    const float* A_ptr = A.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();
    launch_tiled_av(A_ptr, V_ptr, O_ptr, M, N, K);
    return O;
}

torch::Tensor online_softmax(
    torch::Tensor Q,
    torch::Tensor K,
    bool use_causal_mask
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");

    int M = Q.size(0);
    int d_k = Q.size(1);
    int N = K.size(0);

    TORCH_CHECK(K.size(1) == d_k, "Q and K must have same d_k");

    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    auto A = torch::zeros({M, N}, Q.options());

    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    float* A_ptr = A.data_ptr<float>();

    launch_online_softmax(Q_ptr, K_ptr, A_ptr, M, N, d_k, scale, use_causal_mask);

    return A;
}

torch::Tensor flashLite_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool use_causal_mask
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    int M = Q.size(0);
    int d_k = Q.size(1);
    int N = K.size(0);

    TORCH_CHECK(K.size(1) == d_k, "Q and K must have same d_k");
    TORCH_CHECK(V.size(0) == N, "K and V must have same N");
    TORCH_CHECK(V.size(1) == d_k, "V must have same d_k");

    float scale = 1.0f / sqrtf(static_cast<float>(d_k));
    auto O = torch::zeros({M, d_k}, Q.options());

    const float* Q_ptr = Q.data_ptr<float>();
    const float* K_ptr = K.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();

    launch_flashLite_attention(Q_ptr, K_ptr, V_ptr, O_ptr, M, N, d_k, scale, use_causal_mask, 0);

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_qk", &naive_qk, "Q @ K^T with scaling",
          py::arg("Q"), py::arg("K"), py::arg("scale"));

    m.def("naive_softmax", &naive_softmax, "Softmax with causal mask",
          py::arg("S"), py::arg("use_causal_mask"));

    m.def("naive_av", &naive_av, "A @ V multiplication",
          py::arg("A"), py::arg("V"));

    m.def("naive_attention", &naive_attention, "Complete naive attention",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("use_causal_mask") = true);

    m.def("tiled_qk", &tiled_qk, "Tiled Q @ K^T with scaling",
          py::arg("Q"), py::arg("K"), py::arg("scale"));

    m.def("tiled_av", &tiled_av, "Tiled A @ V multiplication",
      py::arg("A"), py::arg("V"));

    m.def("online_softmax", &online_softmax, "Online Softmax with Causal Mask (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("use_causal_mask") = true);

    m.def("flashLite_attention", &flashLite_attention, "Full Flash Attention (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("use_causal_mask") = true);
}

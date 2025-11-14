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

torch::Tensor naive_qk(
    torch::Tensor Q,
    torch::Tensor K,
    float scale
) {
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

torch::Tensor naive_softmax(
    torch::Tensor S,
    bool use_causal_mask
) {
    // 1. check S is cuda tensor
    TORCH_CHECK(S.is_cuda(), "S must be a CUDA tensor");

    // 2. get rows and cols
    int M = S.size(0);
    int N = S.size(1);

    // 3. create output A with the same shape
    auto A = torch::empty({M, N}, S.options());

    // 4. get pointers
    const float* S_ptr = S.data_ptr<float>();
    float* A_ptr = A.data_ptr<float>();

    // 5. launch softmax kernel
    launch_naive_softmax(S_ptr, A_ptr, M, N, use_causal_mask);

    return A;
}

torch::Tensor naive_av(
    torch::Tensor A,
    torch::Tensor V
) {
    // 1. check A & V is cuda tensor
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    // 2. get rows and cols
    int M = A.size(0);
    int N = V.size(0);
    int K = V.size(1);

    // 3. check size
    TORCH_CHECK(V.size(0) == A.size(1), "A cols and V rows must have same size");

    // 4. create output
    auto O = torch::empty({M, K}, A.options());

    // 5. get pointers
    const float* A_ptr = A.data_ptr<float>();
    const float* V_ptr = V.data_ptr<float>();
    float* O_ptr = O.data_ptr<float>();

    launch_naive_av(A_ptr, V_ptr, O_ptr, M, N, K);

    return O;
}

torch::Tensor naive_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool use_causal_mask
) {
    // 1. compute scale
    int d_k = Q.size(1);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // 2. Q @ K^T
    auto S = naive_qk(Q, K, scale);

    // Step 3: Softmax
    auto A = naive_softmax(S, use_causal_mask);

    // Step 4: A @ V
    auto O = naive_av(A, V);

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
}

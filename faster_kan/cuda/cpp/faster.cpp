#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cuda_runtime.h>
#include <utility>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void fwd_launcher(const torch::PackedTensorAccessor64<float, 2> x,
                  const torch::PackedTensorAccessor64<float, 1> grid,
                  torch::PackedTensorAccessor64<float, 3> result,
                  torch::PackedTensorAccessor64<float, 3> th,
                  float hinv, int batchsize, int in_feats, int gridsize);

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                  const torch::PackedTensorAccessor64<float, 2> th,
                  const torch::PackedTensorAccessor64<float, 2> x,
                  const torch::PackedTensorAccessor64<float, 1> grid,
                  torch::PackedTensorAccessor64<float, 2> grad_x,
                  float hinv, int batchsize, int in_feats, int gridsize);

std::tuple<torch::Tensor, torch::Tensor> faster_fwd(torch::Tensor x, torch::Tensor grid, float inv_denom) // inv_denom not learnable
{

    CHECK_INPUT(x);
    CHECK_INPUT(grid);

    const auto x_acc = x.packed_accessor64<float, 2>();
    const auto grid_acc = grid.packed_accessor64<float, 1>();

    int batchsize = x.size(0);
    int in_feats = x.size(1);
    int gridsize = grid.size(0);

    // create leg tensor
    torch::Tensor result = torch::empty({gridsize, batchsize, in_feats},
                                        torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto result_acc = result.packed_accessor64<float, 3>();

    // create th(tanh) tensor
    torch::Tensor th = torch::empty({gridsize, batchsize, in_feats},
                                    torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto th_acc = th.packed_accessor64<float, 3>();

    fwd_launcher(x_acc, grid_acc, result_acc, th_acc, inv_denom, batchsize, in_feats, gridsize);

    cudaDeviceSynchronize();

    return std::tuple<torch::Tensor, torch::Tensor>(
        result.permute({1, 0, 2}).reshape({batchsize, -1}),
        th.permute({1, 0, 2}).reshape({batchsize, -1}));
    // return result;
}

torch::Tensor faster_bwd(torch::Tensor gout, torch::Tensor th, torch::Tensor x, torch::Tensor grid, float inv_denom) // inv_denom not learnable
{

    CHECK_INPUT(gout);
    CHECK_INPUT(x);
    CHECK_INPUT(grid);

    const auto gout_acc = gout.packed_accessor64<float, 2>();
    const auto x_acc = x.packed_accessor64<float, 2>();
    const auto grid_acc = grid.packed_accessor64<float, 1>();
    auto th_acc = th.packed_accessor64<float, 2>();

    int batchsize = x.size(0);
    int in_feats = x.size(1);
    int gridsize = grid.size(0);

    // create leg tensor
    torch::Tensor grad_x = torch::empty({batchsize, in_feats},
                                        torch::device(torch::kCUDA).dtype(torch::kFloat));

    auto gx_acc = grad_x.packed_accessor64<float, 2>();

    bwd_launcher(gout_acc, th_acc, x_acc, grid_acc, gx_acc, inv_denom, batchsize, in_feats, gridsize);

    cudaDeviceSynchronize();

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &faster_fwd, "leg forward");
    m.def("backward", &faster_bwd, "leg backward");
}
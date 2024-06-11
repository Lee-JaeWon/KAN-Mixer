#include <torch/torch.h>
#include <cstdio>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)
#define INDEX3D(a, b, c, db, dc) (((a) * (db) * (dc) + (b) * (dc) + (c)))

__global__ void fwd_kernel(const torch::PackedTensorAccessor64<float, 2> x,
                            const torch::PackedTensorAccessor64<float, 1> grid,
                            torch::PackedTensorAccessor64<float, 3> result,
                            torch::PackedTensorAccessor64<float, 3> th,
                            float hinv, int batchsize, int in_feats, int gridsize, int numThreads){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

            int iB = idx / in_feats;
            int iF = idx % in_feats;

            for(int ig = 0; ig < gridsize; ig++){
                float z = tanhf((x[iB][iF] - grid[ig]) * hinv);
                result[ig][iB][iF] = 1 - z * z;
                th[ig][iB][iF] = z;
            }


    }
}


__global__ void bwd_kernel(const torch::PackedTensorAccessor64<float, 2> gout,
                            const torch::PackedTensorAccessor64<float, 2> th,
                            const torch::PackedTensorAccessor64<float, 2> x,
                            const torch::PackedTensorAccessor64<float, 1> grid,
                            torch::PackedTensorAccessor64<float, 2> grad_x,
                            float hinv, int batchsize, int in_feats, int gridsize, int numThreads){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numThreads) {

        int iB = idx / in_feats;
        int iF = idx % in_feats;

        float gx = 0.0f;

        for(int ig = 0; ig < gridsize; ig++){
            float z = th[iB][ig * in_feats + iF];
            gx += -2.0f * z * (1 - z * z) * gout[iB][ig * in_feats + iF];
        }

        grad_x[iB][iF] = gx;


    }
}



void fwd_launcher(const torch::PackedTensorAccessor64<float, 2> x,
                    const torch::PackedTensorAccessor64<float, 1> grid,
                    torch::PackedTensorAccessor64<float, 3> result,
                    torch::PackedTensorAccessor64<float, 3> th,
                    float hinv, int batchsize, int in_feats, int gridsize){

    int numThreads = batchsize * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    fwd_kernel<<<blockSize, threadSize>>>(x, grid, result, th, 
        hinv, batchsize, in_feats, gridsize, numThreads);
}

void bwd_launcher(const torch::PackedTensorAccessor64<float, 2> gout,
                    const torch::PackedTensorAccessor64<float, 2> th,
                    const torch::PackedTensorAccessor64<float, 2> x,
                    const torch::PackedTensorAccessor64<float, 1> grid,
                    torch::PackedTensorAccessor64<float, 2> grad_x,
                    float hinv, int batchsize, int in_feats, int gridsize){

    int numThreads = batchsize * in_feats;
    dim3 blockSize(DIVUP(numThreads, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    bwd_kernel<<<blockSize, threadSize>>>(gout, th, x, grid, grad_x, 
                                            hinv, batchsize, in_feats, gridsize, numThreads);
}



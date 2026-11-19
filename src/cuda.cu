#include "common.h"
#include <cuda_runtime.h>

constexpr int TILE = 16;

static void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

#define CUDA(call) check_cuda((call), #call)

__global__ void matmul_cuda(dims_t d, const double *A, const double *B, double *C) {
    __shared__ double tile_A[TILE][TILE];
    __shared__ double tile_B[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int i = blockIdx.y * TILE + y, k = blockIdx.x * TILE + x;
    double sum = 0.0;

    for (int tile = 0; tile <= (d.m - 1) / TILE; tile++) {
        int a_col = tile * TILE + x, b_row = tile * TILE + y;
        tile_A[y][x] = (i < d.n && a_col < d.m) ? A[(size_t)i * d.m + a_col] : 0.0;
        tile_B[y][x] = (b_row < d.m && k < d.p) ? B[(size_t)b_row * d.p + k] : 0.0;
        __syncthreads();

        for (int j = 0; j < TILE; j++)
            sum += tile_A[y][j] * tile_B[j][x];
        // Todas as threads devem terminar a leitura antes de carregar o próximo tile.
        __syncthreads();
    }
    if (i < d.n && k < d.p) C[(size_t)i * d.p + k] = sum;
}

int main(int argc, char **argv) {
    dims_t d = parse_dims(argc, argv);
    const dim3 block(TILE, TILE);
    const dim3 grid(1 + (d.p - 1) / TILE, 1 + (d.n - 1) / TILE);
    int device;
    cudaDeviceProp props;
    CUDA(cudaGetDevice(&device));
    CUDA(cudaGetDeviceProperties(&props, device));
    if (grid.x > (unsigned)props.maxGridSize[0] || grid.y > (unsigned)props.maxGridSize[1] ||
        (size_t)grid.x * grid.y > INT_MAX) {
        fprintf(stderr, "Dimensões excedem a capacidade da grade CUDA\n");
        return EXIT_FAILURE;
    }

    double *A = alloc_matrix(d.n, d.m, 0);
    double *B = alloc_matrix(d.m, d.p, 0);
    double *C = alloc_matrix(d.n, d.p, 0);
    init_matrices(d, A, B);
    size_t a_bytes = (size_t)d.n * d.m * sizeof(double);
    size_t b_bytes = (size_t)d.m * d.p * sizeof(double);
    size_t c_bytes = (size_t)d.n * d.p * sizeof(double);
    double *device_A, *device_B, *device_C;
    CUDA(cudaMalloc(&device_A, a_bytes));
    CUDA(cudaMalloc(&device_B, b_bytes));
    CUDA(cudaMalloc(&device_C, c_bytes));
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
    /* Força o carregamento do kernel antes da medição: no CUDA 12 os módulos são
     * carregados sob demanda, e esse custo cairia dentro do tempo do kernel. */
    cudaFuncAttributes attributes;
    CUDA(cudaFuncGetAttributes(&attributes, matmul_cuda));

    double t0 = wall_time();
    CUDA(cudaMemcpy(device_A, A, a_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_B, B, b_bytes, cudaMemcpyHostToDevice));
    CUDA(cudaDeviceSynchronize());
    double t1 = wall_time();
    CUDA(cudaEventRecord(start));
    matmul_cuda<<<grid, block>>>(d, device_A, device_B, device_C);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop));
    CUDA(cudaEventSynchronize(stop));
    double t2 = wall_time();
    CUDA(cudaMemcpy(C, device_C, c_bytes, cudaMemcpyDeviceToHost));
    double t3 = wall_time();

    float kernel_ms = 0.0f;
    CUDA(cudaEventElapsedTime(&kernel_ms, start, stop));
    int valid = verify_matrix(d, A, B, C);
    if (valid) {
        report("cuda", d, (int)(grid.x * grid.y), t3 - t0, checksum(d, C));
        printf("DETAIL,cuda,%.9f,%.9f,%.9f\n", t1 - t0, kernel_ms * 1e-3, t3 - t2);
    }
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(stop));
    CUDA(cudaFree(device_A));
    CUDA(cudaFree(device_B));
    CUDA(cudaFree(device_C));
    free(A);
    free(B);
    free(C);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

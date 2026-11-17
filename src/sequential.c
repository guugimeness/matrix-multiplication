#include "common.h"

/* C (n x p) = A (n x m) * B (m x p), c_ik = sum_j a_ij * b_jk.
 * Ordem i-j-k: o laço interno (k) percorre B e C com stride 1. */
static void matmul_seq(dims_t d, const double *A, const double *B, double *C) {
    const int n = d.n, m = d.m, p = d.p;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            const double a_ij = A[(size_t)i * m + j];
            for (int k = 0; k < p; k++)
                C[(size_t)i * p + k] += a_ij * B[(size_t)j * p + k];
        }
}

int main(int argc, char **argv) {
    dims_t d = parse_dims(argc, argv);
    double *A = alloc_matrix(d.n, d.m, 0);
    double *B = alloc_matrix(d.m, d.p, 0);
    double *C = alloc_matrix(d.n, d.p, 1);
    init_matrices(d, A, B);

    double t0 = wall_time();
    matmul_seq(d, A, B, C);
    double t1 = wall_time();

    int valid = verify_matrix(d, A, B, C);
    if (valid) report("sequential", d, 1, t1 - t0, checksum(d, C));
    free(A); free(B); free(C);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

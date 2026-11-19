#include "common.h"

/* C (n x p) = A (n x m) * B (m x p), c_ik = sum_j a_ij * b_jk.
 * Todas as linhas de C são calculadas pelo núcleo i-j-k de common.h. */
static void matmul_seq(dims_t d, const double *A, const double *B, double *C) {
    matmul_rows(d, 0, d.n, A, B, C);
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

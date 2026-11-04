/* Infraestrutura comum a todas as implementações de multiplicação de matrizes.
 * Notação do enunciado: A (n x m) * B (m x p) = C (n x p), com c_ik = sum_j a_ij * b_jk. */
#ifndef COMMON_H
#define COMMON_H

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { int n, m, p; } dims_t;

/* Tempo de parede em segundos (relógio monotônico, imune a ajustes do relógio do sistema). */
static inline double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Lê as dimensões: "prog n m p", ou "prog n" para matrizes quadradas (padrão: 1024). */
static inline dims_t parse_dims(int argc, char **argv) {
    dims_t d;
    if (argc >= 4) {
        d.n = atoi(argv[1]); d.m = atoi(argv[2]); d.p = atoi(argv[3]);
    } else {
        d.n = d.m = d.p = (argc > 1) ? atoi(argv[1]) : 1024;
    }
    if (d.n <= 0 || d.m <= 0 || d.p <= 0) {
        fprintf(stderr, "Uso: %s n [m p]  (dimensões positivas)\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    return d;
}

/* Aloca uma matriz rows x cols contígua (row-major), opcionalmente zerada. */
static inline double *alloc_matrix(int rows, int cols, int zero) {
    size_t count = (size_t)rows * cols;
    double *M = zero ? calloc(count, sizeof(double)) : malloc(count * sizeof(double));
    if (!M) { fprintf(stderr, "Falha ao alocar %zu bytes\n", count * sizeof(double)); exit(EXIT_FAILURE); }
    return M;
}

/* Preenchimento determinístico: todas as versões multiplicam as mesmas matrizes.
 * a_ij para 0 <= i < n, 0 <= j < m;  b_jk para 0 <= j < m, 0 <= k < p. */
static inline void init_matrices(dims_t d, double *A, double *B) {
    for (int i = 0; i < d.n; i++)
        for (int j = 0; j < d.m; j++)
            A[(size_t)i * d.m + j] = ((i * 7 + j * 3) % 11) / 10.0;
    for (int j = 0; j < d.m; j++)
        for (int k = 0; k < d.p; k++)
            B[(size_t)j * d.p + k] = ((j * 5 + k * 2) % 13) / 10.0;
}

/* Soma de todos os elementos de C (n x p), usada para validar as versões paralelas. */
static inline double checksum(dims_t d, const double *C) {
    double s = 0.0;
    for (size_t idx = 0; idx < (size_t)d.n * d.p; idx++) s += C[idx];
    return s;
}

/* Linha padronizada lida pela função benchmark() do notebook. */
static inline void report(const char *impl, dims_t d, int workers, double seconds, double chk) {
    printf("RESULT,%s,%d,%d,%d,%d,%.6f,%.17g\n", impl, d.n, d.m, d.p, workers, seconds, chk);
}

#endif

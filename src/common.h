/* Infraestrutura comum a todas as implementações de multiplicação de matrizes.
 * Notação do enunciado: A (n x m) * B (m x p) = C (n x p), com c_ik = sum_j a_ij * b_jk. */
#ifndef COMMON_H
#define COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int n, m, p; } dims_t;

/* restrict é palavra-chave de C; em C++ (CUDA) o equivalente é __restrict__. */
#ifdef __cplusplus
#define MATMUL_RESTRICT __restrict__
#else
#define MATMUL_RESTRICT restrict
#endif

/* Tempo de parede em segundos (relógio monotônico, imune a ajustes do relógio do sistema). */
static inline double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline int positive_int(const char *text) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    return (errno || end == text || *end || value <= 0 || value > INT_MAX)
        ? 0 : (int)value;
}

/* Lê as dimensões sem encerrar o programa: devolve 1 se forem válidas.
 * "prog n m p", ou "prog n" para matrizes quadradas (padrão: 1024). */
static inline int try_parse_dims(int argc, char **argv, dims_t *d) {
    *d = (dims_t){1024, 1024, 1024};
    if (argc == 4) {
        d->n = positive_int(argv[1]);
        d->m = positive_int(argv[2]);
        d->p = positive_int(argv[3]);
    } else if (argc != 1)
        d->n = d->m = d->p = (argc == 2) ? positive_int(argv[1]) : 0;
    return d->n > 0 && d->m > 0 && d->p > 0;
}

static inline void print_usage(const char *prog) {
    fprintf(stderr, "Uso: %s n [m p]  (dimensões positivas)\n", prog);
}

/* Versão usada pelos programas sem MPI: encerra com a mensagem de uso se as dimensões forem inválidas. */
static inline dims_t parse_dims(int argc, char **argv) {
    dims_t d;
    if (!try_parse_dims(argc, argv, &d)) {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    return d;
}

/* Aloca uma matriz rows x cols contígua (row-major), opcionalmente zerada. */
static inline double *alloc_matrix(int rows, int cols, int zero) {
    if ((size_t)rows > SIZE_MAX / sizeof(double) / (size_t)cols) {
        fprintf(stderr, "Dimensões excedem o tamanho máximo de alocação\n");
        exit(EXIT_FAILURE);
    }
    size_t count = (size_t)rows * cols;
    if (count == 0) count = 1; /* Um buffer válido para ranks sem linhas. */
    double *M = (double *)(zero ? calloc(count, sizeof(double)) : malloc(count * sizeof(double)));
    if (!M) { fprintf(stderr, "Falha ao alocar %zu bytes\n", count * sizeof(double)); exit(EXIT_FAILURE); }
    return M;
}

/* Preenchimento determinístico: todas as versões multiplicam as mesmas matrizes.
 * a_ij para 0 <= i < n, 0 <= j < m;  b_jk para 0 <= j < m, 0 <= k < p. */
static inline void init_matrices(dims_t d, double *A, double *B) {
    for (int i = 0; i < d.n; i++)
        for (int j = 0; j < d.m; j++)
            A[(size_t)i * d.m + j] = (((size_t)i * 7 + (size_t)j * 3) % 11) / 10.0;
    for (int j = 0; j < d.m; j++)
        for (int k = 0; k < d.p; k++)
            B[(size_t)j * d.p + k] = (((size_t)j * 5 + (size_t)k * 2) % 13) / 10.0;
}

/* Núcleo de cálculo comum às versões de CPU: acumula em C as linhas [row_begin, row_end).
 * Ordem i-j-k: o laço interno (k) percorre B[j][.] e C[i][.] com stride 1.
 * Usar a mesma função na versão sequencial, no OpenMP e no MPI garante que o compilador
 * gere o mesmo código para o laço, e o speedup mede só o efeito da paralelização. */
static inline void matmul_rows(dims_t d, int row_begin, int row_end,
                               const double *MATMUL_RESTRICT A, const double *MATMUL_RESTRICT B,
                               double *MATMUL_RESTRICT C) {
    const int m = d.m, p = d.p;
    for (int i = row_begin; i < row_end; i++)
        for (int j = 0; j < m; j++) {
            const double a_ij = A[(size_t)i * m + j];
            for (int k = 0; k < p; k++)
                C[(size_t)i * p + k] += a_ij * B[(size_t)j * p + k];
        }
}

/* Ative em casos pequenos para conferir cada elemento por produto escalar. */
static inline int verify_matrix(dims_t d, const double *A, const double *B, const double *C) {
    const char *enabled = getenv("MATMUL_VERIFY");
    if (!enabled || strcmp(enabled, "1") != 0) return 1;

    for (int i = 0; i < d.n; i++)
        for (int k = 0; k < d.p; k++) {
            double expected = 0.0;
            for (int j = 0; j < d.m; j++)
                expected += A[(size_t)i * d.m + j] * B[(size_t)j * d.p + k];
            double actual = C[(size_t)i * d.p + k];
            if (!(fabs(actual - expected) <= 1e-12 + 1e-10 * fabs(expected))) {
                fprintf(stderr, "C[%d,%d] = %.17g, esperado %.17g\n", i, k, actual, expected);
                return 0;
            }
        }
    return 1;
}

/* Soma de todos os elementos de C (n x p), usada para validar as versões paralelas. */
static inline double checksum(dims_t d, const double *C) {
    double s = 0.0;
    for (size_t idx = 0; idx < (size_t)d.n * d.p; idx++) s += C[idx];
    return s;
}

/* Linha padronizada lida pela função benchmark() do notebook. */
static inline void report(const char *impl, dims_t d, int workers, double seconds, double chk) {
    printf("RESULT,%s,%d,%d,%d,%d,%.9f,%.17g\n", impl, d.n, d.m, d.p, workers, seconds, chk);
}

#endif

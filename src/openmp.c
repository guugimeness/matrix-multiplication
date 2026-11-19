#include "common.h"
#include <omp.h>

/* Escolhidos na EXECUÇÃO (o programa é compilado uma só vez):
 *   OMP_NUM_THREADS : número de threads do time
 *   OMP_SCHEDULE    : static | dynamic | guided, com chunk opcional (ex.: "dynamic,32") */

/* Multiplica A por B em paralelo e devolve quantas threads participaram. */
static int matmul_omp(dims_t d, const double *A, const double *B, double *C) {
    const int n = d.n;
    int nthreads = 1;

    // Região paralela: cria o time de threads.
    // As matrizes e as dimensões são compartilhadas por todas as threads.
    #pragma omp parallel shared(A, B, C, d, nthreads)
    {
        // Só UMA thread do time registra o tamanho
        #pragma omp single nowait
        nthreads = omp_get_num_threads();

        // Worksharing: as iterações de i (linhas de C) são divididas entre as threads.
        // schedule(runtime): a política (static, dynamic...) é lida de OMP_SCHEDULE.
        // A variável i, declarada no laço, é privada de cada thread.
        // Cada iteração calcula a linha i de C com o mesmo núcleo da versão sequencial
        // (matmul_rows, em common.h), percorrendo B[j][.] e C[i][.] com stride 1.
        #pragma omp for schedule(runtime)
        for (int i = 0; i < n; i++)
            matmul_rows(d, i, i + 1, A, B, C);
        // Ao sair da região paralela, C está completa.
    }

    return nthreads;
}

/* Nome da implementação na linha RESULT: "openmp-<política>", lido de OMP_SCHEDULE.
 * O chunk só aparece quando não é o padrão (ex.: "openmp-dynamic-32"). */
static void impl_name(char *buf, size_t size) {
    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);        // política que schedule(runtime) vai usar

    // O bit mais alto é o modificador "monotonic" (OpenMP 5.0): removido para identificar a política.
    int base = (int)((unsigned)kind & 0x7fffffffu);

    if (base == omp_sched_static || base == omp_sched_dynamic || base == omp_sched_guided) {
        const char *name = (base == omp_sched_static)  ? "static"
                         : (base == omp_sched_dynamic) ? "dynamic" : "guided";
        int chunk_padrao = (base == omp_sched_static) ? 0 : 1;
        if (chunk > chunk_padrao) snprintf(buf, size, "openmp-%s-%d", name, chunk);
        else                      snprintf(buf, size, "openmp-%s", name);
    } else {
        snprintf(buf, size, "openmp-auto");
    }
}

int main(int argc, char **argv) {
    dims_t d = parse_dims(argc, argv);
    double *A = alloc_matrix(d.n, d.m, 0);
    double *B = alloc_matrix(d.m, d.p, 0);
    double *C = alloc_matrix(d.n, d.p, 1);
    init_matrices(d, A, B);

    // Tempo de parede só da multiplicação (a criação das threads entra, pois faz parte do custo).
    double t0 = wall_time();
    int nthreads = matmul_omp(d, A, B, C);
    double t1 = wall_time();

    char impl[64];
    impl_name(impl, sizeof impl);
    int valid = verify_matrix(d, A, B, C);
    if (valid) report(impl, d, nthreads, t1 - t0, checksum(d, C));
    free(A); free(B); free(C);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Adapted from https://colfaxresearch.com/skl-avx512/

#include <stdio.h>
#include <omp.h>

const int n_trials = 1000000000; // Enough to keep cores busy for a while and observe a steady state
const int flops_per_calc = 2; // Multiply + add = 2 instructions
#define n_chained_fmas 10 // Must be tuned for architectures here and in blocks (R) and in (E)
#define FA_SIZE VECTOR_WIDTH*n_chained_fmas

// Compile with 
// gcc -fopenmp -DVECTOR_WIDTH=4 -march=skylake-avx512 -o build/fma_bench benchmarks/fma_bench.c
// gcc -fopenmp -DVECTOR_WIDTH=8 -march="-mavx -mfma" -o build/fma_bench benchmarks/fma_bench.c

int main() {

#pragma omp parallel
  { } // Warm up the threads

  const double t0 = omp_get_wtime(); // start timer
#pragma omp parallel
  { // Benchmark in all threads
    double fa[FA_SIZE] = {[0 ... FA_SIZE-1 ] = 0.0};
    double fb[VECTOR_WIDTH] = {[0 ... VECTOR_WIDTH-1 ] = 0.5};
    double fc[VECTOR_WIDTH] = {[0 ... VECTOR_WIDTH-1 ] = 1.0};

    register double *fa01 = fa +  0*VECTOR_WIDTH; // This is block (R)
    register double *fa02 = fa +  1*VECTOR_WIDTH; // To tune for a specific architecture,
    register double *fa03 = fa +  2*VECTOR_WIDTH; // more or fewer fa* variables
    register double *fa04 = fa +  3*VECTOR_WIDTH; // must be used
    register double *fa05 = fa +  4*VECTOR_WIDTH;
    register double *fa06 = fa +  5*VECTOR_WIDTH;
    register double *fa07 = fa +  6*VECTOR_WIDTH;
    register double *fa08 = fa +  7*VECTOR_WIDTH;
    register double *fa09 = fa +  8*VECTOR_WIDTH;
    register double *fa10 = fa +  9*VECTOR_WIDTH;

    int i, j;
#pragma nounroll // Prevents automatic unrolling by compiler to avoid skewed benchmarks
    for(i = 0; i < n_trials; i++)
#pragma omp simd // Ensures that vectorization does occur
      for (j = 0; j < VECTOR_WIDTH; j++) { // VECTOR_WIDTH=4 for AVX2, =8 for AVX-512
        fa01[j] = fa01[j]*fb[j] + fc[j]; // This is block (E)
        fa02[j] = fa02[j]*fb[j] + fc[j]; // To tune for a specific architecture,
        fa03[j] = fa03[j]*fb[j] + fc[j]; // more or fewer such FMA constructs
        fa04[j] = fa04[j]*fb[j] + fc[j]; // must be used
        fa05[j] = fa05[j]*fb[j] + fc[j];
        fa06[j] = fa06[j]*fb[j] + fc[j];
        fa07[j] = fa07[j]*fb[j] + fc[j];
        fa08[j] = fa08[j]*fb[j] + fc[j];
        fa09[j] = fa09[j]*fb[j] + fc[j];
        fa10[j] = fa10[j]*fb[j] + fc[j];
      }
    for(i = 0; i < FA_SIZE; i++)
      fa[i] *= 2.0; // Prevent dead code elimination
  }
  const double t1 = omp_get_wtime();

  const double gflops = 1.0e-9*(double)VECTOR_WIDTH*(double)n_trials*(double)flops_per_calc*
                        (double)omp_get_max_threads()*(double)n_chained_fmas;
  printf("Chained FMAs=%d, vector width=%d, GFLOPs=%.1f, time=%.6f s, performance=%.1f GFLOP/s\n", 
                        n_chained_fmas, VECTOR_WIDTH, gflops, t1 - t0, gflops/(t1 - t0));
}
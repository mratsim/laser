# Apache v2 License
# Mamy Ratsimbazafy

# A benchmark with the Gauss Seidel Smoother iterative stencil
# This is a good benchmark to test wavefront optimisations
# that is very important in RNNs.
#
# For the moment, the benchmark is pure OpenMP until the Lux
# compiler is ready for comparison.

# ##########################################
# Benchmarking tools
import random, times, stats, strformat, math, sequtils

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = epochTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = epochTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats(name: string, result: openarray) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  # echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GFLOP/s"

template bench(name: string, initialisation, body: untyped) {.dirty.}=
  block: # Actual bench
    var stats: RunningStat
    let global_start = epochTime()
    for _ in 0 ..< nb_samples:
      initialisation
      let start = epochTime()
      body
      let stop = epochTime()
      stats.push stop - start
    let global_stop = epochTime()
    printStats(name, result)

# #############################################
# Params

const
  Time = 100
  Tile = 32
  N = 512

proc benchSerialGaussSeidel(p: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  result = newSeq[float32](p.len)

  let pr = result[0].unsafeAddr

  bench("Reference loop"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    deepCopy(result, p) # We copy the original
  do:
    {.emit: """
      float (* __restrict P)[`N`] = (void*)`pr`;

      // time loop
      for (int t = 0; t < `Time`; ++t)
        // stencil
        for (int i = 1; i < `N` - 1; ++i)
          for (int j = 1; j < `N` - 1; ++j)
            P[i][j] = 0.25 * (P[i][j-1] * // left
                              P[i][j+1] * // right
                              P[i-1][j] * // top
                              P[i+1][j]); // bottom

    """.}

proc benchParallelManualSkew(p: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  # TODO fix Illegal access

  result = newSeq[float32](p.len)

  let pr = result[0].unsafeAddr

  bench("Parallel Manually Skewed"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    deepCopy(result, p) # We copy the original
  do:
    {.emit: """
      float (* __restrict P)[`N`] = (void*)`pr`;

      int NB = `N` / `Tile`;
      #pragma omp parallel
      for (int t = 0; t < `Time`; ++t) {
        // first NB diagonals
        for (int diag = 0; diag < NB; ++diag)
          #pragma omp for
          for (int d = 0; d <= diag; ++d){
            int ii = d;
            int jj = diag - d;
            for (int i = 1+ii*`Tile`; i < (ii+1)*`Tile`; ++i)
              for (int j = 1+jj*`Tile`; i < (jj+1)*`Tile`; ++j)
                P[i][j] = 0.25 * (P[i][j-1] * // left
                                  P[i][j+1] * // right
                                  P[i-1][j] * // top
                                  P[i+1][j]); // bottom
          } // barrier

        // Last NB diagonals
        for (int diag = NB-1; diag >= 0; --diag)
          #pragma omp for
          for (int d = diag; d >= 0; --d){
            int ii = d;
            int jj = diag - d;
            for (int i = 1+ii*`Tile`; i < (ii+1)*`Tile`; ++i)
              for (int j = 1+jj*`Tile`; i < (jj+1)*`Tile`; ++j)
                P[i][j] = 0.25 * (P[i][j-1] * // left
                                  P[i][j+1] * // right
                                  P[i-1][j] * // top
                                  P[i+1][j]); // bottom
          } // barrier
      }
    """.}


proc benchOMP5TaskDepend(p: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  # TODO fix Illegal access

  result = newSeq[float32](p.len)

  let pr = result[0].unsafeAddr

  bench("OpenMP 5.0 Task Depend + 2D Tiling"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    deepCopy(result, p) # We copy the original
  do:
    {.emit: """
      float (* __restrict P)[`N`] = (void*)`pr`;

      int NB = `N` / `Tile`;

      #pragma omp parallel
      #pragma omp single
      // time loop
      for (int t = 0; t < `Time`; ++t)
        // stencil
        for (int ii = 1; ii < `N` - 1; ii += `Tile`)
          for (int jj = 1; jj < `N` - 1; jj += `Tile`)
            #pragma omp task depend(inout: p[ii:TS][jj:TS]) depend(in: p[ii-`T`:`T`][jj:`T`], p[ii+`T`:`T`][jj:`T`], p[ii:`T`][jj-`T`:`T`], p[ii:`T`][jj+`T`:`T`])
            {
              for (int i = ii; 1<(1+ii)*`Tile`; ++i)
                for (int j = jj; 1<(1+jj)*`Tile`; ++j)
                  P[i][j] = 0.25 * (P[i][j-1] * // left
                                    P[i][j+1] * // right
                                    P[i-1][j] * // top
                                    P[i+1][j]); // bottom
            }

    """.}

# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

when isMainModule:
  import ../../laser/private/error_functions

  randomize(42) # For reproducibility
  warmup() # Not needed with ref implementation warmup
  echo ""
  echo "Matrix shape: ", N, "x", N
  echo "Timesteps: ", Time
  echo "TimeSize: ", Tile

  block:
    let a = newSeqWith(N*N, float32 rand(-0.1..0.1))

    let reference = benchSerialGaussSeidel(a, 10)
    # let manualSkewTiled = benchParallelManualSkew(a, 10)
    # let omp4TaskDepTiled = benchOMP5TaskDepend(a, 10)

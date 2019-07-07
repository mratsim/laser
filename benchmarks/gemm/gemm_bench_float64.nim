# Apache v2 License
# Mamy Ratsimbazafy

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

template printStats(name: string, output: openarray) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GFLOP/s"
  echo "\nDisplay output[0] to make sure it's not optimized away"
  echo output[0] # Prevents compiler from optimizing stuff away

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
    printStats(name, output)

# #############################################
# Params
import
  ./gemm_common,
  ../third_party/blas,
  ./arraymancer/blas_l3_gemm,
  ../../laser/primitives/matrix_multiplication/gemm

const
  M     = 8*6*20
  K     = 8*6*20
  N     = 8*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big
  CpuGhz = 2.7      # Assuming no turbo
  NumCpuCores = 2
  CpuFlopCycle = 16 # AVX2: 2xFMA/cycle = 2x4x2 - 2 x 4 float64 x (1 add + 1 mul)

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(float64) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

# #############################################

proc benchOpenBLAS(a, b: seq[float64], nb_samples: int) =
  var output = newSeq[float64](out_size)
  bench("OpenBLAS benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm(
      rowMajor, noTranspose, noTranspose,
      M, N, K,
      1, a[0].unsafeaddr, K,
      b[0].unsafeAddr, N,
      0, output[0].addr, N
    )

proc benchArraymancerFallback(a, b: seq[float64], nb_samples: int) =
  var output = newSeq[float64](out_size)
  bench("Arraymancer fallback BLAS"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'f64,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'f64, output, 0, N, 1
    )

proc benchSimpleTiling(a, b: seq[float64], nb_samples: int) {.noinline.}=
  var output = newSeq[float64](out_size)

  let pa = a[0].unsafeAddr
  let pb = b[0].unsafeAddr
  let po = output[0].addr
  const blck = 32

  bench("Simple Tiling"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    {.emit: """
      #define min(a,b) (((a)<(b))?(a):(b))

      double (* restrict A)[`K`] = (void*)`pa`;
      double (* restrict B)[`N`] = (void*)`pb`;
      double (* restrict C)[`N`] = (void*)`po`;

      // TODO: where to parallelize?
      for (int j = 0; j < `N`; j+=`blck`)
        for (int k = 0; k < `K`; k+=`blck`)
          for (int i = 0; i < `M`; i++)
            for (int jj = j; jj<min(j+`blck`, `N`); jj++)
              for (int kk = k; kk<min(k+`blck`, `K`); kk++)
                C[i][jj] += A[i][kk] * B[kk][jj];

    """.}

proc benchLaserGEMM(a, b: seq[float64], nb_samples: int) =
  var output = newSeq[float64](out_size)

  let a_ptr{.restrict.} = a[0].unsafeAddr
  let b_ptr{.restrict.} = b[0].unsafeAddr
  let c_ptr{.restrict.} = output[0].addr
  bench("Laser production implementation"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_strided(
      M, N, K,
      1'f64,  a_ptr, K, 1,       # stride row, stride col
              b_ptr, N, 1,
      0'f64,  c_ptr, N, 1
    )
# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  echo ""
  echo "A matrix shape: " & $ashape
  echo "B matrix shape: " & $bshape
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  echo &"Theoretical peak single-core:  {CpuGhz * CpuFlopCycle:>9.3f} GFLOP/s"
  echo &"Theoretical peak multi:        {CpuGhz * CpuFlopCycle * NumCpuCores:>9.3f} GFLOP/s"
  echo "Make sure to not bench Apple Accelerate or the default Linux BLAS."
  block:
    let a = newSeqWith(M*K, float64 rand(1.0))
    let b = newSeqWith(K*N, float64 rand(1.0))

    # when not defined(openmp):
    #   benchSimpleTiling(a, b, NbSamples) # for some reason stalled with OpenMP
    # benchArraymancerFallback(a, b, NbSamples)
    benchOpenBLAS(a, b, NbSamples)
    benchLaserGEMM(a, b, NbSamples)

# Seems like my original Arraymancer BLAS has false sharing issue
# FYI Apple accelerate is about 117~122GFLOP/s on my machine.

###############################
# OpenMP

# Warmup: 1.1890 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 960, N: 960)
# B matrix shape: (M: 960, N: 960)
# Output shape: (M: 960, N: 960)
# Required number of operations:  1769.472 millions
# Required bytes:                   14.746 MB
# Arithmetic intensity:            120.000 FLOP/byte
# Theoretical peak single-core:     43.200 GFLOP/s
# Theoretical peak multi:           86.400 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.315 seconds
# Average time: 31.429 ms
# Stddev  time: 2.503 ms
# Min     time: 29.868 ms
# Max     time: 37.536 ms
# Perf:         56.300 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 232.3620566397699

# Laser production implementation
# Collected 10 samples in 0.327 seconds
# Average time: 32.625 ms
# Stddev  time: 3.080 ms
# Min     time: 31.182 ms
# Max     time: 41.327 ms
# Perf:         54.236 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 232.36205663977

###############################
# Serial

# Warmup: 1.1948 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 960, N: 960)
# B matrix shape: (M: 960, N: 960)
# Output shape: (M: 960, N: 960)
# Required number of operations:  1769.472 millions
# Required bytes:                   14.746 MB
# Arithmetic intensity:            120.000 FLOP/byte
# Theoretical peak single-core:     43.200 GFLOP/s
# Theoretical peak multi:           86.400 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.566 seconds
# Average time: 56.528 ms
# Stddev  time: 2.482 ms
# Min     time: 55.359 ms
# Max     time: 63.552 ms
# Perf:         31.303 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 232.3620566397699

# Laser production implementation
# Collected 10 samples in 0.531 seconds
# Average time: 53.075 ms
# Stddev  time: 1.592 ms
# Min     time: 51.679 ms
# Max     time: 55.885 ms
# Perf:         33.339 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 232.36205663977

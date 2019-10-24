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

import ../third_party/manu/manu/matrix as manu

const
  M     = 8*6*20
  K     = 8*6*20
  N     = 8*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big
  CpuGhz = 3.5      # i9-9980XE OC All turbo 4.1GHz (AVX2 4.0GHz, AVX512 3.5GHz)
  NumCpuCores = 18
  VectorWidth = 8   # 4 float64 for AVX2, 8 for AVX512
  InstrCycle = 2    # How many instructions per cycle, (2xFMAs or 1xFMA for example)
  FlopInstr = 2     # How many FLOP per instr (FMAs = 1 add + 1 mul)

  TheoSerialPeak = CpuGhz * VectorWidth * InstrCycle * FlopInstr
  TheoThreadedPeak = TheoSerialPeak * NumCpuCores

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

proc benchManu(a, b: seq[float64], nb_samples: int) =
  let Amat = manu.matrix(a, M)
  let Bmat = manu.matrix(N, b)
  var C: manu.Matrix
  # let output = C.data.addr # data is not exposed :/
  var output: array[1, float64] # The bench display the first item for sanity checks

  bench("Manu implementation"):
    # No initialization needed, Manu doesn't work in-place
    discard
  do:
    # Main work
    C = Amat * Bmat
    output[0] = C[0, 0]

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
  # warmup() # Not needed with ref implementation warmup
  echo ""
  echo "A matrix shape: " & $ashape
  echo "B matrix shape: " & $bshape
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  echo &"Theoretical peak single-core:  {TheoSerialPeak:>9.3f} GFLOP/s"
  echo &"Theoretical peak multi:        {TheoThreadedPeak:>9.3f} GFLOP/s"
  echo "Make sure to not bench Apple Accelerate or the default Linux BLAS."
  echo "Due to strange OpenMP interferences, separate the run of code-sections using OpenMP, see https://github.com/numforge/laser/issues/40"
  block:
    let a = newSeqWith(M*K, float64 rand(1.0))
    let b = newSeqWith(K*N, float64 rand(1.0))

    # when not defined(openmp):
    #   benchSimpleTiling(a, b, NbSamples) # for some reason stalled with OpenMP
    # benchArraymancerFallback(a, b, NbSamples)
    # benchOpenBLAS(a, b, NbSamples)
    benchLaserGEMM(a, b, NbSamples)
    benchManu(a, b, NbSamples)

# Seems like my original Arraymancer BLAS has false sharing issue

###############################
# OpenMP
# Due to strange OpenMP interferences, OpenMP code sections should be run independently
# see https://github.com/numforge/laser/issues/40

# Run 1: OpenBLAS vs Manu

# A matrix shape: (M: 960, N: 960)
# B matrix shape: (M: 960, N: 960)
# Output shape: (M: 960, N: 960)
# Required number of operations:  1769.472 millions
# Required bytes:                   14.746 MB
# Arithmetic intensity:            120.000 FLOP/byte
# Theoretical peak single-core:    112.000 GFLOP/s
# Theoretical peak multi:         2016.000 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.
# Due to strange OpenMP interferences, separate the run of code-sections using OpenMP, see https://github.com/numforge/laser/issues/40
#
# OpenBLAS benchmark
# Collected 10 samples in 0.033 seconds
# Average time: 3.256 ms
# Stddev  time: 0.567 ms
# Min     time: 2.910 ms
# Max     time: 4.715 ms
# Perf:         543.396 GFLOP/s
#
# Display output[0] to make sure it's not optimized away
# 232.3620566397699
#
# Manu implementation
# Collected 10 samples in 8.477 seconds
# Average time: 847.700 ms
# Stddev  time: 10.644 ms
# Min     time: 842.805 ms
# Max     time: 877.909 ms
# Perf:         2.087 GFLOP/s
#
# Display output[0] to make sure it's not optimized away
# 237.8399578000516

# Run 2: Laser vs Manu

# Laser production implementation
# Collected 10 samples in 0.041 seconds
# Average time: 4.008 ms
# Stddev  time: 5.121 ms
# Min     time: 2.232 ms
# Max     time: 18.579 ms
# Perf:         441.537 GFLOP/s
#
# Display output[0] to make sure it's not optimized away
# 232.36205663977
#
# Manu implementation
# Collected 10 samples in 8.490 seconds
# Average time: 848.983 ms
# Stddev  time: 0.997 ms
# Min     time: 847.062 ms
# Max     time: 850.112 ms
# Perf:         2.084 GFLOP/s
#
# Display output[0] to make sure it's not optimized away
# 237.8399578000516

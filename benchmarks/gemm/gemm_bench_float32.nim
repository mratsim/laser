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

template printStats(name: string, result: openarray) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GFLOP/s"
  echo "\nDisplay result[0] to make sure it's not optimized away"
  echo result[0] # Prevents compiler from optimizing stuff away

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
import
  ./gemm_common,
  ../blas,
  ./arraymancer/blas_l3_gemm,
  ../../laser/primitives/matrix_multiplication/gemm

const
  M     = 16*6*20
  K     = 16*6*20
  N     = 16*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big
  CpuGhz = 3.6      # i9-9980XE OC All turbo 4.1GHz (AVX2 4.0GHz, AVX512 3.6GHz)
  NumCpuCores = 18
  CpuFlopCycle = 64 # AVX2: 2xFMA/cycle = 2x8x2 - 2 x 8 floats x (1 add + 1 mul)

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(float32) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

# #############################################
# C and C++ FFI

import ospaths, strutils
from os import DirSep

const cSourcesPath = currentSourcePath.rsplit(DirSep, 1)[0] & '/'

# #############################################

proc benchOpenBLAS(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)
  bench("OpenBLAS benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm(
      rowMajor, noTranspose, noTranspose,
      M, N, K,
      1, a[0].unsafeaddr, K,
      b[0].unsafeAddr, N,
      0, result[0].addr, N
    )

proc benchArraymancerFallback(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)
  bench("Arraymancer fallback BLAS"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'f32,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'f32, result, 0, N, 1
    )

proc benchSimpleTiling(a, b: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  result = newSeq[float32](out_size)

  let pa = a[0].unsafeAddr
  let pb = b[0].unsafeAddr
  let pr = result[0].addr
  const blck = 32

  bench("Simple Tiling"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size) # We zero memory between computation
  do:
    {.emit: """
      #define min(a,b) (((a)<(b))?(a):(b))

      float (* __restrict A)[`K`] = (void*)`pa`;
      float (* __restrict B)[`N`] = (void*)`pb`;
      float (* __restrict C)[`N`] = (void*)`pr`;

      #pragma omp parallel
      #pragma omp single
      for (int j = 0; j < `N`; j+=`blck`)
        for (int k = 0; k < `K`; k+=`blck`)
          for (int i = 0; i < `M`; i+=`blck`)
      #pragma omp task \
            depend(in: A[i:`blck`][k:`blck`], B[k:`blck`][j:`blck`]) \
            depend(inout: C[i:`blck`][j:`blck`])
            for (int ii = i; ii<min(i+`blck`, `M`); ++ii)
              for (int jj = j; jj<min(j+`blck`, `N`); ++jj)
                for (int kk = k; kk<min(k+`blck`, `K`); ++kk)
                  C[ii][jj] += A[ii][kk] * B[kk][jj];

    """.}

proc benchLaserGEMM(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  let a_ptr{.restrict.} = a[0].unsafeAddr
  let b_ptr{.restrict.} = b[0].unsafeAddr
  let c_ptr{.restrict.} = result[0].addr
  bench("Laser production implementation"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_strided(
      M, N, K,
      1'f32,  a_ptr, K, 1,       # stride row, stride col
              b_ptr, N, 1,
      0'f32,  c_ptr, N, 1
    )

{.passC: "-I" & cSourcesPath & "pytorch_glow/".}

import pytorch_glow/libjit_matmul
  # Hack due to conflicts between "-std=c++11" requires by Glow
  # and incompatible with C files in cpuinfo.
  # We can't use the proper:
    # {.compile: "pytorch_glow/libjit_matmul.cpp".}
    # {.passC: "-std=c++11 -mavx -mfma".}
    # ^^^ This is configured in nim.cfg instead

proc libjit_matmul_f(
          c, a, b: ptr float32,
          cDims, aDims, bDims: ptr array[2, int]
      ) {.importc, cdecl.}
  # Note: Matrix C will be zero-mem'ed by libjit

proc benchPyTorchGlow(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  let a_ptr{.restrict.} = a[0].unsafeAddr
  let b_ptr{.restrict.} = b[0].unsafeAddr
  let c_ptr{.restrict.} = result[0].addr

  let cDims = [M, N]
  let aDims = [M, K]
  let bDims = [K, N]

  let cDims_ptr{.restrict.} = cDims.unsafeAddr
  let aDims_ptr{.restrict.} = aDims.unsafeAddr
  let bDims_ptr{.restrict.} = bDims.unsafeAddr

  bench("PyTorch Glow: libjit matmul implementation"):
    discard # zeroMem done by libjit
  do:
    # Main work
    libjit_matmul_f(
      c_ptr, a_ptr, b_ptr,
      cDims_ptr, aDims_ptr, bDims_ptr
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
  import ../../laser/private/error_functions

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
    let a = newSeqWith(M*K, float32 rand(1.0))
    let b = newSeqWith(K*N, float32 rand(1.0))

    # benchSimpleTiling(a, b, NbSamples)
    # benchArraymancerFallback(a, b, NbSamples)
    let reference = benchOpenBLAS(a, b, NbSamples)
    let challenger = benchLaserGEMM(a, b, NbSamples)
    # benchPyTorchGlow(a, b, NbSamples)

    block:
      var error = mean_relative_error(challenger, reference)
      doAssert error <= 1e-3'f32, $error

# Seems like my original Arraymancer BLAS has false sharing issue
# FYI Apple accelerate is about 117~122GFLOP/s on my machine.

###############################
# Compilation command
# $ nim cpp -r -d:release -d:openmp -o:build/bench_gemm benchmarks/gemm/gemm_bench_float32.nim

# Don't forget to add OpenBLAS in your path:
# For example on Mac with OpenBLAS from Homebrew
# `export LD_LIBRARY_PATH=/usr/local/opt/openblas/lib`

###############################
# OpenMP

# i9_9980XE Skylake-X 18 cores overclocked 4.1 GHz all-turbo, 4.0 GHz AVX turbo, 3.6 GHz AVX512 turbo
# PyTorch Glow compiled with AVX2 as AVX512 is slower
# Warmup: 0.9018 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 3840, N: 3840)
# B matrix shape: (M: 3840, N: 3840)
# Output shape: (M: 3840, N: 3840)
# Required number of operations: 113246.208 millions
# Required bytes:                  117.965 MB
# Arithmetic intensity:            960.000 FLOP/byte
# Theoretical peak single-core:    230.400 GFLOP/s
# Theoretical peak multi:         4147.200 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.504 seconds
# Average time: 49.841 ms
# Stddev  time: 4.290 ms
# Min     time: 48.066 ms
# Max     time: 61.994 ms
# Perf:         2272.149 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1965942382812

# Laser production implementation
# Collected 10 samples in 0.653 seconds
# Average time: 64.678 ms
# Stddev  time: 2.742 ms
# Min     time: 63.140 ms
# Max     time: 71.649 ms
# Perf:         1750.928 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1968383789062

# PyTorch Glow: libjit matmul implementation
# Collected 10 samples in 16.555 seconds
# Average time: 1655.510 ms
# Stddev  time: 0.204 ms
# Min     time: 1655.276 ms
# Max     time: 1655.983 ms
# Perf:         68.406 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1965942382812

###############################
# i5-5227U 2.7 GHz Broadwell dual core AVX2

# $  ./build/bench_gemm
# Warmup: 1.1900 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:     86.400 GFLOP/s
# Theoretical peak multi:          172.800 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.977 seconds
# Average time: 97.464 ms
# Stddev  time: 1.479 ms
# Min     time: 95.994 ms
# Max     time: 101.073 ms
# Perf:         145.241 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781372070312

# Laser production implementation
# Collected 10 samples in 1.039 seconds
# Average time: 103.659 ms
# Stddev  time: 4.858 ms
# Min     time: 97.393 ms
# Max     time: 114.670 ms
# Perf:         136.561 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.778076171875

###############################
# Serial - Nim code compiled without -d:openmp
# i9_9980XE Skylake-X 18 cores overclocked 4.1 GHz all-turbo, 4.0 GHz AVX turbo, 3.6 GHz AVX512 turbo
# PyTorch Glow compiled with AVX2 as AVX512 is slower
# For some reason OPENBLAS_NUM_THREADS=1 is ignore on Linux ...

# # $ OPENBLAS_NUM_THREADS=1 ./build/bench_gemm
# Warmup: 0.9034 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 3840, N: 3840)
# B matrix shape: (M: 3840, N: 3840)
# Output shape: (M: 3840, N: 3840)
# Required number of operations: 113246.208 millions
# Required bytes:                  117.965 MB
# Arithmetic intensity:            960.000 FLOP/byte
# Theoretical peak single-core:    230.400 GFLOP/s
# Theoretical peak multi:         4147.200 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.499 seconds
# Average time: 49.279 ms
# Stddev  time: 3.924 ms
# Min     time: 47.855 ms
# Max     time: 60.436 ms
# Perf:         2298.061 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1965942382812

# Laser production implementation
# Collected 10 samples in 6.828 seconds
# Average time: 682.218 ms
# Stddev  time: 9.549 ms
# Min     time: 667.896 ms
# Max     time: 693.479 ms
# Perf:         165.997 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1968383789062

# PyTorch Glow: libjit matmul implementation
# Collected 10 samples in 17.060 seconds
# Average time: 1705.967 ms
# Stddev  time: 0.332 ms
# Min     time: 1705.659 ms
# Max     time: 1706.847 ms
# Perf:         66.382 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 950.1965942382812

###############################
# i5-5227U 2.7 GHz Broadwell dual core AVX2

# $ OPENBLAS_NUM_THREADS=1 ./build/bench_gemm
# Warmup: 1.1973 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:     86.400 GFLOP/s
# Theoretical peak multi:          172.800 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 1.893 seconds
# Average time: 189.041 ms
# Stddev  time: 2.558 ms
# Min     time: 186.120 ms
# Max     time: 193.507 ms
# Perf:         74.882 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781372070312

# Laser production implementation
# Collected 10 samples in 1.942 seconds
# Average time: 193.975 ms
# Stddev  time: 4.279 ms
# Min     time: 190.571 ms
# Max     time: 205.327 ms
# Perf:         72.978 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.778076171875

# PyTorch Glow: libjit matmul implementation
# Collected 10 samples in 2.216 seconds
# Average time: 221.567 ms
# Stddev  time: 2.270 ms
# Min     time: 218.499 ms
# Max     time: 225.679 ms
# Perf:         63.889 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.778076171875

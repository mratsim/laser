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
  ../blas,
  ./arraymancer/blas_l3_gemm,
  ../../laser/primitives/matrix_multiplication/gemm

const
  M     = 16*6*20
  K     = 16*6*20
  N     = 16*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big
  CpuGhz = 2.7      # Assuming no turbo
  NumCpuCores = 2
  CpuFlopCycle = 32 # AVX2: 2xFMA/cycle = 2x8x2 - 2 x 8 floats x (1 add + 1 mul)

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

proc benchOpenBLAS(a, b: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
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

proc benchArraymancerFallback(a, b: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  bench("Arraymancer fallback BLAS"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'f32,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'f32, output, 0, N, 1
    )

proc benchSimpleTiling(a, b: seq[float32], nb_samples: int) {.noinline.}=
  var output = newSeq[float32](out_size)

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

      float (* restrict A)[`K`] = (void*)`pa`;
      float (* restrict B)[`N`] = (void*)`pb`;
      float (* restrict C)[`N`] = (void*)`po`;

      // TODO: where to parallelize?
      for (int j = 0; j < `N`; j+=`blck`)
        for (int k = 0; k < `K`; k+=`blck`)
          for (int i = 0; i < `M`; i++)
            for (int jj = j; jj<min(j+`blck`, `N`); jj++)
              for (int kk = k; kk<min(k+`blck`, `K`); kk++)
                C[i][jj] += A[i][kk] * B[kk][jj];

    """.}

proc benchLaserGEMM(a, b: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)

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
      ) {.importc.}
  # Note: Matrix C will be zero-mem'ed by libjit

proc benchPyTorchGlow(a, b: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)

  let a_ptr{.restrict.} = a[0].unsafeAddr
  let b_ptr{.restrict.} = b[0].unsafeAddr
  let c_ptr{.restrict.} = output[0].addr

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

    # when not defined(openmp):
    #   benchSimpleTiling(a, b, NbSamples) # for some reason stalled with OpenMP
    # benchArraymancerFallback(a, b, NbSamples)
    benchOpenBLAS(a, b, NbSamples)
    benchLaserGEMM(a, b, NbSamples)
    benchPyTorchGlow(a, b, NbSamples)

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

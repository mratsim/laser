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
  ../third_party/blas,
  ./arraymancer/blas_l3_gemm,
  ../../laser/primitives/matrix_multiplication/gemm

const
  M     = 16*6*20
  K     = 16*6*20
  N     = 16*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big
  CpuGhz = 3.5      # i9-9980XE OC All turbo 4.1GHz (AVX2 4.0GHz, AVX512 3.5GHz)
  NumCpuCores = 18
  VectorWidth = 16  # 8 float32 for AVX2, 16 for AVX512
  InstrCycle = 2    # How many instructions per cycle, (2xFMAs or 1xFMA for example)
  FlopInstr = 2     # How many FLOP per instr (FMAs = 1 add + 1 mul)

  TheoSerialPeak = CpuGhz * VectorWidth * InstrCycle * FlopInstr
  TheoThreadedPeak = TheoSerialPeak * NumCpuCores

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
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
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
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'f32,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'f32, result, 0, N, 1
    )

proc benchReference(a, b: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  result = newSeq[float32](out_size)

  let pa = a[0].unsafeAddr
  let pb = b[0].unsafeAddr
  let pr = result[0].addr

  bench("Reference loop"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    {.emit: """
      float (* __restrict A)[`K`] = (void*)`pa`;
      float (* __restrict B)[`N`] = (void*)`pb`;
      float (* __restrict C)[`N`] = (void*)`pr`;

      for (int i = 0; i < `M`; ++i)
        for (int k = 0; k < `K`; ++k)
          for (int j = 0; j < `N`; ++j)
            C[i][j] += A[i][k] * B[k][j];

    """.}

proc benchSimpleTiling(a, b: seq[float32], nb_samples: int): seq[float32] {.noinline.}=
  result = newSeq[float32](out_size)

  let pa = a[0].unsafeAddr
  let pb = b[0].unsafeAddr
  let pr = result[0].addr
  const blck = 32

  bench("Simple Tiling"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    {.emit: """
      #define min(a,b) (((a)<(b))?(a):(b))

      float (* __restrict A)[`K`] = (void*)`pa`;
      float (* __restrict B)[`N`] = (void*)`pb`;
      float (* __restrict C)[`N`] = (void*)`pr`;

      #pragma omp parallel
      #pragma omp single
      for (int i = 0; i < `M`; i+=`blck`)
        for (int k = 0; k < `K`; k+=`blck`)
          for (int j = 0; j < `N`; j+=`blck`)
      #pragma omp task \
            depend(in: A[i:`blck`][k:`blck`], B[k:`blck`][j:`blck`]) \
            depend(inout: C[i:`blck`][j:`blck`])
            for (int ii = i; ii<min(i+`blck`, `M`); ++ii)
              for (int kk = k; kk<min(k+`blck`, `K`); ++kk)
                for (int jj = j; jj<min(j+`blck`, `N`); ++jj)
                  C[ii][jj] += A[ii][kk] * B[kk][jj];

    """.}

proc benchLaserGEMM(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  let a_ptr{.restrict.} = a[0].unsafeAddr
  let b_ptr{.restrict.} = b[0].unsafeAddr
  let c_ptr{.restrict.} = result[0].addr
  bench("Laser production implementation"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    # Main work
    gemm_strided(
      M, N, K,
      1'f32,  a_ptr, K, 1,       # stride row, stride col
              b_ptr, N, 1,
      0'f32,  c_ptr, N, 1
    )

import ../third_party/pytorch_glow/libjit_matmul
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

  bench("PyTorch Glow: libjit matmul implementation (with AVX+FMA)"):
    discard # zeroMem done by libjit
  do:
    # Main work
    libjit_matmul_f(
      c_ptr, a_ptr, b_ptr,
      cDims_ptr, aDims_ptr, bDims_ptr
    )

import ../third_party/mkldnn
proc benchMkldnnRef(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  var # MKL-DNN wants pointers as input
    trans = 'N'
    m = int32 M
    n = int32 N
    k = int32 K
    alpha = 1'f32
    lda = int32 K
    ldb = int32 N
    beta = 0'f32
    ldc = int32 N

  bench("MKL-DNN reference GEMM benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    # Main work
    discard mkldnn_ref_gemm(
      trans.addr, trans.addr,
      m.addr, n.addr, k.addr,
      alpha.addr, a[0].unsafeaddr, lda.addr,
                  b[0].unsafeAddr, ldb.addr,
      beta.addr,  result[0].addr, ldc.addr,
                  bias = nil
    )

proc benchMkldnnJitAVX(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  var # MKL-DNN wants pointers as input
    trans = 'N'
    m = int32 M
    n = int32 N
    k = int32 K
    alpha = 1'f32
    lda = int32 K
    ldb = int32 N
    beta = 0'f32
    ldc = int32 N

  bench("MKL-DNN JIT AVX benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    # Main work
    discard mkldnn_jit_avx_gemm_f32(
      trans.addr, trans.addr,
      m.addr, n.addr, k.addr,
      alpha.addr, a[0].unsafeaddr, lda.addr,
                  b[0].unsafeAddr, ldb.addr,
      beta.addr,  result[0].addr, ldc.addr,
                  bias = nil
    )

proc benchMkldnnJitAVX512(a, b: seq[float32], nb_samples: int): seq[float32] =
  result = newSeq[float32](out_size)

  var # MKL-DNN wants pointers as input
    trans = 'N'
    m = int32 M
    n = int32 N
    k = int32 K
    alpha = 1'f32
    lda = int32 K
    ldb = int32 N
    beta = 0'f32
    ldc = int32 N

  bench("MKL-DNN JIT AVX512 benchmark"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(float32)) # We zero memory between computation
  do:
    # Main work
    discard mkldnn_jit_avx512_common_gemm_f32(
      trans.addr, trans.addr,
      m.addr, n.addr, k.addr,
      alpha.addr, a[0].unsafeaddr, lda.addr,
                  b[0].unsafeAddr, ldb.addr,
      beta.addr,  result[0].addr, ldc.addr,
                  bias = nil
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
    let a = newSeqWith(M*K, float32 rand(-0.1..0.1))
    let b = newSeqWith(K*N, float32 rand(-0.1..0.1))

    # let reference = benchReference(a, b, NbSamples)
    # let simpleTiling = benchSimpleTiling(a, b, NbSamples)
    # let arraymancer = benchArraymancerFallback(a, b, NbSamples)
    let laser = benchLaserGEMM(a, b, NbSamples)
    # let vendorBlas = benchOpenBLAS(a, b, NbSamples)
    # let glow = benchPyTorchGlow(a, b, NbSamples)
    # let mkldnnref = benchMkldnnRef(a, b, NbSamples)
    # let mkldnnjitavx = benchMkldnnJitAVX(a, b, NbSamples)
    # let mkldnnjitavx512 = benchMkldnnJitAVX512(a, b, NbSamples)

    # block:
    #   # var error = mean_relative_error(vendorBlas, reference)
    #   # echo "Mean Relative Error of OpenBLAS vs reference: ", error
    #   # doAssert error <= 1e-5'f32, $error

    #   # error = mean_relative_error(challenger, reference)
    #   # echo "Mean Relative Error compared to Reference: ", error
    #   # doAssert error <= 1e-5'f32, $error

    #   var error = mean_relative_error(vendorBlas, laser)
    #   echo "Mean Relative Error compared to vendor BLAS: ", error
    #   doAssert error <= 1e-5'f32, $error

# Seems like my original Arraymancer BLAS has false sharing issue
# FYI Apple accelerate is about 117~122GFLOP/s on my machine.

###############################
# Compilation command
# $ nim cpp -r -d:release -d:danger -d:openmp --outdir:build benchmarks/gemm/gemm_bench_float32.nim

# Don't forget to add OpenBLAS in your path:
# For example on Mac with OpenBLAS from Homebrew
# `export LD_LIBRARY_PATH=/usr/local/opt/openblas/lib`

###############################
# OpenMP

# i9_9980XE Skylake-X 18 cores overclocked 4.1 GHz all-turbo, 4.0 GHz AVX turbo, 3.5 GHz AVX512 turbo
# PyTorch Glow compiled with AVX2 as AVX512 is slower

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:    224.000 GFLOP/s
# Theoretical peak multi:         4032.000 GFLOP/s
# Make sure to not bench Apple Accelerate or the default Linux BLAS.

# OpenBLAS benchmark
# Collected 10 samples in 0.089 seconds
# Average time: 8.172 ms
# Stddev  time: 5.513 ms
# Min     time: 6.410 ms
# Max     time: 23.863 ms
# Perf:         1732.227 GFLOP/s

# Laser production implementation
# Collected 10 samples in 0.082 seconds
# Average time: 7.553 ms
# Stddev  time: 4.509 ms
# Min     time: 5.866 ms
# Max     time: 20.314 ms
# Perf:         1874.073 GFLOP/s

# PyTorch Glow: libjit matmul implementation (with AVX+FMA)
# Collected 10 samples in 2.042 seconds
# Average time: 204.186 ms
# Stddev  time: 0.598 ms
# Min     time: 203.783 ms
# Max     time: 205.815 ms
# Perf:         69.328 GFLOP/s

# MKL-DNN reference GEMM benchmark
# Collected 10 samples in 0.331 seconds
# Average time: 32.286 ms
# Stddev  time: 4.983 ms
# Min     time: 30.018 ms
# Max     time: 46.264 ms
# Perf:         438.449 GFLOP/s

# MKL-DNN JIT AVX benchmark
# Collected 10 samples in 0.105 seconds
# Average time: 9.752 ms
# Stddev  time: 5.647 ms
# Min     time: 7.749 ms
# Max     time: 25.768 ms
# Perf:         1451.603 GFLOP/s

# MKL-DNN JIT AVX512 benchmark
# Collected 10 samples in 0.088 seconds
# Average time: 8.148 ms
# Stddev  time: 10.751 ms
# Min     time: 4.572 ms
# Max     time: 38.731 ms
# Perf:         1737.346 GFLOP/s
# Mean Relative Error compared to vendor BLAS: 3.045843413929106e-06

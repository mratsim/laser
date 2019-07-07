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
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GINTOP/s"

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
  ./arraymancer/blas_l3_gemm,
  ../../laser/primitives/matrix_multiplication/gemm

const
  M     = 16*6*20
  K     = 16*6*20
  N     = 16*6*20
  NbSamples = 10    # This might stresss the allocator when packing if the matrices are big

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(int32) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

# #############################################

proc benchArraymancerFallback(a, b: seq[int32], nb_samples: int): seq[int32] =
  result = newSeq[int32](out_size)
  bench("Arraymancer fallback BLAS"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(int32)) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'i32,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'i32, result, 0, N, 1
    )

proc benchSimpleTiling(a, b: seq[int32], nb_samples: int): seq[int32] {.noinline.}=
  result = newSeq[int32](out_size * sizeof(int32))

  let pa = a[0].unsafeAddr
  let pb = b[0].unsafeAddr
  let po = result[0].addr
  const blck = 32

  bench("Simple Tiling"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(result[0].addr, out_size * sizeof(int32)) # We zero memory between computation
  do:
    {.emit: """
      #define min(a,b) (((a)<(b))?(a):(b))

      int32_t (* __restrict A)[`K`] = (void*)`pa`;
      int32_t (* __restrict B)[`N`] = (void*)`pb`;
      int32_t (* __restrict C)[`N`] = (void*)`po`;

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

proc benchLaserGEMM(a, b: seq[int32], nb_samples: int): seq[int32] =
  result = newSeq[int32](out_size * sizeof(int32))

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
      1'i32,  a_ptr, K, 1,       # stride row, stride col
              b_ptr, N, 1,
      0'i32,  c_ptr, N, 1
    )

# import ../third_party/mkldnn
# proc benchMkldnnRef(a, b: seq[int32], nb_samples: int): seq[int32] =
#   result = newSeq[int32](out_size * sizeof(int32))

#   var # MKL-DNN wants pointers as input
#     trans = 'N'
#     m = int32 M
#     n = int32 N
#     k = int32 K
#     alpha = 1'i32
#     lda = int32 K
#     ldb = int32 N
#     beta = 0'i32
#     ldc = int32 N

#     bias = newSeq[int32](N)


#   bench("MKL-DNN reference GEMM benchmark (note that it also add a bias)"):
#     # Initialisation, not measured apart for the "Collected n samples in ... seconds"
#     zeroMem(result[0].addr, out_size * sizeof(int32)) # We zero memory between computation
#   do:
#     # Main work
#     discard mkldnn_ref_gemm(
#       trans.addr, trans.addr,
#       m.addr, n.addr, k.addr,
#       alpha.addr, a[0].unsafeaddr, lda.addr,
#                   b[0].unsafeAddr, ldb.addr,
#       beta.addr,  result[0].addr, ldc.addr,
#                   bias[0].addr
#     )

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
  echo &"Theoretical peak single-core:  Unknown"
  echo &"Theoretical peak multi:        Unknown"
  block:
    let a = newSeqWith(M*K, int32 rand(100))
    let b = newSeqWith(K*N, int32 rand(100))

    discard benchSimpleTiling(a, b, NbSamples)
    # discard benchArraymancerFallback(a, b, NbSamples)
    discard benchLaserGEMM(a, b, NbSamples)
    # discard benchMkldnnRef(a, b, NbSamples)

# ####################################################

# Serial (which is faster than OMP probably because of false sharing)

# Warmup: 1.1935 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:  Unknown
# Theoretical peak multi:        Unknown

# Simple Tiling
# Collected 10 samples in 24.949 seconds
# Average time: 2494.638 ms
# Stddev  time: 36.238 ms
# Min     time: 2459.590 ms
# Max     time: 2570.874 ms
# Perf:         5.674 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# Arraymancer fallback BLAS
# Collected 10 samples in 21.870 seconds
# Average time: 2186.740 ms
# Stddev  time: 26.963 ms
# Min     time: 2161.838 ms
# Max     time: 2246.702 ms
# Perf:         6.473 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# Laser production implementation
# Collected 10 samples in 9.853 seconds
# Average time: 985.001 ms
# Stddev  time: 4.657 ms
# Min     time: 979.731 ms
# Max     time: 997.174 ms
# Perf:         14.371 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# ########################################################
# OpenMP

# Warmup: 1.1920 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:  Unknown
# Theoretical peak multi:        Unknown

# Arraymancer fallback BLAS
# Collected 10 samples in 26.885 seconds
# Average time: 2688.255 ms
# Stddev  time: 23.790 ms
# Min     time: 2665.262 ms
# Max     time: 2733.505 ms
# Perf:         5.266 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# Laser production implementation
# Collected 10 samples in 5.077 seconds
# Average time: 507.399 ms
# Stddev  time: 7.426 ms
# Min     time: 499.704 ms
# Max     time: 523.043 ms
# Perf:         27.899 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

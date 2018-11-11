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
  ./laser/laser_gemm

const
  M     =  224
  K     =  16*3*20*3*3 # to make required ops similar to conv
  N     =  224

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(float32) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

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

proc benchSimpleTiling(a, b: seq[float32], nb_samples: int) =
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
  bench("New Laser GEMM implementation - Generic SIMD"):
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
  block:
    let a = newSeqWith(M*K, float32 rand(1.0))
    let b = newSeqWith(K*N, float32 rand(1.0))

    benchOpenBLAS(a, b, nb_samples = 20)
    benchArraymancerFallback(a, b, nb_samples = 20)
    benchSimpleTiling(a, b, nb_samples = 20)
    benchLaserGEMM(a, b, nb_samples = 20)

# Seems like my BLAS has false sharing issue

###############################
# OpenMP

# Warmup: 1.3832 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 224, N: 8640)
# B matrix shape: (M: 8640, N: 224)
# Output shape: (M: 224, N: 224)
# Required number of operations:   867.041 millions
# Required bytes:                   15.483 MB
# Arithmetic intensity:             56.000 FLOP/byte

# OpenBLAS benchmark
# Collected 20 samples in 0.412 seconds
# Average time: 20.614 ms
# Stddev  time: 5.328 ms
# Min     time: 12.964 ms
# Max     time: 33.944 ms
# Perf:         42.060 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 2158.1591796875

# Arraymancer fallback BLAS
# Collected 20 samples in 4.727 seconds
# Average time: 236.333 ms
# Stddev  time: 24.520 ms
# Min     time: 212.795 ms
# Max     time: 313.838 ms
# Perf:         3.669 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 2158.158935546875

###############################
# Serial

# OpenBLAS benchmark
# Collected 20 samples in 0.558 seconds
# Average time: 27.884 ms
# Stddev  time: 10.439 ms
# Min     time: 14.651 ms
# Max     time: 47.797 ms
# Perf:         31.094 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 2158.1591796875

# Arraymancer fallback BLAS
# Collected 20 samples in 1.925 seconds
# Average time: 96.261 ms
# Stddev  time: 33.895 ms
# Min     time: 74.643 ms
# Max     time: 201.256 ms
# Perf:         9.007 GFLOP/s

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
  M     =  16*6*20  # 1500
  K     =  16*6*20  # 1500 # 16*3*20*3*3 # to make required ops similar to conv
  N     =  16*6*20  # 1500
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
  block:
    let a = newSeqWith(M*K, float32 rand(1.0))
    let b = newSeqWith(K*N, float32 rand(1.0))

    when not defined(openmp):
      benchSimpleTiling(a, b, NbSamples) # for some reason stalled with OpenMP
    benchArraymancerFallback(a, b, NbSamples)
    benchOpenBLAS(a, b, NbSamples)
    benchLaserGEMM(a, b, NbSamples)

# Seems like my original Arraymancer BLAS has false sharing issue

###############################
# OpenMP (The simple loop tiling doesn't run properly for unknown reason so deactivated)

# Warmup: 1.1915 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:     86.400 GFLOP/s
# Theoretical peak multi:          172.800 GFLOP/s

# Arraymancer fallback BLAS
# Collected 10 samples in 24.931 seconds
# Average time: 2492.867 ms
# Stddev  time: 42.635 ms
# Min     time: 2459.191 ms
# Max     time: 2587.520 ms
# Perf:         5.679 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.778076171875

# OpenBLAS benchmark
# Collected 10 samples in 1.101 seconds
# Average time: 109.882 ms
# Stddev  time: 3.064 ms
# Min     time: 107.750 ms
# Max     time: 117.494 ms
# Perf:         128.827 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781677246094

# New Laser GEMM implementation - Generic SIMD
# Collected 10 samples in 1.205 seconds
# Average time: 120.216 ms
# Stddev  time: 2.914 ms
# Min     time: 117.720 ms
# Max     time: 127.660 ms
# Perf:         117.753 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781677246094

###############################
# Serial (openBLAS is still parallel)

# Warmup: 1.1933 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:     86.400 GFLOP/s
# Theoretical peak multi:          172.800 GFLOP/s

# Simple Tiling
# Collected 10 samples in 23.062 seconds
# Average time: 2305.908 ms
# Stddev  time: 8.366 ms
# Min     time: 2297.823 ms
# Max     time: 2326.359 ms
# Perf:         6.139 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7776184082031

# Arraymancer fallback BLAS
# Collected 10 samples in 9.289 seconds
# Average time: 928.651 ms
# Stddev  time: 26.071 ms
# Min     time: 911.806 ms
# Max     time: 997.996 ms
# Perf:         15.243 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.778076171875

# OpenBLAS benchmark
# Collected 10 samples in 1.107 seconds
# Average time: 110.437 ms
# Stddev  time: 3.799 ms
# Min     time: 107.482 ms
# Max     time: 119.737 ms
# Perf:         128.180 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781677246094

# New Laser GEMM implementation - Generic SIMD
# Collected 10 samples in 2.177 seconds
# Average time: 217.430 ms
# Stddev  time: 9.996 ms
# Min     time: 208.479 ms
# Max     time: 233.767 ms
# Perf:         65.105 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 470.7781677246094

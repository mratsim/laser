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
  # CpuIntopCycle = # Unknown

const
  ashape: MatrixShape = (M, K)
  bshape: MatrixShape = (K, N)

let req_ops = gemm_required_ops(ashape, bshape)
let req_bytes = sizeof(int32) * gemm_required_data(ashape, bshape)

let out_shape: MatrixShape = gemm_out_shape(ashape, bshape)
let out_size = out_shape.M * out_shape.N

# #############################################

proc benchArraymancerFallback(a, b: seq[int32], nb_samples: int) =
  var output = newSeq[int32](out_size)
  bench("Arraymancer fallback BLAS"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    gemm_nn_fallback(
      M, N, K,
      1'i32,      a, 0, K, 1,       # offset, stride row, stride col
                  b, 0, N, 1,
      0'i32, output, 0, N, 1
    )

proc benchSimpleTiling(a, b: seq[int32], nb_samples: int) {.noinline.}=
  var output = newSeq[int32](out_size)

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

      int32_t (* restrict A)[`K`] = (void*)`pa`;
      int32_t (* restrict B)[`N`] = (void*)`pb`;
      int32_t (* restrict C)[`N`] = (void*)`po`;

      // TODO: where to parallelize?
      for (int j = 0; j < `N`; j+=`blck`)
        for (int k = 0; k < `K`; k+=`blck`)
          for (int i = 0; i < `M`; i++)
            for (int jj = j; jj<min(j+`blck`, `N`); jj++)
              for (int kk = k; kk<min(k+`blck`, `K`); kk++)
                C[i][jj] += A[i][kk] * B[kk][jj];

    """.}

proc benchLaserGEMM(a, b: seq[int32], nb_samples: int) =
  var output = newSeq[int32](out_size)

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
      1'i32,  a_ptr, K, 1,       # stride row, stride col
              b_ptr, N, 1,
      0'i32,  c_ptr, N, 1
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
  echo &"Theoretical peak single-core:  Unknown" #{CpuGhz * CpuFlopCycle:>9.3f} GINTOP/s"
  echo &"Theoretical peak multi:        Unknown" #{CpuGhz * CpuFlopCycle * NumCpuCores:>9.3f} GINTOP/s"
  block:
    let a = newSeqWith(M*K, int32 rand(100))
    let b = newSeqWith(K*N, int32 rand(100))

    when not defined(openmp):
      benchSimpleTiling(a, b, NbSamples) # for some reason stalled with OpenMP
    benchArraymancerFallback(a, b, NbSamples)
    benchLaserGEMM(a, b, NbSamples)

# ####################################################

# Serial (which is faster than OMP probably because of false sharing)

# Warmup: 1.1932 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 1920, N: 1920)
# B matrix shape: (M: 1920, N: 1920)
# Output shape: (M: 1920, N: 1920)
# Required number of operations: 14155.776 millions
# Required bytes:                   29.491 MB
# Arithmetic intensity:            480.000 FLOP/byte
# Theoretical peak single-core:  Unknown
# Theoretical peak multi:        Unknown

# Simple Tiling
# Collected 10 samples in 26.355 seconds
# Average time: 2635.233 ms
# Stddev  time: 18.358 ms
# Min     time: 2600.231 ms
# Max     time: 2665.695 ms
# Perf:         5.372 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# Arraymancer fallback BLAS
# Collected 10 samples in 21.775 seconds
# Average time: 2177.268 ms
# Stddev  time: 10.250 ms
# Min     time: 2166.361 ms
# Max     time: 2193.774 ms
# Perf:         6.502 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

# Laser production implementation
# Collected 10 samples in 26.827 seconds
# Average time: 2682.440 ms
# Stddev  time: 11.262 ms
# Min     time: 2672.534 ms
# Max     time: 2709.101 ms
# Perf:         5.277 GINTOP/s

# Display output[0] to make sure it's not optimized away
# 4699774

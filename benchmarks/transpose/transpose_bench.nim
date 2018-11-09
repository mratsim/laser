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
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GMEMOPs/s"
  echo "\nDisplay output[1] to make sure it's not optimized away"
  echo output[1] # Prevents compiler from optimizing stuff away

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
  ./transpose_common, # ../blas,
  ./transpose_naive_tensor,
  ./transpose_divide_conquer,
  ../../laser/dynamic_stack_arrays,
  ../../laser/compiler_optim_hints

const
  M     =  4000
  N     =  2000
  NbSamples = 250

const
  ashape: MatrixShape = (M, N)

let req_ops = M*N
let req_bytes = sizeof(float32) * M*N

let out_shape: MatrixShape = (N, M)
let out_size = out_shape.M * out_shape.N

# #############################################

# TODO: could not import: cblas_somatcopy
# proc benchBLAS(a: seq[float32], nb_samples: int) =
#   var output = newSeq[float32](out_size)
#   bench("BLAS omatcopy benchmark"):
#     # Initialisation, not measured apart for the "Collected n samples in ... seconds"
#     zeroMem(output[0].addr, out_size) # We zero memory between computation
#   do:
#     # Main work
#     omatcopy(
#       rowMajor, noTranspose,
#       M, N, 1,
#       a[0].unsafeaddr, N,
#       output[0].unsafeAddr, N,
#     )

proc benchNaive(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = cast[ptr UncheckedArray[float32]](a[0].unsafeAddr)
  let po{.restrict.} = cast[ptr UncheckedArray[float32]](output[0].addr)

  bench("Naive transpose"):
    discard
  do:
    for i in `||`(0, M-1):
      for j in `||`(0, N-1, "simd"): # This only add "#pragma omp simd"
        po[i+j*M] = pa[j+i*N]
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchNaiveExchanged(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = cast[ptr UncheckedArray[float32]](a[0].unsafeAddr)
  let po{.restrict.} = cast[ptr UncheckedArray[float32]](output[0].addr)

  bench("Naive transpose - input row iteration"):
    discard
  do:
    for j in `||`(0, N-1):
      for i in `||`(0, M-1, "simd"): # This only add "#pragma omp simd"
        po[i+j*M] = pa[j+i*N]
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchForEachStrided(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)

  var ti = a.buildTensorView(M, N)
  ti.shape = ti.shape.reversed()
  ti.strides = ti.strides.reversed()

  var to = output.buildTensorView(N, M)

  bench("Laser ForEachStrided"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    zeroMem(output[0].addr, out_size) # We zero memory between computation
  do:
    # Main work
    transpose_naive_forEach(to, ti)
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCollapsed(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr

  bench("Collapsed OpenMP"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < `M`; i++)
      for (int j = 0; j < `N`; j++)
        `po`[i+j*`M`] = `pa`[j+i*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCollapsedExchanged(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr

  bench("Collapsed OpenMP - input row iteration"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for simd collapse(2)
    for (int j = 0; j < `N`; j++)
      for (int i = 0; i < `M`; i++)
        `po`[i+j*`M`] = `pa`[j+i*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCacheBlocking(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 64

  bench("Cache blocking"):
    discard
  do:
    {.emit: """
    // No min function in C ...
    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `N`; ++j)
        #pragma omp simd
        for (int ii = i; ii < min(i+`blck`,`M`); ++ii)
          `po`[ii+j*`M`] = `pa`[j+ii*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCacheBlockingExchanged(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 64

  bench("Cache blocking - input row iteration"):
    discard
  do:
    {.emit: """
    // No min function in C ...
    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; ++i)
        #pragma omp simd
        for (int jj = j; jj < min(j+`blck`,`N`); ++jj)
          `po`[i+(jj)*`M`] = `pa`[jj+i*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc bench2Dtiling(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 128

  bench("2D Tiling"):
    discard
  do:
    {.emit: """
    // No min function in C ...
    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `N`; j+=`blck`)
        for (int ii = i; ii<i+`blck` && ii<`M`; ii++)
          #pragma omp simd
          for (int jj = j; jj<min(j+`blck`,`N`); jj++)
            `po`[ii+jj*`M`] = `pa`[jj+ii*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc bench2DtilingExchanged(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 64

  bench("2D Tiling - input row iteration"):
    discard
  do:
    {.emit: """
    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`N`; jj++)
          #pragma omp simd
          for (int ii = i; ii<min(i+`blck`,`M`); ii++)
            `po`[ii+jj*`M`] = `pa`[jj+ii*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCacheBlockingPrefetch(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 32

  # This seems to trigger constant and loop folding
  # and does not seem to be replicable if not defined inline
  bench("Cache blocking with Prefetch"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `N`; ++j)
        #pragma omp simd
        for (int ii = i; ii<min(i+`blck`,`M`); ii++)
          `po`[ii+j*`M`] = `pa`[j+ii*`N`];
      __builtin_prefetch(&`pa`[(i+1)*`N`], 0, 1); // Prefetch read with low temporal locality
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc bench2DtilingExchangedPrefetch(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 32

  bench("2D Tiling + Prefetch - input row iteration"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`N`; jj++)
          #pragma omp simd
          for (int ii = i; ii<min(i+`blck`,`M`); ii++)
            `po`[ii+jj*`M`] = `pa`[jj+ii*`N`];
        __builtin_prefetch(&`pa`[(i+1)*`N`], 0, 1); // Prefetch read with low temporal locality
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

import ../../laser/primitives/swapaxes
proc benchProdImpl(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)

  bench("Production implementation"):
    discard
  do:
    transpose2D_copy(output[0].addr, a[0].unsafeAddr, M, N)

# TODO buggy
# proc benchCacheOblivious(a: seq[float32], nb_samples: int) =
#   var output = newSeq[float32](out_size)
#
#   let a_ptr{.restrict.} = cast[ptr UncheckedArray[float32]](a.unsafeAddr)
#   let o_ptr{.restrict.} = cast[ptr UncheckedArray[float32]](output.addr)
#
#   bench("Cache oblivious recursive"):
#     discard
#   do:
#     # Main work
#     transpose_cache_oblivious(o_ptr, a_ptr, M, N)
#     # echo a.toString((M, N))
#     # echo output.toString((N, M))

# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  echo ""
  echo "A matrix shape: " & $ashape
  echo "Output shape: " & $out_shape
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  block:
    let a = newSeqWith(M*N, float32 rand(1.0))

    # benchBLAS(a, NbSamples)
    benchForEachStrided(a, NbSamples)
    benchNaive(a, NbSamples)
    benchNaiveExchanged(a, NbSamples)
    benchCollapsed(a, NbSamples)
    benchCollapsedExchanged(a, NbSamples)
    benchCacheBlocking(a, NbSamples)
    benchCacheBlockingExchanged(a, NbSamples)
    bench2Dtiling(a, NbSamples)
    bench2DtilingExchanged(a, NbSamples)
    benchCacheBlockingPrefetch(a, NbSamples)
    bench2DtilingExchangedPrefetch(a, NbSamples)
    benchProdImpl(a, NbSamples)
    # benchCacheOblivious(a, NbSamples)


## With OpenMP
## Note - OpenMP is faster when iterating on input row in inner loop
##        but in serial case its input col in inner loop that is faster
## Prefetch helps a lot in serial case but doesn't at all with OpenMP
## In serial mode, prefetch hints the compiler to unroll the loop
## as the bounds are known at compile-time

########################################################################
# OpenMP

# Warmup: 1.7140 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 4000, N: 2000)
# Output shape: (M: 2000, N: 4000)
# Required number of operations:     8.000 millions
# Required bytes:                   32.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 250 samples in 6.334 seconds
# Average time: 24.745 ms
# Stddev  time: 2.511 ms
# Min     time: 21.225 ms
# Max     time: 35.859 ms
# Perf:         0.323 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose
# Collected 250 samples in 4.781 seconds
# Average time: 19.121 ms
# Stddev  time: 1.315 ms
# Min     time: 15.100 ms
# Max     time: 30.852 ms
# Perf:         0.418 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose - input row iteration
# Collected 250 samples in 5.708 seconds
# Average time: 22.828 ms
# Stddev  time: 1.938 ms
# Min     time: 18.554 ms
# Max     time: 36.763 ms
# Perf:         0.350 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP
# Collected 250 samples in 4.806 seconds
# Average time: 19.223 ms
# Stddev  time: 1.576 ms
# Min     time: 16.045 ms
# Max     time: 34.059 ms
# Perf:         0.416 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP - input row iteration
# Collected 250 samples in 6.219 seconds
# Average time: 24.875 ms
# Stddev  time: 1.895 ms
# Min     time: 20.163 ms
# Max     time: 32.635 ms
# Perf:         0.322 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking
# Collected 250 samples in 2.307 seconds
# Average time: 9.225 ms
# Stddev  time: 0.939 ms
# Min     time: 7.918 ms
# Max     time: 17.105 ms
# Perf:         0.867 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking - input row iteration
# Collected 250 samples in 2.873 seconds
# Average time: 11.490 ms
# Stddev  time: 1.136 ms
# Min     time: 9.738 ms
# Max     time: 20.180 ms
# Perf:         0.696 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling
# Collected 250 samples in 2.021 seconds
# Average time: 8.081 ms
# Stddev  time: 0.884 ms
# Min     time: 6.884 ms
# Max     time: 13.235 ms
# Perf:         0.990 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling - input row iteration
# Collected 250 samples in 2.038 seconds
# Average time: 8.151 ms
# Stddev  time: 0.810 ms
# Min     time: 6.845 ms
# Max     time: 14.160 ms
# Perf:         0.981 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking with Prefetch
# Collected 250 samples in 2.392 seconds
# Average time: 9.565 ms
# Stddev  time: 1.150 ms
# Min     time: 7.705 ms
# Max     time: 17.514 ms
# Perf:         0.836 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling + Prefetch - input row iteration
# Collected 250 samples in 2.299 seconds
# Average time: 9.193 ms
# Stddev  time: 1.517 ms
# Min     time: 7.217 ms
# Max     time: 20.560 ms
# Perf:         0.870 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Production implementation
# Collected 250 samples in 2.206 seconds
# Average time: 8.824 ms
# Stddev  time: 0.943 ms
# Min     time: 7.388 ms
# Max     time: 15.261 ms
# Perf:         0.907 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

########################################################################
# Serial
########################################################################

# The cache blocking + prefetch seems to trigger constant and loop folding
# and does not seem to be replicable if not defined inline

# Warmup: 1.1947 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 4000, N: 2000)
# Output shape: (M: 2000, N: 4000)
# Required number of operations:     8.000 millions
# Required bytes:                   32.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 250 samples in 9.176 seconds
# Average time: 36.125 ms
# Stddev  time: 3.632 ms
# Min     time: 34.089 ms
# Max     time: 77.802 ms
# Perf:         0.221 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose
# Collected 250 samples in 7.090 seconds
# Average time: 28.360 ms
# Stddev  time: 1.315 ms
# Min     time: 26.625 ms
# Max     time: 37.690 ms
# Perf:         0.282 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose - input row iteration
# Collected 250 samples in 8.863 seconds
# Average time: 35.451 ms
# Stddev  time: 1.025 ms
# Min     time: 34.050 ms
# Max     time: 43.612 ms
# Perf:         0.226 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP
# Collected 250 samples in 7.342 seconds
# Average time: 29.367 ms
# Stddev  time: 2.905 ms
# Min     time: 26.573 ms
# Max     time: 46.812 ms
# Perf:         0.272 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP - input row iteration
# Collected 250 samples in 8.934 seconds
# Average time: 35.738 ms
# Stddev  time: 1.543 ms
# Min     time: 34.098 ms
# Max     time: 46.705 ms
# Perf:         0.224 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking
# Collected 250 samples in 2.940 seconds
# Average time: 11.758 ms
# Stddev  time: 0.754 ms
# Min     time: 10.696 ms
# Max     time: 18.864 ms
# Perf:         0.680 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking - input row iteration
# Collected 250 samples in 4.878 seconds
# Average time: 19.513 ms
# Stddev  time: 1.021 ms
# Min     time: 18.702 ms
# Max     time: 27.463 ms
# Perf:         0.410 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling
# Collected 250 samples in 3.225 seconds
# Average time: 12.901 ms
# Stddev  time: 0.747 ms
# Min     time: 12.030 ms
# Max     time: 18.211 ms
# Perf:         0.620 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling - input row iteration
# Collected 250 samples in 3.194 seconds
# Average time: 12.774 ms
# Stddev  time: 0.497 ms
# Min     time: 11.603 ms
# Max     time: 17.648 ms
# Perf:         0.626 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking with Prefetch
# Collected 250 samples in 2.595 seconds
# Average time: 10.381 ms
# Stddev  time: 0.830 ms
# Min     time: 9.449 ms
# Max     time: 18.195 ms
# Perf:         0.771 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling + Prefetch - input row iteration
# Collected 250 samples in 2.576 seconds
# Average time: 10.306 ms
# Stddev  time: 0.835 ms
# Min     time: 9.241 ms
# Max     time: 17.958 ms
# Perf:         0.776 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Production implementation
# Collected 250 samples in 3.403 seconds
# Average time: 13.613 ms
# Stddev  time: 0.791 ms
# Min     time: 12.026 ms
# Max     time: 20.408 ms
# Perf:         0.588 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318
# kami-no-itte:laser tesuji$ ./build/transpose_bench_serial
# Warmup: 1.1977 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 4000, N: 2000)
# Output shape: (M: 2000, N: 4000)
# Required number of operations:     8.000 millions
# Required bytes:                   32.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 250 samples in 8.877 seconds
# Average time: 34.944 ms
# Stddev  time: 0.836 ms
# Min     time: 34.210 ms
# Max     time: 40.388 ms
# Perf:         0.229 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose
# Collected 250 samples in 7.098 seconds
# Average time: 28.390 ms
# Stddev  time: 1.747 ms
# Min     time: 26.535 ms
# Max     time: 44.813 ms
# Perf:         0.282 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose - input row iteration
# Collected 250 samples in 8.873 seconds
# Average time: 35.491 ms
# Stddev  time: 0.821 ms
# Min     time: 34.104 ms
# Max     time: 41.378 ms
# Perf:         0.225 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP
# Collected 250 samples in 7.064 seconds
# Average time: 28.257 ms
# Stddev  time: 1.202 ms
# Min     time: 26.525 ms
# Max     time: 36.251 ms
# Perf:         0.283 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP - input row iteration
# Collected 250 samples in 9.003 seconds
# Average time: 36.012 ms
# Stddev  time: 1.561 ms
# Min     time: 34.149 ms
# Max     time: 50.441 ms
# Perf:         0.222 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking
# Collected 250 samples in 2.900 seconds
# Average time: 11.601 ms
# Stddev  time: 0.630 ms
# Min     time: 10.797 ms
# Max     time: 18.948 ms
# Perf:         0.690 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking - input row iteration
# Collected 250 samples in 4.911 seconds
# Average time: 19.644 ms
# Stddev  time: 1.005 ms
# Min     time: 18.644 ms
# Max     time: 27.572 ms
# Perf:         0.407 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling
# Collected 250 samples in 3.232 seconds
# Average time: 12.927 ms
# Stddev  time: 0.782 ms
# Min     time: 12.036 ms
# Max     time: 19.736 ms
# Perf:         0.619 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling - input row iteration
# Collected 250 samples in 3.197 seconds
# Average time: 12.786 ms
# Stddev  time: 0.652 ms
# Min     time: 11.448 ms
# Max     time: 19.202 ms
# Perf:         0.626 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking with Prefetch
# Collected 250 samples in 2.617 seconds
# Average time: 10.468 ms
# Stddev  time: 0.874 ms
# Min     time: 9.438 ms
# Max     time: 19.032 ms
# Perf:         0.764 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling + Prefetch - input row iteration
# Collected 250 samples in 2.630 seconds
# Average time: 10.520 ms
# Stddev  time: 1.414 ms
# Min     time: 9.234 ms
# Max     time: 22.264 ms
# Perf:         0.760 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Production implementation
# Collected 250 samples in 3.086 seconds
# Average time: 12.345 ms
# Stddev  time: 0.926 ms
# Min     time: 10.753 ms
# Max     time: 19.060 ms
# Perf:         0.648 GMEMOPs/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

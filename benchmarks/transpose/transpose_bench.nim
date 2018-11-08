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
    for i in `||`(0, M-1, "parallel for simd"):
      for j in 0 ..< N:
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
    for j in `||`(0, N-1, "parallel for simd"):
      for i in 0 ..< M:
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
    #pragma omp parallel for simd
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `N`; ++j)
        for (int b = 0; b < `blck` && i+b<`M`; ++b)
          `po`[i+j*`M` + b] = `pa`[j+(i+b)*`N`];
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
    #pragma omp parallel for simd
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; ++i)
        for (int b = 0; b < `blck` && j+b<`N`; ++b)
          `po`[i+(j+b)*`M`] = `pa`[j+b+i*`N`];
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
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `M`; j+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`N`; jj++)
          for (int ii = i; ii<i+`blck` && ii<`M`; ii++)
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
    #pragma omp parallel for simd collapse(2)
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`N`; jj++)
          for (int ii = i; ii<i+`blck` && ii<`M`; ii++)
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
    #pragma omp parallel for simd
    for (int i = 0; i < `M`; i+=`blck`)
      for (int j = 0; j < `N`; ++j)
        for (int b = 0; b < `blck` && i+b<`M`; ++b)
          `po`[i+j*`M` + b] = `pa`[j+(i+b)*`N`];
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
    #pragma omp parallel for simd collapse(2)
    for (int j = 0; j < `N`; j+=`blck`)
      for (int i = 0; i < `M`; i+=`blck`)
        for (int jj = j; jj<j+`blck` && jj<`N`; jj++)
          for (int ii = i; ii<i+`blck` && ii<`M`; ii++)
            `po`[ii+jj*`M`] = `pa`[jj+ii*`N`];
        __builtin_prefetch(&`pa`[(i+1)*`N`], 0, 1); // Prefetch read with low temporal locality
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

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
    # benchCacheOblivious(a, NbSamples)


## With OpenMP
## Note - OpenMP is faster when iterating on input row in inner loop
##        but in serial case its input col in inner loop that is faster
## Prefetch helps a lot in serial case but doesn't at all with OpenMP
## Maybe due to prefetching on the wrong CPU

########################################################################
# OpenMP

# Warmup: 1.1952 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 4000, N: 2000)
# Output shape: (M: 2000, N: 4000)
# Required number of operations:     8.000 millions
# Required bytes:                   32.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 250 samples in 6.231 seconds
# Average time: 24.342 ms
# Stddev  time: 1.587 ms
# Min     time: 21.526 ms
# Max     time: 34.755 ms
# Perf:         0.329 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose
# Collected 250 samples in 4.728 seconds
# Average time: 18.909 ms
# Stddev  time: 1.739 ms
# Min     time: 15.022 ms
# Max     time: 34.356 ms
# Perf:         0.423 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose - input row iteration
# Collected 250 samples in 5.440 seconds
# Average time: 21.759 ms
# Stddev  time: 1.470 ms
# Min     time: 17.792 ms
# Max     time: 33.235 ms
# Perf:         0.368 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP
# Collected 250 samples in 4.706 seconds
# Average time: 18.821 ms
# Stddev  time: 1.339 ms
# Min     time: 16.527 ms
# Max     time: 31.409 ms
# Perf:         0.425 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP - input row iteration
# Collected 250 samples in 5.939 seconds
# Average time: 23.754 ms
# Stddev  time: 1.973 ms
# Min     time: 19.449 ms
# Max     time: 40.228 ms
# Perf:         0.337 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking
# Collected 250 samples in 2.224 seconds
# Average time: 8.891 ms
# Stddev  time: 0.818 ms
# Min     time: 8.397 ms
# Max     time: 16.753 ms
# Perf:         0.900 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking - input row iteration
# Collected 250 samples in 3.051 seconds
# Average time: 12.203 ms
# Stddev  time: 1.734 ms
# Min     time: 10.663 ms
# Max     time: 24.437 ms
# Perf:         0.656 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling
# Collected 250 samples in 2.510 seconds
# Average time: 10.038 ms
# Stddev  time: 1.097 ms
# Min     time: 8.659 ms
# Max     time: 21.243 ms
# Perf:         0.797 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling - input row iteration
# Collected 250 samples in 1.903 seconds
# Average time: 7.612 ms
# Stddev  time: 0.845 ms
# Min     time: 7.056 ms
# Max     time: 14.593 ms
# Perf:         1.051 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking with Prefetch
# Collected 250 samples in 2.360 seconds
# Average time: 9.437 ms
# Stddev  time: 1.074 ms
# Min     time: 8.664 ms
# Max     time: 16.393 ms
# Perf:         0.848 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling + Prefetch - input row iteration
# Collected 250 samples in 1.990 seconds
# Average time: 7.960 ms
# Stddev  time: 0.720 ms
# Min     time: 7.399 ms
# Max     time: 14.150 ms
# Perf:         1.005 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

########################################################################
# Serial
########################################################################

# The cache blocking + prefetch seems to trigger constant and loop folding
# and does not seem to be replicable if not defined inline

# Warmup: 1.4231 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 4000, N: 2000)
# Output shape: (M: 2000, N: 4000)
# Required number of operations:     8.000 millions
# Required bytes:                   32.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 250 samples in 9.150 seconds
# Average time: 36.039 ms
# Stddev  time: 6.383 ms
# Min     time: 34.098 ms
# Max     time: 75.195 ms
# Perf:         0.222 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose
# Collected 250 samples in 7.037 seconds
# Average time: 28.147 ms
# Stddev  time: 1.580 ms
# Min     time: 26.689 ms
# Max     time: 33.603 ms
# Perf:         0.284 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Naive transpose - input row iteration
# Collected 250 samples in 8.782 seconds
# Average time: 35.129 ms
# Stddev  time: 1.069 ms
# Min     time: 34.082 ms
# Max     time: 44.597 ms
# Perf:         0.228 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP
# Collected 250 samples in 6.935 seconds
# Average time: 27.741 ms
# Stddev  time: 1.085 ms
# Min     time: 26.723 ms
# Max     time: 37.929 ms
# Perf:         0.288 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Collapsed OpenMP - input row iteration
# Collected 250 samples in 8.872 seconds
# Average time: 35.486 ms
# Stddev  time: 1.247 ms
# Min     time: 34.120 ms
# Max     time: 44.905 ms
# Perf:         0.225 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking
# Collected 250 samples in 3.121 seconds
# Average time: 12.485 ms
# Stddev  time: 0.961 ms
# Min     time: 12.018 ms
# Max     time: 24.380 ms
# Perf:         0.641 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking - input row iteration
# Collected 250 samples in 5.073 seconds
# Average time: 20.292 ms
# Stddev  time: 0.997 ms
# Min     time: 19.466 ms
# Max     time: 30.588 ms
# Perf:         0.394 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling
# Collected 250 samples in 3.339 seconds
# Average time: 13.355 ms
# Stddev  time: 0.748 ms
# Min     time: 12.172 ms
# Max     time: 19.928 ms
# Perf:         0.599 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling - input row iteration
# Collected 250 samples in 3.335 seconds
# Average time: 13.342 ms
# Stddev  time: 0.654 ms
# Min     time: 12.950 ms
# Max     time: 21.406 ms
# Perf:         0.600 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# Cache blocking with Prefetch
# Collected 250 samples in 2.541 seconds
# Average time: 10.164 ms
# Stddev  time: 0.792 ms
# Min     time: 9.741 ms
# Max     time: 18.762 ms
# Perf:         0.787 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

# 2D Tiling + Prefetch - input row iteration
# Collected 250 samples in 3.360 seconds
# Average time: 13.441 ms
# Stddev  time: 0.864 ms
# Min     time: 13.025 ms
# Max     time: 21.967 ms
# Perf:         0.595 GFLOP/s

# Display output[1] to make sure it's not optimized away
# 0.7808474898338318

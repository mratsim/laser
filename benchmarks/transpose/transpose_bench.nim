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
  ../../laser/dynamic_stack_arrays,
  ../../laser/compiler_optim_hints

const
  M     =  2000
  N     =  1000

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
  const blck = 32

  bench("Cache blocking"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for simd
    for (int i = 0; i < `N`; i+=`blck`)
      for (int j = 0; j < `M`; ++j)
        for (int b = 0; b < `blck` && i+b<`N`; ++b)
          `po`[i+j*`M` + b] = `pa`[j+(i+b)*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))

proc benchCacheBlockingExchanged(a: seq[float32], nb_samples: int) =
  var output = newSeq[float32](out_size)
  withCompilerOptimHints()

  let pa{.restrict.} = a[0].unsafeAddr
  let po{.restrict.} = output[0].addr
  const blck = 32

  bench("Cache blocking - input row iteration"):
    discard
  do:
    {.emit: """
    #pragma omp parallel for simd
    for (int j = 0; j < `M`; j+=`blck`)
      for (int i = 0; i < `N`; ++i)
        for (int b = 0; b < `blck` && j+b<`M`; ++b)
          `po`[j+i*`M` + b] = `pa`[i+(j+b)*`N`];
    """.}
    # echo a.toString((M, N))
    # echo output.toString((N, M))
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

    # benchBLAS(a, nb_samples = 1000)
    benchForEachStrided(a, nb_samples = 1000)
    benchNaive(a, nb_samples = 1000)
    benchNaiveExchanged(a, nb_samples = 1000)
    benchCollapsed(a, nb_samples = 1000)
    benchCollapsedExchanged(a, nb_samples = 1000)
    benchCacheBlocking(a, nb_samples = 1000)
    benchCacheBlockingExchanged(a, nb_samples = 1000)

# Warmup: 1.1959 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A matrix shape: (M: 2000, N: 1000)
# Output shape: (M: 1000, N: 2000)
# Required number of operations:     2.000 millions
# Required bytes:                    8.000 MB
# Arithmetic intensity:              0.250 FLOP/byte

# Laser ForEachStrided
# Collected 1000 samples in 5.399 seconds
# Average time: 5.272 ms
# Stddev  time: 0.479 ms
# Min     time: 4.353 ms
# Max     time: 13.638 ms
# Perf:         0.379 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 1.564622351679645e-07

# Naive transpose
# Collected 1000 samples in 3.237 seconds
# Average time: 3.235 ms
# Stddev  time: 0.662 ms
# Min     time: 2.781 ms
# Max     time: 10.299 ms
# Perf:         0.618 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 1.564622351679645e-07

# Collapsed OpenMP
# Collected 1000 samples in 3.251 seconds
# Average time: 3.249 ms
# Stddev  time: 0.409 ms
# Min     time: 2.929 ms
# Max     time: 11.542 ms
# Perf:         0.616 GFLOP/s

# Display output[0] to make sure it's not optimized away
# 1.564622351679645e-07

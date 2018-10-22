# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

# Latency: Number of cycles to wait before using a result in case of data dependency
# Throughput: Number of cycles an instructions takes.

# This benchmarks simple reduction operations with a varying amount of accumulators.
# FP base operations have a latency of 3~5 clock cycles before Skylake
# but a throughput of 1 per clock cycle before Haswell and 2 per clock cycle after Haswell.
#
# This means that if we accumulate a sum in the same result variable CPU is mostly waiting
# due to latency, while by using multiple accumulators we avoid the data dependency.
#
# The compiler should automatically do that for integers or
# with fast-math for floating points. Without FP addition will not be
# considered associative and it will not create temporary buffers.
#
# Note that many temp variables will increase register pressure and might lead to register spilling.


import
  ../../laser/strided_iteration/map_foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays]

withCompilerOptimHints()

proc newTensor*[T](shape: varargs[int]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  setZero(result, check_contiguous = false)

proc newTensor*[T](shape: Metadata): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  setZero(result, check_contiguous = false)

proc randomTensor*[T](shape: openarray[int], max: T): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  forEachContiguousSerial val in result:
    val = T(rand(max))

proc randomTensor*[T](shape: openarray[int], valrange: Slice[T]): Tensor[T] =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)
  forEachContiguousSerial val in result:
    val = T(rand(valrange))

func transpose*(t: Tensor): Tensor =
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage

func getIndex[T](t: Tensor[T], idx: varargs[int]): int =
  ## Convert [i, j, k, l ...] to the memory location referred by the index
  result = t.offset
  for i in 0 ..< t.shape.len:
    result += t.strides[i] * idx[i]

func `[]`*[T](t: Tensor[T], idx: varargs[int]): T {.inline.}=
  ## Index tensor
  t.storage.raw_data[t.getIndex(idx)]

################################################################


import random, times, stats, strformat

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = cpuTime()
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = cpuTime()
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats(name: string, accum: float32) {.dirty.} =
  echo "\n" & name & " - float64"
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f}ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f}ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f}ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f}ms"
  echo "\nDisplay output[[0,0]] to make sure it's not optimized away"
  echo accum # Prevents compiler from optimizing stuff away

template bench(name: string, accum: var float32, body: untyped) {.dirty.}=
  block: # Actual bench
    var stats: RunningStat
    let global_start = cpuTime()
    for _ in 0 ..< nb_samples:
      let start = cpuTime()
      body
      let stop = cpuTime()
      stats.push stop - start
    let global_stop = cpuTime()
    printStats(name, accum)

proc mainBench_1_accum_simple(a: Tensor, nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 1 accumulator - simple iter", accum):
    for i in 0 ..< a.size:
      accum += a.storage.raw_data[i]

proc mainBench_1_accum_macro(a: Tensor, nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 1 accumulator - macro iter", accum):
    forEachContiguousSerial val in a:
      accum += val

when defined(fastmath):
  {.passC:"-ffast-math".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  block: # All contiguous
    let
      a = randomTensor([10000, 1000], -1.0'f32 .. 1.0'f32)
    mainBench_1_accum_simple(a, 1000)
    mainBench_1_accum_macro(a, 1000)


## Without fastmath:
# Warmup: 1.1932 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 10.334 seconds
# Average time: 10.330ms
# Stddev  time: 0.237ms
# Min     time: 9.889ms
# Max     time: 15.416ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 10.448 seconds
# Average time: 10.445ms
# Stddev  time: 0.355ms
# Min     time: 9.868ms
# Max     time: 14.489ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

#########################

##  With fastmath

# Warmup: 1.1923 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 2.694 seconds
# Average time: 2.690ms
# Stddev  time: 0.276ms
# Min     time: 2.370ms
# Max     time: 7.708ms

# Display output[[0,0]] to make sure it's not optimized away
# -355854.53125

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 2.649 seconds
# Average time: 2.646ms
# Stddev  time: 0.325ms
# Min     time: 2.363ms
# Max     time: 7.964ms

# Display output[[0,0]] to make sure it's not optimized away
# -355854.53125

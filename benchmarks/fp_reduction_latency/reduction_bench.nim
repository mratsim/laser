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

func round_down_multiple(x: Natural, step: static Natural): int {.inline.} =
  x - x mod step

proc mainBench_1_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 1 accumulator - simple iter", accum):
    for i in 0 ..< a.size:
      accum += a.storage.raw_data[i]

proc mainBench_1_accum_macro(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 1 accumulator - macro iter", accum):
    forEachContiguousSerial val in a:
      accum += val

proc mainBench_2_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 2 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(2)
    var
      accum1 = 0'f32
    for i in countup(0, unroll_stop - 1, 2):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1

proc mainBench_3_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 3 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(3)
    var
      accum1 = 0'f32
      accum2 = 0'f32
    for i in countup(0, unroll_stop - 1, 3):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1 + accum2

proc mainBench_4_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 4 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(4)
    var
      accum1 = 0'f32
      accum2 = 0'f32
      accum3 = 0'f32
    for i in countup(0, unroll_stop - 1, 4):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
      accum3 += a.storage.raw_data[i+3]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1
    accum2 += accum3
    accum += accum2

proc mainBench_5_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 5 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(5)
    var
      accum1 = 0'f32
      accum2 = 0'f32
      accum3 = 0'f32
      accum4 = 0'f32
    for i in countup(0, unroll_stop - 1, 5):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
      accum3 += a.storage.raw_data[i+3]
      accum4 += a.storage.raw_data[i+4]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum2 += accum3 + accum4
    accum += accum1
    accum += accum2

proc mainBench_6_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 6 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(6)
    var
      accum1 = 0'f32
      accum2 = 0'f32
      accum3 = 0'f32
      accum4 = 0'f32
      accum5 = 0'f32
    for i in countup(0, unroll_stop - 1, 6):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
      accum3 += a.storage.raw_data[i+3]
      accum4 += a.storage.raw_data[i+4]
      accum5 += a.storage.raw_data[i+5]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1
    accum2 += accum3
    accum4 += accum5
    accum += accum2
    accum += accum4

proc mainBench_7_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 7 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(7)
    var
      accum1 = 0'f32
      accum2 = 0'f32
      accum3 = 0'f32
      accum4 = 0'f32
      accum5 = 0'f32
      accum6 = 0'f32
    for i in countup(0, unroll_stop - 1, 7):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
      accum3 += a.storage.raw_data[i+3]
      accum4 += a.storage.raw_data[i+4]
      accum5 += a.storage.raw_data[i+5]
      accum6 += a.storage.raw_data[i+6]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1
    accum2 += accum3
    accum4 += accum5 + accum6
    accum += accum2
    accum += accum4

proc mainBench_8_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 8 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(8)
    var
      accum1 = 0'f32
      accum2 = 0'f32
      accum3 = 0'f32
      accum4 = 0'f32
      accum5 = 0'f32
      accum6 = 0'f32
      accum7 = 0'f32
    for i in countup(0, unroll_stop - 1, 8):
      accum  += a.storage.raw_data[i]
      accum1 += a.storage.raw_data[i+1]
      accum2 += a.storage.raw_data[i+2]
      accum3 += a.storage.raw_data[i+3]
      accum4 += a.storage.raw_data[i+4]
      accum5 += a.storage.raw_data[i+5]
      accum6 += a.storage.raw_data[i+6]
      accum7 += a.storage.raw_data[i+7]
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1
    accum2 += accum3
    accum4 += accum5
    accum6 += accum7
    accum += accum2
    accum4 += accum6
    accum += accum4

when defined(fastmath):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  block: # All contiguous
    let
      a = randomTensor([10000, 1000], -1.0'f32 .. 1.0'f32)
    mainBench_1_accum_simple(a, 1000)
    mainBench_1_accum_macro(a, 1000)
    mainBench_2_accum_simple(a, 1000)
    mainBench_3_accum_simple(a, 1000)
    mainBench_4_accum_simple(a, 1000)
    mainBench_5_accum_simple(a, 1000)
    mainBench_6_accum_simple(a, 1000)
    mainBench_7_accum_simple(a, 1000)
    mainBench_8_accum_simple(a, 1000)


# Results on a i5-5257U mobile Broadwell 2.7GhZ Turbo 3.1
# post Haswell so 2 FP add per cycle

# # Normal

# Warmup: 1.1915 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 10.393 seconds
# Average time: 10.389ms
# Stddev  time: 0.233ms
# Min     time: 10.003ms
# Max     time: 15.001ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 10.292 seconds
# Average time: 10.288ms
# Stddev  time: 0.218ms
# Min     time: 10.033ms
# Max     time: 13.859ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

# Reduction - 2 accumulators - simple iter - float64
# Collected 1000 samples in 5.477 seconds
# Average time: 5.473ms
# Stddev  time: 0.282ms
# Min     time: 5.133ms
# Max     time: 9.514ms

# Display output[[0,0]] to make sure it's not optimized away
# -356332.21875

# Reduction - 3 accumulators - simple iter - float64
# Collected 1000 samples in 4.167 seconds
# Average time: 4.163ms
# Stddev  time: 0.429ms
# Min     time: 3.614ms
# Max     time: 10.036ms

# Display output[[0,0]] to make sure it's not optimized away
# -355871.09375

# Reduction - 4 accumulators - simple iter - float64
# Collected 1000 samples in 3.977 seconds
# Average time: 3.973ms
# Stddev  time: 0.357ms
# Min     time: 3.568ms
# Max     time: 7.726ms

# Display output[[0,0]] to make sure it's not optimized away
# -353982.25

# Reduction - 5 accumulators - simple iter - float64
# Collected 1000 samples in 2.880 seconds
# Average time: 2.876ms
# Stddev  time: 0.277ms
# Min     time: 2.575ms
# Max     time: 9.094ms

# Display output[[0,0]] to make sure it's not optimized away
# -354863.09375

# Reduction - 6 accumulators - simple iter - float64
# Collected 1000 samples in 3.132 seconds
# Average time: 3.128ms
# Stddev  time: 0.277ms
# Min     time: 2.823ms
# Max     time: 7.419ms

# Display output[[0,0]] to make sure it's not optimized away
# -360794.875

# Reduction - 7 accumulators - simple iter - float64
# Collected 1000 samples in 2.932 seconds
# Average time: 2.928ms
# Stddev  time: 0.247ms
# Min     time: 2.648ms
# Max     time: 7.745ms

# Display output[[0,0]] to make sure it's not optimized away
# -355866.96875

# Reduction - 8 accumulators - simple iter - float64
# Collected 1000 samples in 3.256 seconds
# Average time: 3.252ms
# Stddev  time: 0.364ms
# Min     time: 2.853ms
# Max     time: 10.200ms

# Display output[[0,0]] to make sure it's not optimized away
# -354784.0625


# #################################
# # Native

# Warmup: 1.1933 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 10.263 seconds
# Average time: 10.259ms
# Stddev  time: 0.316ms
# Min     time: 9.834ms
# Max     time: 15.145ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 10.284 seconds
# Average time: 10.281ms
# Stddev  time: 0.264ms
# Min     time: 9.832ms
# Max     time: 15.251ms

# Display output[[0,0]] to make sure it's not optimized away
# -353593.96875

# Reduction - 2 accumulators - simple iter - float64
# Collected 1000 samples in 5.340 seconds
# Average time: 5.336ms
# Stddev  time: 0.290ms
# Min     time: 4.984ms
# Max     time: 9.020ms

# Display output[[0,0]] to make sure it's not optimized away
# -356332.21875

# Reduction - 3 accumulators - simple iter - float64
# Collected 1000 samples in 3.862 seconds
# Average time: 3.858ms
# Stddev  time: 0.396ms
# Min     time: 3.530ms
# Max     time: 11.119ms

# Display output[[0,0]] to make sure it's not optimized away
# -355871.09375

# Reduction - 4 accumulators - simple iter - float64
# Collected 1000 samples in 4.094 seconds
# Average time: 4.090ms
# Stddev  time: 0.407ms
# Min     time: 3.724ms
# Max     time: 10.514ms

# Display output[[0,0]] to make sure it's not optimized away
# -353982.25

# Reduction - 5 accumulators - simple iter - float64
# Collected 1000 samples in 2.753 seconds
# Average time: 2.749ms
# Stddev  time: 0.303ms
# Min     time: 2.520ms
# Max     time: 7.043ms

# Display output[[0,0]] to make sure it's not optimized away
# -354863.09375

# Reduction - 6 accumulators - simple iter - float64
# Collected 1000 samples in 2.992 seconds
# Average time: 2.988ms
# Stddev  time: 0.300ms
# Min     time: 2.742ms
# Max     time: 7.467ms

# Display output[[0,0]] to make sure it's not optimized away
# -360794.875

# Reduction - 7 accumulators - simple iter - float64
# Collected 1000 samples in 2.852 seconds
# Average time: 2.848ms
# Stddev  time: 0.389ms
# Min     time: 2.605ms
# Max     time: 8.985ms

# Display output[[0,0]] to make sure it's not optimized away
# -355866.96875

# Reduction - 8 accumulators - simple iter - float64
# Collected 1000 samples in 3.216 seconds
# Average time: 3.212ms
# Stddev  time: 0.358ms
# Min     time: 2.929ms
# Max     time: 10.408ms

# Display output[[0,0]] to make sure it's not optimized away
# -354784.0625

# #############################################@
# # Fastmath

# Warmup: 1.1960 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 2.572 seconds
# Average time: 2.568ms
# Stddev  time: 0.205ms
# Min     time: 2.279ms
# Max     time: 4.146ms

# Display output[[0,0]] to make sure it's not optimized away
# -355854.53125

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 2.488 seconds
# Average time: 2.484ms
# Stddev  time: 0.359ms
# Min     time: 2.278ms
# Max     time: 7.938ms

# Display output[[0,0]] to make sure it's not optimized away
# -355854.53125

# Reduction - 2 accumulators - simple iter - float64
# Collected 1000 samples in 2.618 seconds
# Average time: 2.614ms
# Stddev  time: 0.315ms
# Min     time: 2.320ms
# Max     time: 9.847ms

# Display output[[0,0]] to make sure it's not optimized away
# -354428.1875

# Reduction - 3 accumulators - simple iter - float64
# Collected 1000 samples in 3.086 seconds
# Average time: 3.082ms
# Stddev  time: 0.366ms
# Min     time: 2.675ms
# Max     time: 9.336ms

# Display output[[0,0]] to make sure it's not optimized away
# -357209.96875

# Reduction - 4 accumulators - simple iter - float64
# Collected 1000 samples in 2.916 seconds
# Average time: 2.912ms
# Stddev  time: 0.355ms
# Min     time: 2.681ms
# Max     time: 8.978ms

# Display output[[0,0]] to make sure it's not optimized away
# -355789.15625

# Reduction - 5 accumulators - simple iter - float64
# Collected 1000 samples in 3.055 seconds
# Average time: 3.051ms
# Stddev  time: 0.333ms
# Min     time: 2.838ms
# Max     time: 9.121ms

# Display output[[0,0]] to make sure it's not optimized away
# -354452.90625

# Reduction - 6 accumulators - simple iter - float64
# Collected 1000 samples in 3.079 seconds
# Average time: 3.075ms
# Stddev  time: 0.338ms
# Min     time: 2.859ms
# Max     time: 9.758ms

# Display output[[0,0]] to make sure it's not optimized away
# -357209.9375

# Reduction - 7 accumulators - simple iter - float64
# Collected 1000 samples in 2.989 seconds
# Average time: 2.985ms
# Stddev  time: 0.348ms
# Min     time: 2.743ms
# Max     time: 8.959ms

# Display output[[0,0]] to make sure it's not optimized away
# -356960.78125

# Reduction - 8 accumulators - simple iter - float64
# Collected 1000 samples in 3.241 seconds
# Average time: 3.237ms
# Stddev  time: 0.356ms
# Min     time: 2.719ms
# Max     time: 8.559ms

# Display output[[0,0]] to make sure it's not optimized away
# -355789.15625

# #############################################
# # Fastmath + march=native

# Warmup: 1.1936 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float64
# Collected 1000 samples in 2.549 seconds
# Average time: 2.545ms
# Stddev  time: 0.147ms
# Min     time: 2.356ms
# Max     time: 3.010ms

# Display output[[0,0]] to make sure it's not optimized away
# -355800.28125

# Reduction - 1 accumulator - macro iter - float64
# Collected 1000 samples in 2.535 seconds
# Average time: 2.531ms
# Stddev  time: 0.271ms
# Min     time: 2.350ms
# Max     time: 8.269ms

# Display output[[0,0]] to make sure it's not optimized away
# -355800.28125

# Reduction - 2 accumulators - simple iter - float64
# Collected 1000 samples in 2.556 seconds
# Average time: 2.552ms
# Stddev  time: 0.271ms
# Min     time: 2.362ms
# Max     time: 7.172ms

# Display output[[0,0]] to make sure it's not optimized away
# -356442.21875

# Reduction - 3 accumulators - simple iter - float64
# Collected 1000 samples in 2.530 seconds
# Average time: 2.525ms
# Stddev  time: 0.267ms
# Min     time: 2.347ms
# Max     time: 6.759ms

# Display output[[0,0]] to make sure it's not optimized away
# -356974.125

# Reduction - 4 accumulators - simple iter - float64
# Collected 1000 samples in 3.101 seconds
# Average time: 3.096ms
# Stddev  time: 0.309ms
# Min     time: 2.867ms
# Max     time: 8.936ms

# Display output[[0,0]] to make sure it's not optimized away
# -356450.53125

# Reduction - 5 accumulators - simple iter - float64
# Collected 1000 samples in 2.861 seconds
# Average time: 2.856ms
# Stddev  time: 0.399ms
# Min     time: 2.613ms
# Max     time: 10.193ms

# Display output[[0,0]] to make sure it's not optimized away
# -354880.15625

# Reduction - 6 accumulators - simple iter - float64
# Collected 1000 samples in 3.084 seconds
# Average time: 3.080ms
# Stddev  time: 0.362ms
# Min     time: 2.837ms
# Max     time: 8.398ms

# Display output[[0,0]] to make sure it's not optimized away
# -356971.59375

# Reduction - 7 accumulators - simple iter - float64
# Collected 1000 samples in 3.706 seconds
# Average time: 3.702ms
# Stddev  time: 0.379ms
# Min     time: 3.449ms
# Max     time: 9.547ms

# Display output[[0,0]] to make sure it's not optimized away
# -356266.84375

# Reduction - 8 accumulators - simple iter - float64
# Collected 1000 samples in 4.258 seconds
# Average time: 4.253ms
# Stddev  time: 0.270ms
# Min     time: 4.039ms
# Max     time: 9.031ms

# Display output[[0,0]] to make sure it's not optimized away
# -356453.53125

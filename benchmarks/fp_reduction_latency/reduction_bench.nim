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
  ../../laser/strided_iteration/foreach,
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


import random, times, stats, strformat, math

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
  echo "\n" & name & " - float32"
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Theoretical perf: {a.size.float / (float(10^6) * stats.mean):>4.3f} MFLOP/s"
  echo "\nDisplay sum of samples sums to make sure it's not optimized away"
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

#############################################################
# Assembly generated in march=native

# 10.75 s   20.8%	10.75 s	 	     mainBench1_accum_simple_QKAy4s19aaqk31KNFq64WfA
# 10.69 s   20.7%	10.68 s	 	     mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2
# 5.72 s   11.0%	5.71 s	 	     mainBench2_accum_simple_QKAy4s19aaqk31KNFq64WfA_3
# 4.71 s    9.1%	4.71 s	 	     mainBench4_accum_simple_QKAy4s19aaqk31KNFq64WfA_5
# 4.46 s    8.6%	4.46 s	 	     mainBench3_accum_simple_QKAy4s19aaqk31KNFq64WfA_4
# 3.76 s    7.3%	3.76 s	 	     mainBench8_accum_simple_QKAy4s19aaqk31KNFq64WfA_9
# 3.52 s    6.8%	3.52 s	 	     mainBench6_accum_simple_QKAy4s19aaqk31KNFq64WfA_7
# 3.33 s    6.4%	3.33 s	 	     mainBench7_accum_simple_QKAy4s19aaqk31KNFq64WfA_8
# 3.32 s    6.4%	3.32 s	 	     mainBench5_accum_simple_QKAy4s19aaqk31KNFq64WfA_6

## mainBench_1_accum_simple
# +0xd5	    nopw                %cs:(%rax,%rax)
# +0xe0	        vaddss              (%rdx,%rsi,4), %xmm0, %xmm0
# +0xe5	        vaddss              4(%rdx,%rsi,4), %xmm0, %xmm0
# +0xeb	        vaddss              8(%rdx,%rsi,4), %xmm0, %xmm0
# +0xf1	        vaddss              12(%rdx,%rsi,4), %xmm0, %xmm0
# +0xf7	        vaddss              16(%rdx,%rsi,4), %xmm0, %xmm0
# +0xfd	        vaddss              20(%rdx,%rsi,4), %xmm0, %xmm0
# +0x103	      vaddss              24(%rdx,%rsi,4), %xmm0, %xmm0
# +0x109	      vaddss              28(%rdx,%rsi,4), %xmm0, %xmm0
# +0x10f	      addq                $8, %rsi
# +0x113	      cmpq                %rsi, %rax
# +0x116	      jne                 "mainBench1_accum_simple_QKAy4s19aaqk31KNFq64WfA+0xe0"

## mainBench_1_accum_macro
# +0x12e	    xorl                %edx, %edx
# +0x130	        vaddss              -28(%rsi,%rdx,4), %xmm0, %xmm0
# +0x136	        vaddss              -24(%rsi,%rdx,4), %xmm0, %xmm0
# +0x13c	        vaddss              -20(%rsi,%rdx,4), %xmm0, %xmm0
# +0x142	        vaddss              -16(%rsi,%rdx,4), %xmm0, %xmm0
# +0x148	        vaddss              -12(%rsi,%rdx,4), %xmm0, %xmm0
# +0x14e	        vaddss              -8(%rsi,%rdx,4), %xmm0, %xmm0
# +0x154	        vaddss              -4(%rsi,%rdx,4), %xmm0, %xmm0
# +0x15a	        vaddss              (%rsi,%rdx,4), %xmm0, %xmm0
# +0x15f	        addq                $8, %rdx
# +0x163	        cmpq                %rdx, %rax
# +0x166	        jne                 "mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2+0x130"

## mainBench_2_accum_simple
# +0x119	    nopl                (%rax)
# +0x120	        vmovsd              (%rdi,%rdx,4), %xmm0
# +0x125	        vaddps              %xmm0, %xmm1, %xmm0
# +0x129	        vmovsd              8(%rdi,%rdx,4), %xmm1
# +0x12f	        vaddps              %xmm1, %xmm0, %xmm0
# +0x133	        vmovsd              16(%rdi,%rdx,4), %xmm1
# +0x139	        vaddps              %xmm1, %xmm0, %xmm0
# +0x13d	        vmovsd              24(%rdi,%rdx,4), %xmm1
# +0x143	        vaddps              %xmm1, %xmm0, %xmm0
# +0x147	        vmovsd              32(%rdi,%rdx,4), %xmm1
# +0x14d	        vaddps              %xmm1, %xmm0, %xmm0
# +0x151	        vmovsd              40(%rdi,%rdx,4), %xmm1
# +0x157	        vaddps              %xmm1, %xmm0, %xmm0
# +0x15b	        vmovsd              48(%rdi,%rdx,4), %xmm1
# +0x161	        vaddps              %xmm1, %xmm0, %xmm0
# +0x165	        vmovsd              56(%rdi,%rdx,4), %xmm1
# +0x16b	        vaddps              %xmm1, %xmm0, %xmm1
# +0x16f	        addq                $16, %rdx
# +0x173	        addq                $8, %rbx
# +0x177	        jne                 "mainBench2_accum_simple_QKAy4s19aaqk31KNFq64WfA_3+0x120"

## mainBench_3_accum_simple
# +0xbe	    nop
# +0xc0	        vaddss              (%rsi,%rdi,4), %xmm2, %xmm2
# +0xc5	        vaddss              4(%rsi,%rdi,4), %xmm1, %xmm1
# +0xcb	        vaddss              8(%rsi,%rdi,4), %xmm0, %xmm0
# +0xd1	        addq                $3, %rdi
# +0xd5	        cmpq                %rax, %rdi
# +0xd8	        jl                  "mainBench3_accum_simple_QKAy4s19aaqk31KNFq64WfA_4+0xc0"

## mainBench_4_accum_simple
# +0x125	    nopw                %cs:(%rax,%rax)
# +0x130	        vaddss              (%rsi,%rdi,4), %xmm3, %xmm3
# +0x135	        vaddss              4(%rsi,%rdi,4), %xmm2, %xmm2
# +0x13b	        vaddss              8(%rsi,%rdi,4), %xmm1, %xmm1
# +0x141	        vaddss              12(%rsi,%rdi,4), %xmm0, %xmm0
# +0x147	        vaddss              16(%rsi,%rdi,4), %xmm3, %xmm3
# +0x14d	        vaddss              20(%rsi,%rdi,4), %xmm2, %xmm2
# +0x153	        vaddss              24(%rsi,%rdi,4), %xmm1, %xmm1
# +0x159	        vaddss              28(%rsi,%rdi,4), %xmm0, %xmm0
# +0x15f	        vaddss              32(%rsi,%rdi,4), %xmm3, %xmm3
# +0x165	        vaddss              36(%rsi,%rdi,4), %xmm2, %xmm2
# +0x16b	        vaddss              40(%rsi,%rdi,4), %xmm1, %xmm1
# +0x171	        vaddss              44(%rsi,%rdi,4), %xmm0, %xmm0
# +0x177	        vaddss              48(%rsi,%rdi,4), %xmm3, %xmm3
# +0x17d	        vaddss              52(%rsi,%rdi,4), %xmm2, %xmm2
# +0x183	        vaddss              56(%rsi,%rdi,4), %xmm1, %xmm1
# +0x189	        vaddss              60(%rsi,%rdi,4), %xmm0, %xmm0
# +0x18f	        addq                $16, %rdi
# +0x193	        addq                $4, %rbx
# +0x197	        jne                 "mainBench4_accum_simple_QKAy4s19aaqk31KNFq64WfA_5+0x130"

## mainBench_5_accum_simple (fastest)
# +0xbd	    nopl                (%rax)
# +0xc0	        vaddss              (%rsi,%rdi,4), %xmm3, %xmm3
# +0xc5	        vaddps              4(%rsi,%rdi,4), %xmm0, %xmm0
# +0xcb	        addq                $5, %rdi
# +0xcf	        cmpq                %rax, %rdi
# +0xd2	        jl                  "mainBench5_accum_simple_QKAy4s19aaqk31KNFq64WfA_6+0xc0"

## mainBench_6_accum_simple (3rd fastest)
# +0xc1	    nopw                %cs:(%rax,%rax)
# +0xd0	        vaddss              (%rsi,%rdi,4), %xmm2, %xmm2
# +0xd5	        vaddps              4(%rsi,%rdi,4), %xmm0, %xmm0
# +0xdb	        vaddss              20(%rsi,%rdi,4), %xmm1, %xmm1
# +0xe1	        addq                $6, %rdi
# +0xe5	        cmpq                %rax, %rdi
# +0xe8	        jl                  "mainBench6_accum_simple_QKAy4s19aaqk31KNFq64WfA_7+0xd0"

## mainBench_7_accum_simple (2nd fastest)
# +0xc8	    nopl                (%rax,%rax)
# +0xd0	        vaddss              (%rsi,%rdi,4), %xmm3, %xmm3
# +0xd5	        vaddps              4(%rsi,%rdi,4), %xmm0, %xmm0
# +0xdb	        vmovsd              20(%rsi,%rdi,4), %xmm2
# +0xe1	        vaddps              %xmm2, %xmm1, %xmm1
# +0xe5	        addq                $7, %rdi
# +0xe9	        cmpq                %rax, %rdi
# +0xec	        jl                  "mainBench7_accum_simple_QKAy4s19aaqk31KNFq64WfA_8+0xd0"

## mainBench_8_accum_simple (4th fastest)
# +0xdb	    nopl                (%rax,%rax)
# +0xe0	        vaddss              (%rsi,%rdi,4), %xmm4, %xmm4
# +0xe5	        vaddps              4(%rsi,%rdi,4), %xmm0, %xmm0
# +0xeb	        vaddss              20(%rsi,%rdi,4), %xmm3, %xmm3
# +0xf1	        vaddss              24(%rsi,%rdi,4), %xmm2, %xmm2
# +0xf7	        vaddss              28(%rsi,%rdi,4), %xmm1, %xmm1
# +0xfd	        vaddss              32(%rsi,%rdi,4), %xmm4, %xmm4
# +0x103	      vaddps              36(%rsi,%rdi,4), %xmm0, %xmm0
# +0x109	      vaddss              52(%rsi,%rdi,4), %xmm3, %xmm3
# +0x10f	      vaddss              56(%rsi,%rdi,4), %xmm2, %xmm2
# +0x115	      vaddss              60(%rsi,%rdi,4), %xmm1, %xmm1
# +0x11b	      addq                $16, %rdi
# +0x11f	      addq                $2, %rdx
# +0x123	      jne                 "mainBench8_accum_simple_QKAy4s19aaqk31KNFq64WfA_9+0xe0"

#############################################################
# Assembly generated in fastmath

# 3.78 s   11.6%	3.77 s	 	     mainBench8_accum_simple_QKAy4s19aaqk31KNFq64WfA_9
# 3.68 s   11.3%	3.68 s	 	     mainBench3_accum_simple_QKAy4s19aaqk31KNFq64WfA_4
# 3.63 s   11.2%	3.62 s	 	     mainBench5_accum_simple_QKAy4s19aaqk31KNFq64WfA_6
# 3.63 s   11.2%	3.62 s	 	     mainBench6_accum_simple_QKAy4s19aaqk31KNFq64WfA_7
# 3.54 s   10.9%	3.53 s	 	     mainBench7_accum_simple_QKAy4s19aaqk31KNFq64WfA_8
# 3.46 s   10.6%	3.45 s	 	     mainBench4_accum_simple_QKAy4s19aaqk31KNFq64WfA_5
# 3.19 s    9.8%	3.18 s	 	     mainBench2_accum_simple_QKAy4s19aaqk31KNFq64WfA_3
# 3.14 s    9.6%	3.13 s	 	     mainBench1_accum_simple_QKAy4s19aaqk31KNFq64WfA
# 3.05 s    9.4%	3.05 s	 	     mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2

## mainBench_5_accum_simple
# +0x171	    nopw                %cs:(%rax,%rax)
# +0x180	        movaps              %xmm14, -368(%rbp)
# +0x188	        movaps              %xmm6, -384(%rbp)
# +0x18f	        movaps              %xmm4, -400(%rbp)
# +0x196	        movaps              %xmm15, -416(%rbp)
# +0x19e	        movaps              %xmm12, -432(%rbp)
# +0x1a6	        movaps              %xmm0, -448(%rbp)
# +0x1ad	        movupd              64(%rbx), %xmm10
# +0x1b3	        movapd              %xmm10, -352(%rbp)
# +0x1bc	        movupd              (%rbx), %xmm12
# +0x1c1	        movups              16(%rbx), %xmm6
# +0x1c5	        movupd              32(%rbx), %xmm13
# +0x1cb	        movups              48(%rbx), %xmm7
# +0x1cf	        movaps              %xmm7, -288(%rbp)
# +0x1d6	        movupd              144(%rbx), %xmm11
# +0x1df	        movapd              %xmm11, -272(%rbp)
# +0x1e8	        movups              96(%rbx), %xmm14
# +0x1ed	        movupd              80(%rbx), %xmm4
# +0x1f2	        movups              128(%rbx), %xmm1
# +0x1f9	        movupd              112(%rbx), %xmm15
# +0x1ff	        movapd              %xmm12, %xmm3
# +0x204	        movapd              %xmm12, %xmm0
# +0x209	        movapd              %xmm13, %xmm8
# +0x20e	        blendpd             $2, %xmm12, %xmm8
# +0x215	        blendps             $2, %xmm6, %xmm12
# +0x21c	        blendpd             $2, %xmm13, %xmm10
# +0x223	        movapd              %xmm10, -320(%rbp)
# +0x22c	        blendpd             $2, %xmm6, %xmm3
# +0x232	        movapd              %xmm3, -304(%rbp)
# +0x23a	        blendps             $8, %xmm6, %xmm0
# +0x240	        movaps              %xmm0, -336(%rbp)
# +0x247	        movaps              %xmm6, %xmm10
# +0x24b	        blendps             $2, %xmm13, %xmm10
# +0x252	        blendps             $8, %xmm7, %xmm13
# +0x259	        blendpd             $2, %xmm13, %xmm12
# +0x260	        movaps              -112(%rbp), %xmm0
# +0x264	        addps               %xmm12, %xmm0
# +0x268	        movaps              %xmm0, -112(%rbp)
# +0x26c	        movapd              %xmm4, %xmm3
# +0x270	        movapd              %xmm4, %xmm12
# +0x275	        movapd              %xmm15, %xmm13
# +0x27a	        blendpd             $2, %xmm4, %xmm13
# +0x281	        movapd              %xmm4, %xmm9
# +0x286	        blendps             $2, %xmm14, %xmm9
# +0x28d	        movapd              %xmm11, %xmm6
# +0x292	        blendpd             $2, %xmm15, %xmm6
# +0x299	        blendpd             $2, %xmm14, %xmm3
# +0x2a0	        blendps             $8, %xmm14, %xmm12
# +0x2a7	        blendps             $2, %xmm15, %xmm14
# +0x2ae	        movaps              %xmm14, %xmm11
# +0x2b2	        movaps              %xmm15, %xmm7
# +0x2b6	        blendps             $8, %xmm1, %xmm7
# +0x2bc	        blendpd             $2, %xmm7, %xmm9
# +0x2c3	        movaps              -416(%rbp), %xmm15
# +0x2cb	        movaps              -176(%rbp), %xmm4
# +0x2d2	        addps               %xmm9, %xmm4
# +0x2d6	        movaps              %xmm4, -176(%rbp)
# +0x2dd	        movaps              -368(%rbp), %xmm14
# +0x2e5	        movaps              -400(%rbp), %xmm4
# +0x2ec	        movaps              -304(%rbp), %xmm7
# +0x2f3	        shufps              $57, -320(%rbp), %xmm7
# +0x2fb	        movaps              -432(%rbp), %xmm9
# +0x303	        addps               %xmm7, %xmm5
# +0x306	        shufps              $57, %xmm6, %xmm3
# +0x30a	        addps               %xmm3, %xmm2
# +0x30d	        movaps              -288(%rbp), %xmm6
# +0x314	        movaps              %xmm6, %xmm3
# +0x317	        movaps              -352(%rbp), %xmm0
# +0x31e	        blendps             $2, %xmm0, %xmm3
# +0x324	        movapd              -336(%rbp), %xmm7
# +0x32c	        shufpd              $1, %xmm3, %xmm7
# +0x331	        addps               %xmm7, %xmm14
# +0x335	        movaps              %xmm1, %xmm3
# +0x338	        movaps              -272(%rbp), %xmm7
# +0x33f	        blendps             $2, %xmm7, %xmm3
# +0x345	        shufpd              $1, %xmm3, %xmm12
# +0x34b	        addps               %xmm12, %xmm4
# +0x34f	        movaps              %xmm6, %xmm3
# +0x352	        blendpd             $2, %xmm0, %xmm3
# +0x358	        shufps              $147, %xmm3, %xmm8
# +0x35d	        addps               %xmm8, %xmm15
# +0x361	        movaps              %xmm1, %xmm3
# +0x364	        blendpd             $2, %xmm7, %xmm3
# +0x36a	        shufps              $147, %xmm3, %xmm13
# +0x36f	        movaps              -384(%rbp), %xmm3
# +0x376	        addps               %xmm13, %xmm9
# +0x37a	        movaps              %xmm9, %xmm12
# +0x37e	        blendps             $8, %xmm0, %xmm6
# +0x384	        blendpd             $2, %xmm6, %xmm10
# +0x38b	        addps               %xmm10, %xmm3
# +0x38f	        movaps              %xmm3, %xmm6
# +0x392	        blendps             $8, %xmm7, %xmm1
# +0x398	        blendpd             $2, %xmm1, %xmm11
# +0x39f	        movaps              -448(%rbp), %xmm0
# +0x3a6	        addps               %xmm11, %xmm0
# +0x3aa	        addq                $160, %rbx
# +0x3b1	        addq                $-8, %rdi
# +0x3b5	        jne                 "mainBench5_accum_simple_QKAy4s19aaqk31KNFq64WfA_6+0x180"


## mainBench_1_accum_macro (fastest)
# +0x1c8	    nopl                (%rax,%rax)
# +0x1d0	        movups              -112(%rdi,%rsi,4), %xmm1
# +0x1d5	        addps               %xmm2, %xmm1
# +0x1d8	        movups              -96(%rdi,%rsi,4), %xmm2
# +0x1dd	        addps               %xmm0, %xmm2
# +0x1e0	        movups              -80(%rdi,%rsi,4), %xmm0
# +0x1e5	        movups              -64(%rdi,%rsi,4), %xmm3
# +0x1ea	        movups              -48(%rdi,%rsi,4), %xmm4
# +0x1ef	        addps               %xmm0, %xmm4
# +0x1f2	        addps               %xmm1, %xmm4
# +0x1f5	        movups              -32(%rdi,%rsi,4), %xmm1
# +0x1fa	        addps               %xmm3, %xmm1
# +0x1fd	        addps               %xmm2, %xmm1
# +0x200	        movups              -16(%rdi,%rsi,4), %xmm2
# +0x205	        addps               %xmm4, %xmm2
# +0x208	        movups              (%rdi,%rsi,4), %xmm0
# +0x20c	        addps               %xmm1, %xmm0
# +0x20f	        addq                $32, %rsi
# +0x213	        addq                $4, %rcx
# +0x217	        jne                 "mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2+0x1d0"

#############################################################
# Assembly generated in fastmath + march=native

# 3.78 s   11.6%	3.77 s	 	     mainBench8_accum_simple_QKAy4s19aaqk31KNFq64WfA_9
# 3.68 s   11.3%	3.68 s	 	     mainBench3_accum_simple_QKAy4s19aaqk31KNFq64WfA_4
# 3.63 s   11.2%	3.62 s	 	     mainBench5_accum_simple_QKAy4s19aaqk31KNFq64WfA_6
# 3.63 s   11.2%	3.62 s	 	     mainBench6_accum_simple_QKAy4s19aaqk31KNFq64WfA_7
# 3.54 s   10.9%	3.53 s	 	     mainBench7_accum_simple_QKAy4s19aaqk31KNFq64WfA_8
# 3.46 s   10.6%	3.45 s	 	     mainBench4_accum_simple_QKAy4s19aaqk31KNFq64WfA_5
# 3.19 s    9.8%	3.18 s	 	     mainBench2_accum_simple_QKAy4s19aaqk31KNFq64WfA_3
# 3.14 s    9.6%	3.13 s	 	     mainBench1_accum_simple_QKAy4s19aaqk31KNFq64WfA
# 3.05 s    9.4%	3.05 s	 	     mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2

## mainBench_1_accum_macro

# +0x177	    nopw                (%rax,%rax)
# +0x180	        vaddps              -480(%rdi,%rsi,4), %ymm0, %ymm0
# +0x189	        vaddps              -448(%rdi,%rsi,4), %ymm1, %ymm1
# +0x192	        vaddps              -416(%rdi,%rsi,4), %ymm2, %ymm2
# +0x19b	        vaddps              -384(%rdi,%rsi,4), %ymm3, %ymm3
# +0x1a4	        vaddps              -352(%rdi,%rsi,4), %ymm0, %ymm0
# +0x1ad	        vaddps              -320(%rdi,%rsi,4), %ymm1, %ymm1
# +0x1b6	        vaddps              -288(%rdi,%rsi,4), %ymm2, %ymm2
# +0x1bf	        vaddps              -256(%rdi,%rsi,4), %ymm3, %ymm3
# +0x1c8	        vaddps              -224(%rdi,%rsi,4), %ymm0, %ymm0
# +0x1d1	        vaddps              -192(%rdi,%rsi,4), %ymm1, %ymm1
# +0x1da	        vaddps              -160(%rdi,%rsi,4), %ymm2, %ymm2
# +0x1e3	        vaddps              -128(%rdi,%rsi,4), %ymm3, %ymm3
# +0x1e9	        vaddps              -96(%rdi,%rsi,4), %ymm0, %ymm0
# +0x1ef	        vaddps              -64(%rdi,%rsi,4), %ymm1, %ymm1
# +0x1f5	        vaddps              -32(%rdi,%rsi,4), %ymm2, %ymm2
# +0x1fb	        vaddps              (%rdi,%rsi,4), %ymm3, %ymm3
# +0x200	        subq                $-128, %rsi
# +0x204	        addq                $4, %rbx
# +0x208	        jne                 "mainBench1_accum_macro_QKAy4s19aaqk31KNFq64WfA_2+0x180"


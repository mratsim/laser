# Apache v2.0 License
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
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays],
  ../../laser/simd

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
    accum = 0

func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

func sum_ps_sse3(vec: m128): float32 {.inline.} =
  let shuf = mm_movehdup_ps(vec)
  let sums = mm_add_ps(vec, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  result = mm_add_ss(sums, shuf2).mm_cvtss_f32

when defined(fastmath):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}

proc mainBench_sse_accum_1(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 4 elem by 1 accumulator - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(4)
    var
      accum_sse = mm_setzero_ps()
    for i in countup(0, unroll_stop - 1, 4):
      let data  = mm_load_ps(a.storage.raw_data[i].unsafeaddr)
      accum += data.sum_ps_sse3
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]

proc mainBench_sse_accum_2(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 4 elem by 2 accumulators - simple iter", accum):
    var accum1 = 0'f32
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    for i in countup(0, unroll_stop - 1, 8):
      accum  += a.storage.raw_data[i  ].unsafeaddr.mm_load_ps.sum_ps_sse3()
      accum1 += a.storage.raw_data[i+4].unsafeaddr.mm_load_ps.sum_ps_sse3()
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1

proc mainBench_sse_accum_2_interleaved(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - 4 elem by 2 accumulators - interleaved", accum):
    var accum1 = 0'f32
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    for i in countup(0, unroll_stop - 1, 8):
      let data0 = a.storage.raw_data[i  ].unsafeaddr.mm_load_ps()
      let data1 = a.storage.raw_data[i+4].unsafeaddr.mm_load_ps()
      let shuf0 = mm_movehdup_ps(data0)
      let shuf1 = mm_movehdup_ps(data1)
      let sums0 = mm_add_ps(data0, shuf0)
      let sums1 = mm_add_ps(data1, shuf1)
      let mvhi0 = mm_movehl_ps(sums0, sums0)
      let mvhi1 = mm_movehl_ps(sums1, sums1)
      accum  += mm_add_ss(sums0, mvhi0).mm_cvtss_f32()
      accum1 += mm_add_ss(sums1, mvhi1).mm_cvtss_f32()
    for i in unroll_stop ..< size:
      accum += a.storage.raw_data[i]
    accum += accum1

when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  block: # All contiguous
    let
      a = randomTensor([10000, 1000], -1.0'f32 .. 1.0'f32)
    mainBench_sse_accum_1(a, 1000)
    mainBench_sse_accum_2(a, 1000)
    mainBench_sse_accum_2_interleaved(a, 1000)

# Warmup: 1.1924 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 4 elem by 1 accumulator - simple iter - float64
# Collected 1000 samples in 3.714 seconds
# Average time: 3.710ms
# Stddev  time: 0.403ms
# Min     time: 3.125ms
# Max     time: 12.023ms

# Display output[[0,0]] to make sure it's not optimized away
# -363180.8125

# Reduction - 4 elem by 2 accumulators - simple iter - float64
# Collected 1000 samples in 3.640 seconds
# Average time: 3.636ms
# Stddev  time: 0.378ms
# Min     time: 3.125ms
# Max     time: 9.111ms

# Display output[[0,0]] to make sure it's not optimized away
# -363361.5625

# Reduction - 4 elem by 2 accumulators - interleaved - float64
# Collected 1000 samples in 3.645 seconds
# Average time: 3.641ms
# Stddev  time: 0.387ms
# Min     time: 3.123ms
# Max     time: 10.324ms

# Display output[[0,0]] to make sure it's not optimized away
# -363361.5625

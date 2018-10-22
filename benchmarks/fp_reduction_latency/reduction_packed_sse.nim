# Apache v2.0 License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  ../../laser/strided_iteration/map_foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays],
  ../../laser/simd,
  ../../laser/hpc_kernels/sum_sse3

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
  # FLOPS: for sum, we have one add per element
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

func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

func sum_ps_sse3(vec: m128): float32 =
  let shuf = mm_movehdup_ps(vec)
  let sums = mm_add_ps(vec, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  result = mm_add_ss(sums, shuf2).mm_cvtss_f32

func sum_ps_avx(vec: m256): float32 =
  let lo = mm256_castps256_ps128(vec)
  let hi = mm256_extractf128_ps(vec, 1)
  result = mm_add_ps(lo, hi).sum_ps_sse3()

proc mainBench_4_packed_sse_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 4 accumulators SSE", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(4)
    var accums: m128
    var ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 4):
      let data4 = a.storage.raw_data[i].unsafeaddr.mm_load_ps() # Can't use ptr_data, no address :/
      accums = mm_add_ps(accums, data4)
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    accum += accums.sum_ps_sse3()

proc mainBench_8_packed_sse_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 8 accumulators SSE", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    var accums0, accums1: m128
    var ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 8):
      # Can't use ptr_data, no address :/
      let
        data4_0 = a.storage.raw_data[i  ].unsafeaddr.mm_load_ps()
        data4_1 = a.storage.raw_data[i+4].unsafeaddr.mm_load_ps()
      accums0 = mm_add_ps(accums0, data4_0)
      accums1 = mm_add_ps(accums1, data4_1)
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    let accum0 = accums0.sum_ps_sse3()
    let accum1 = accums1.sum_ps_sse3()
    accum += accum0
    accum += accum1

proc mainBench_12_packed_sse_prod(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 12 accumulators SSE - prod impl", accum):
    accum += sum_sse3(a.storage.raw_data, a.size)

proc mainBench_8_packed_avx_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 8 accumulators AVX", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    var accums0: m256
    var ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 8):
      # Can't use ptr_data, no address :/
      let data8_0 = a.storage.raw_data[i  ].unsafeaddr.mm256_load_ps()
      accums0 = mm256_add_ps(accums0, data8_0)
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    accum += accums0.sum_ps_avx()

proc mainBench_16_packed_avx_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 16 accumulators AVX", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(16)
    var accums0, accums1: m256
    var ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 16):
      # Can't use ptr_data, no address :/
      let data8_0 = a.storage.raw_data[i  ].unsafeaddr.mm256_load_ps()
      let data8_1 = a.storage.raw_data[i+8].unsafeaddr.mm256_load_ps()
      accums0 = mm256_add_ps(accums0, data8_0)
      accums1 = mm256_add_ps(accums1, data8_1)
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    accum += accums0.sum_ps_avx() + accums1.sum_ps_avx()

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
    mainBench_4_packed_sse_accums(a, 1000)
    mainBench_8_packed_sse_accums(a, 1000)
    mainBench_12_packed_sse_prod(a, 1000)

    {.passC: "-mavx".}
    mainBench_8_packed_avx_accums(a, 1000)
    mainBench_16_packed_avx_accums(a, 1000)

## Bench on i5 Broadwell

# Warmup: 1.1929 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - packed 4 accumulators SSE - float64
# Collected 1000 samples in 2.934 seconds
# Average time: 2.930ms
# Stddev  time: 0.216ms
# Min     time: 2.652ms
# Max     time: 4.563ms

# Display output[[0,0]] to make sure it's not optimized away
# -356696.15625

# Reduction - packed 8 accumulators SSE - float64
# Collected 1000 samples in 2.667 seconds
# Average time: 2.664ms
# Stddev  time: 0.332ms
# Min     time: 2.415ms
# Max     time: 10.863ms

# Display output[[0,0]] to make sure it's not optimized away
# -356923.1875

# Reduction - packed 8 accumulators AVX - float64
# Collected 1000 samples in 2.736 seconds
# Average time: 2.732ms
# Stddev  time: 0.297ms
# Min     time: 2.497ms
# Max     time: 9.951ms

# Display output[[0,0]] to make sure it's not optimized away
# -356915.03125

# Reduction - packed 16 accumulators AVX - float64
# Collected 1000 samples in 2.780 seconds
# Average time: 2.776ms
# Stddev  time: 0.323ms
# Min     time: 2.496ms
# Max     time: 7.415ms

# Display output[[0,0]] to make sure it's not optimized away
# -356914.875

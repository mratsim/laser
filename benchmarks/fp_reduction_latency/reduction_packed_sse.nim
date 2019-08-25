# Apache v2.0 License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  ../../laser/strided_iteration/foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays],
  ../../laser/simd,
  ../../laser/primitives/reductions

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
  t.storage.raw_buffer[t.getIndex(idx)]

################################################################


import random, times, stats, strformat, math

proc warmup() =
  # Warmup - make sure cpu is on max perf
  let start = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
  var foo = 123
  for i in 0 ..< 300_000_000:
    foo += i*i mod 456
    foo = foo mod 789

  # Compiler shouldn't optimize away the results as cpuTime rely on sideeffects
  let stop = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
  echo &"Warmup: {stop - start:>4.4f} s, result {foo} (displayed to avoid compiler optimizing warmup away)"

template printStats(name: string, accum: float32) {.dirty.} =
  echo "\n" & name & " - float32"
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  # FLOPS: for sum, we have one add per element
  echo &"Perf:         {a.size.float / (float(10^9) * stats.mean):>4.3f} GFLOP/s"
  echo "\nDisplay sum of samples sums to make sure it's not optimized away"
  echo accum # Prevents compiler from optimizing stuff away

template bench(name: string, accum: var float32, body: untyped) {.dirty.}=
  block: # Actual bench
    var stats: RunningStat
    let global_start = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
    for _ in 0 ..< nb_samples:
      let start = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
      body
      let stop = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
      stats.push stop - start
    let global_stop = epochTime() # cpuTime() - cannot use cpuTime for multithreaded
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
      let data4 = a.storage.raw_buffer[i].unsafeaddr.mm_load_ps() # Can't use ptr_data, no address :/
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
        data4_0 = a.storage.raw_buffer[i  ].unsafeaddr.mm_load_ps()
        data4_1 = a.storage.raw_buffer[i+4].unsafeaddr.mm_load_ps()
      accums0 = mm_add_ps(accums0, data4_0)
      accums1 = mm_add_ps(accums1, data4_1)
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    let accum0 = accums0.sum_ps_sse3()
    let accum1 = accums1.sum_ps_sse3()
    accum += accum0
    accum += accum1

proc mainBench_packed_sse_prod(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - prod impl", accum):
    accum += reduce_sum(a.storage.raw_buffer, a.size)

proc mainBench_8_packed_avx_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 8 accumulators AVX", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    var accums0: m256
    var ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 8):
      # Can't use ptr_data, no address :/
      let data8_0 = a.storage.raw_buffer[i  ].unsafeaddr.mm256_load_ps()
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
      let data8_0 = a.storage.raw_buffer[i  ].unsafeaddr.mm256_load_ps()
      let data8_1 = a.storage.raw_buffer[i+8].unsafeaddr.mm256_load_ps()
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
    mainBench_packed_sse_prod(a, 1000)

    {.passC: "-mavx".}
    mainBench_8_packed_avx_accums(a, 1000)
    mainBench_16_packed_avx_accums(a, 1000)

## Bench on i5 Broadwell - serial implementation

# Warmup: 1.1946 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - packed 4 accumulators SSE - float32
# Collected 1000 samples in 2.841 seconds
# Average time: 2.837 ms
# Stddev  time: 0.251 ms
# Min     time: 2.569 ms
# Max     time: 5.680 ms
# Theoretical perf: 3524.917 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356696.15625

# Reduction - packed 8 accumulators SSE - float32
# Collected 1000 samples in 2.502 seconds
# Average time: 2.498 ms
# Stddev  time: 0.213 ms
# Min     time: 2.299 ms
# Max     time: 5.111 ms
# Theoretical perf: 4003.616 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356923.1875

# Reduction - prod impl - float32
# Collected 1000 samples in 2.442 seconds
# Average time: 2.439 ms
# Stddev  time: 0.162 ms
# Min     time: 2.274 ms
# Max     time: 4.916 ms
# Theoretical perf: 4100.865 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -170817.09375

# Reduction - packed 8 accumulators AVX - float32
# Collected 1000 samples in 2.567 seconds
# Average time: 2.563 ms
# Stddev  time: 0.186 ms
# Min     time: 2.373 ms
# Max     time: 5.158 ms
# Theoretical perf: 3902.290 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356915.03125

# Reduction - packed 16 accumulators AVX - float32
# Collected 1000 samples in 2.580 seconds
# Average time: 2.576 ms
# Stddev  time: 0.230 ms
# Min     time: 2.371 ms
# Max     time: 5.134 ms
# Theoretical perf: 3881.285 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356914.875

################################################################
## Bench on i5 Broadwell - prod implementation is OpenMP-enabled
# Unfortunately we are memory-bandwith bound

# Warmup: 1.1888 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - packed 4 accumulators SSE - float32
# Collected 1000 samples in 2.825 seconds
# Average time: 2.824 ms
# Stddev  time: 0.259 ms
# Min     time: 2.552 ms
# Max     time: 5.193 ms
# Theoretical perf: 3540.637 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356696.15625

# Reduction - packed 8 accumulators SSE - float32
# Collected 1000 samples in 2.498 seconds
# Average time: 2.498 ms
# Stddev  time: 0.227 ms
# Min     time: 2.266 ms
# Max     time: 4.867 ms
# Theoretical perf: 4003.727 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356923.1875

# Reduction - prod impl - float32
# Collected 1000 samples in 2.129 seconds
# Average time: 2.129 ms
# Stddev  time: 0.190 ms
# Min     time: 1.925 ms
# Max     time: 3.508 ms
# Theoretical perf: 4697.260 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356874.40625

# Reduction - packed 8 accumulators AVX - float32
# Collected 1000 samples in 2.539 seconds
# Average time: 2.538 ms
# Stddev  time: 0.255 ms
# Min     time: 2.327 ms
# Max     time: 5.358 ms
# Theoretical perf: 3939.552 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356915.03125

# Reduction - packed 16 accumulators AVX - float32
# Collected 1000 samples in 2.528 seconds
# Average time: 2.528 ms
# Stddev  time: 0.221 ms
# Min     time: 2.336 ms
# Max     time: 5.301 ms
# Theoretical perf: 3955.728 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# -356914.875

# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

# Similar to sum reduction bench but with max
# as compiler should be able to reorder even the naive case


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
  var accum = float32(-Inf)
  bench("Reduction - 1 accumulator - simple iter", accum):
    for i in 0 ..< a.size:
      accum = max(accum, a.storage.raw_data[i])

proc mainBench_1_accum_macro(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Reduction - 1 accumulator - macro iter", accum):
    forEachContiguousSerial val in a:
      accum = max(accum, val)

proc mainBench_2_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Reduction - 2 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(2)
    var
      accum1 = float32(-Inf)
    for i in countup(0, unroll_stop - 1, 2):
      accum = max(accum, a.storage.raw_data[i])
      accum1 = max(accum1, a.storage.raw_data[i+1])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_data[i])
    accum = max(accum, accum1)

proc mainBench_3_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Reduction - 3 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(3)
    var
      accum1 = float32(-Inf)
      accum2 = float32(-Inf)
    for i in countup(0, unroll_stop - 1, 3):
      accum = max(accum, a.storage.raw_data[i])
      accum1 = max(accum1, a.storage.raw_data[i+1])
      accum2 = max(accum2, a.storage.raw_data[i+2])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_data[i])
    accum = max(accum, max(accum1, accum2))

proc mainBench_4_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Reduction - 4 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(4)
    var
      accum1 = float32(-Inf)
      accum2 = float32(-Inf)
      accum3 = float32(-Inf)
    for i in countup(0, unroll_stop - 1, 4):
      accum = max(accum, a.storage.raw_data[i])
      accum1 = max(accum1, a.storage.raw_data[i+1])
      accum2 = max(accum2, a.storage.raw_data[i+2])
      accum3 = max(accum3, a.storage.raw_data[i+3])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_data[i])
    accum = max(accum2, accum1)
    accum2 = max(accum2, accum3)
    accum = max(accum, accum2)

proc mainBench_5_accum_simple(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Reduction - 5 accumulators - simple iter", accum):
    let size = a.size
    let unroll_stop = size.round_down_multiple(5)
    var
      accum1 = float32(-Inf)
      accum2 = float32(-Inf)
      accum3 = float32(-Inf)
      accum4 = float32(-Inf)
    for i in countup(0, unroll_stop - 1, 5):
      accum = max(accum, a.storage.raw_data[i])
      accum1 = max(accum1, a.storage.raw_data[i+1])
      accum2 = max(accum2, a.storage.raw_data[i+2])
      accum3 = max(accum3, a.storage.raw_data[i+3])
      accum4 = max(accum4, a.storage.raw_data[i+4])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_data[i])
    accum2 = max(accum2, max(accum3, accum4))
    accum = max(accum, accum1)
    accum = max(accum, accum2)

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

#### Bench - this is awfully slow ...

# Warmup: 1.8282 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - 1 accumulator - simple iter - float32
# Collected 1000 samples in 21.723 seconds
# Average time: 21.719 ms
# Stddev  time: 6.931 ms
# Min     time: 19.716 ms
# Max     time: 57.036 ms
# Theoretical perf: 460.436 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 1 accumulator - macro iter - float32
# Collected 1000 samples in 20.310 seconds
# Average time: 20.306 ms
# Stddev  time: 0.689 ms
# Min     time: 19.696 ms
# Max     time: 26.119 ms
# Theoretical perf: 492.467 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 2 accumulators - simple iter - float32
# Collected 1000 samples in 21.408 seconds
# Average time: 21.404 ms
# Stddev  time: 0.484 ms
# Min     time: 21.165 ms
# Max     time: 29.999 ms
# Theoretical perf: 467.209 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 3 accumulators - simple iter - float32
# Collected 1000 samples in 18.117 seconds
# Average time: 18.112 ms
# Stddev  time: 0.477 ms
# Min     time: 17.841 ms
# Max     time: 24.668 ms
# Theoretical perf: 552.106 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 4 accumulators - simple iter - float32
# Collected 1000 samples in 19.172 seconds
# Average time: 19.168 ms
# Stddev  time: 0.511 ms
# Min     time: 18.925 ms
# Max     time: 28.118 ms
# Theoretical perf: 521.705 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999994039535522

# Reduction - 5 accumulators - simple iter - float32
# Collected 1000 samples in 18.725 seconds
# Average time: 18.721 ms
# Stddev  time: 0.441 ms
# Min     time: 18.483 ms
# Max     time: 27.379 ms
# Theoretical perf: 534.164 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313


#####

# Assembly for FP32 max:
# +0x00	pushq               %rbp
# +0x01	movq                %rsp, %rbp
# +0x04	movaps              %xmm1, %xmm2
# +0x07	cmpless             %xmm0, %xmm2
# +0x0c	andps               %xmm2, %xmm0
# +0x0f	andnps              %xmm1, %xmm2
# +0x12	orps                %xmm2, %xmm0
# +0x15	popq                %rbp
# +0x16	retq
# +0x17	nopw                (%rax,%rax)

# And the simple loop is not unrolled ...

# +0x98	    nopl                (%rax,%rax)
# +0xa0	        movq                120(%r12), %rax
# +0xa5	        movq                (%rax), %rax
# +0xa8	        movss               (%rax,%r15,4), %xmm1
# +0xae	        callq               "max_FD6II0MuRlcOS7s9alzR0nA"
# +0xb3	        incq                %r15
# +0xb6	        cmpq                %r15, %rbx
# +0xb9	        jne                 "mainBench1_accum_simple_WC3Emwg71YNnFD8IeLSnQA+0xa0"


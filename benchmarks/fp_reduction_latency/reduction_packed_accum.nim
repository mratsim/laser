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


func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

proc mainBench_4_packed_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 4 accumulators", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(4)
    var accums{.align_variable.}: array[4, float32]
    let ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 4):
      accums[0] += ptr_data[i]
      accums[1] += ptr_data[i+1]
      accums[2] += ptr_data[i+2]
      accums[3] += ptr_data[i+3]
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    accums[0] += accums[2]
    accums[1] += accums[3]
    accum += accums[0]
    accum += accums[1]

proc mainBench_8_packed_accums(a: Tensor[float32], nb_samples: int) =
  var accum = 0'f32
  bench("Reduction - packed 8 accumulators", accum):
    let size = a.size
    let unroll_stop = size.round_down_power_of_2(8)
    var accums{.align_variable.}: array[8, float32]
    let ptr_data = a.unsafe_raw_data()
    for i in countup(0, unroll_stop - 1, 8):
      accums[0] += ptr_data[i]
      accums[1] += ptr_data[i+1]
      accums[2] += ptr_data[i+2]
      accums[3] += ptr_data[i+3]
      accums[4] += ptr_data[i+4]
      accums[5] += ptr_data[i+5]
      accums[6] += ptr_data[i+6]
      accums[7] += ptr_data[i+7]
    for i in unroll_stop ..< size:
      accum += ptr_data[i]
    accums[0] += accums[4]
    accums[1] += accums[5]
    accums[2] += accums[6]
    accums[3] += accums[7]

    accums[0] += accums[2]
    accums[1] += accums[3]
    accum += accums[0]
    accum += accums[1]

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
    mainBench_4_packed_accums(a, 1000)
    mainBench_8_packed_accums(a, 1000)


# Warmup: 1.1962 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Reduction - packed 4 accumulators - float64
# Collected 1000 samples in 4.084 seconds
# Average time: 4.080ms
# Stddev  time: 0.486ms
# Min     time: 3.457ms
# Max     time: 10.308ms

# Display output[[0,0]] to make sure it's not optimized away
# -356696.125

# Reduction - packed 8 accumulators - float64
# Collected 1000 samples in 4.106 seconds
# Average time: 4.103ms
# Stddev  time: 0.466ms
# Min     time: 3.463ms
# Max     time: 9.341ms

# Display output[[0,0]] to make sure it's not optimized away
# -356909.59375


### Assembly
# GCC doesn't want to generate SSE even on x86_64

# +0x108	    xorps               %xmm0, %xmm0
# +0x10b	    xorps               %xmm1, %xmm1
# +0x10e	    xorps               %xmm2, %xmm2
# +0x111	    xorps               %xmm3, %xmm3
# +0x114	    xorl                %edx, %edx
# +0x116	    nopw                %cs:(%rax,%rax)
# +0x120	        addss               -28(%rbx,%rdx,4), %xmm3
# +0x126	        addss               -24(%rbx,%rdx,4), %xmm2
# +0x12c	        addss               -20(%rbx,%rdx,4), %xmm1
# +0x132	        addss               -16(%rbx,%rdx,4), %xmm0
# +0x138	        addss               -12(%rbx,%rdx,4), %xmm3
# +0x13e	        addss               -8(%rbx,%rdx,4), %xmm2
# +0x144	        addss               -4(%rbx,%rdx,4), %xmm1
# +0x14a	        addss               (%rbx,%rdx,4), %xmm0
# +0x14f	        addq                $8, %rdx
# +0x153	        addq                $2, %rsi
# +0x157	        jne                 "mainBench4_packed_accums_FJ4b0cMfz8qgUolj9a8gSsA+0x120"


# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

# Similar to sum reduction bench but with max
# as compiler should be able to reorder even the naive case


import
  ../../laser/strided_iteration/foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays],
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
      accum = max(accum, a.storage.raw_buffer[i])

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
      accum = max(accum, a.storage.raw_buffer[i])
      accum1 = max(accum1, a.storage.raw_buffer[i+1])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_buffer[i])
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
      accum = max(accum, a.storage.raw_buffer[i])
      accum1 = max(accum1, a.storage.raw_buffer[i+1])
      accum2 = max(accum2, a.storage.raw_buffer[i+2])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_buffer[i])
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
      accum = max(accum, a.storage.raw_buffer[i])
      accum1 = max(accum1, a.storage.raw_buffer[i+1])
      accum2 = max(accum2, a.storage.raw_buffer[i+2])
      accum3 = max(accum3, a.storage.raw_buffer[i+3])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_buffer[i])
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
      accum = max(accum, a.storage.raw_buffer[i])
      accum1 = max(accum1, a.storage.raw_buffer[i+1])
      accum2 = max(accum2, a.storage.raw_buffer[i+2])
      accum3 = max(accum3, a.storage.raw_buffer[i+3])
      accum4 = max(accum4, a.storage.raw_buffer[i+4])
    for i in unroll_stop ..< size:
      accum = max(accum, a.storage.raw_buffer[i])
    accum2 = max(accum2, max(accum3, accum4))
    accum = max(accum, accum1)
    accum = max(accum, accum2)

proc mainBench_packed_sse_prod(a: Tensor[float32], nb_samples: int) =
  var accum = float32(-Inf)
  bench("Max reduction - prod impl", accum):
    accum = max(accum, reduce_max(a.storage.raw_buffer, a.size))

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
    mainBench_packed_sse_prod(a, 1000)
    mainBench_1_accum_simple(a, 1000)
    mainBench_1_accum_macro(a, 1000)
    mainBench_2_accum_simple(a, 1000)
    mainBench_3_accum_simple(a, 1000)
    mainBench_4_accum_simple(a, 1000)
    mainBench_5_accum_simple(a, 1000)

#### Bench - naive is awfully slow ...
# Issue opened in https://github.com/nim-lang/Nim/issues/9514

# Warmup: 1.1938 s, result 224 (displayed to avoid compiler optimizing warmup away)

# Max reduction - prod impl - float32
# Collected 1000 samples in 2.780 seconds
# Average time: 2.776 ms
# Stddev  time: 0.348 ms
# Min     time: 2.432 ms
# Max     time: 7.568 ms
# Theoretical perf: 3602.164 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 1 accumulator - simple iter - float32
# Collected 1000 samples in 20.024 seconds
# Average time: 20.019 ms
# Stddev  time: 0.611 ms
# Min     time: 19.669 ms
# Max     time: 28.292 ms
# Theoretical perf: 499.515 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 1 accumulator - macro iter - float32
# Collected 1000 samples in 20.452 seconds
# Average time: 20.447 ms
# Stddev  time: 1.382 ms
# Min     time: 19.642 ms
# Max     time: 31.142 ms
# Theoretical perf: 489.063 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 2 accumulators - simple iter - float32
# Collected 1000 samples in 21.111 seconds
# Average time: 21.107 ms
# Stddev  time: 1.972 ms
# Min     time: 19.817 ms
# Max     time: 31.056 ms
# Theoretical perf: 473.785 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 3 accumulators - simple iter - float32
# Collected 1000 samples in 18.397 seconds
# Average time: 18.393 ms
# Stddev  time: 0.747 ms
# Min     time: 17.889 ms
# Max     time: 25.060 ms
# Theoretical perf: 543.690 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313

# Reduction - 4 accumulators - simple iter - float32
# Collected 1000 samples in 18.173 seconds
# Average time: 18.168 ms
# Stddev  time: 0.724 ms
# Min     time: 17.704 ms
# Max     time: 26.864 ms
# Theoretical perf: 550.413 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999994039535522

# Reduction - 5 accumulators - simple iter - float32
# Collected 1000 samples in 18.444 seconds
# Average time: 18.440 ms
# Stddev  time: 1.286 ms
# Min     time: 17.805 ms
# Max     time: 29.549 ms
# Theoretical perf: 542.298 MFLOP/s

# Display sum of samples sums to make sure it's not optimized away
# 0.9999996423721313


##############################

# Assembly for naive FP32 max:
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

##############################
# With fastmath + march-native
# max is properly detected bu we have a huge function call overhead

# +0x00	pushq               %rbp
# +0x01	movq                %rsp, %rbp
# +0x04	vmaxss              %xmm1, %xmm0, %xmm0
# +0x08	popq                %rbp
# +0x09	retq
# +0x0a	nopw                (%rax,%rax)

# ------------------------------------------

# +0x99	    nopl                (%rax)
# +0xa0	        movq                120(%r12), %rax
# +0xa5	        movq                (%rax), %rax
# +0xa8	        vmovss              (%rax,%r15,4), %xmm1
# +0xae	        callq               "max_FD6II0MuRlcOS7s9alzR0nA"
# +0xb3	        addq                $1, %r15
# +0xb7	        cmpq                %r15, %rbx
# +0xba	        jne                 "mainBench1_accum_simple_WC3Emwg71YNnFD8IeLSnQA_2+0xa0"

###########################################################
# The SSE3 implementation hits memory bandwidth bottleneck:

# +0x67	nopw                (%rax,%rax)
# +0x70	    maxps               (%rdi,%rcx,4), %xmm1
# +0x74	    maxps               16(%rdi,%rcx,4), %xmm3
# +0x79	    maxps               32(%rdi,%rcx,4), %xmm2
# +0x7e	    maxps               48(%rdi,%rcx,4), %xmm4
# +0x83	    maxps               64(%rdi,%rcx,4), %xmm1
# +0x88	    maxps               80(%rdi,%rcx,4), %xmm3 # Bottleneck here
# +0x8d	    maxps               96(%rdi,%rcx,4), %xmm2
# +0x92	    maxps               112(%rdi,%rcx,4), %xmm4
# +0x97	    addq                $32, %rcx
# +0x9b	    addq                $2, %rdx
# +0x9f	    jne                 "max_sse3_skqjc9ccvpz3qvNJidlNb9aw+0x70"

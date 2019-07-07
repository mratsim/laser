# Apache v2 License
# Mamy Ratsimbazafy


# ##########################################
# Tensor primitives

import
  ../../laser/strided_iteration/foreach,
  ../../laser/tensor/[allocator, datatypes, initialization],
  ../../laser/[compiler_optim_hints, dynamic_stack_arrays],
  ../../laser/simd,
  ../../laser/primitives/reductions

withCompilerOptimHints()

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

template printStats(name: string, output: typed) {.dirty.} =
  echo "\n" & name
  echo &"Collected {stats.n} samples in {global_stop - global_start:>4.3f} seconds"
  echo &"Average time: {stats.mean * 1000 :>4.3f} ms"
  echo &"Stddev  time: {stats.standardDeviationS * 1000 :>4.3f} ms"
  echo &"Min     time: {stats.min * 1000 :>4.3f} ms"
  echo &"Max     time: {stats.max * 1000 :>4.3f} ms"
  echo &"Perf:         {req_ops.float / stats.mean / float(10^9):>4.3f} GEXPOP/s"
  echo "\nDisplay output[0] to make sure it's not optimized away"
  echo output[0] # Prevents compiler from optimizing stuff away

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

const
  N     = 100*50000 # For example for use in softmax for a batch of 100 with dictionary size of 50000 words
  NbSamples = 100

let req_ops = N
let req_bytes = sizeof(float32) * N

# #############################################
import ../../laser/simd

func round_down_power_of_2(x: Natural, step: static Natural): int {.inline.} =
  static: assert (step and (step - 1)) == 0, "Step must be a power of 2"
  result = x and not(step - 1)

import ospaths, strutils
from os import DirSep

const cSourcesPath = currentSourcePath.rsplit(DirSep, 1)[0] & '/'
{.passC: "-I" & cSourcesPath .}

proc benchBaseline(a: Tensor[float32], nb_samples: int) =
  var output = newTensor[float32](a.shape)
  bench("Baseline <math.h>"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    output.setZero() # We zero memory between computation
  do:
    # Main work
    for i in 0 ..< a.size:
      output.storage.raw_data[i] = exp(a.storage.raw_data[i])

template vectorize(
      wrapped_func,
      funcname,
      simd_load,
      simd_store: untyped,
      unroll_factor: int) =
  proc funcname(dst, src: ptr UncheckedArray[float32], len: Natural) =
    let unroll_stop = len.round_down_power_of_2(unroll_factor)

    for i in countup(0, unroll_stop - 1, unroll_factor):
      dst[i].addr.simd_store src[i].addr.simd_load.wrapped_func
    for i in unroll_stop ..< len:
      dst[i] = src[i]

{.passC: "-DUSE_SSE2".}
proc sse_mathfun_exp_ps(x: m128): m128 {.importc: "exp_ps", header: cSourcesPath & "lib_sse_mathfun.h".}
vectorize(sse_mathfun_exp_ps, sse_mathfun_exp_ps, mm_load_ps, mm_store_ps, 4)

proc benchSSEMathfun(a: Tensor[float32], nb_samples: int) =
  var output = newTensor[float32](a.shape)
  bench("SSE mathfun"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    output.setZero() # We zero memory between computation
  do:
    # Main work
    sse_mathfun_exp_ps(output.storage.raw_data, a.storage.raw_data, a.size)

proc sse_fmath_exp_ps(x: m128): m128 {.importcpp: "fmath::exp_ps(@)", header: cSourcesPath & "lib_fmath.hpp".}
vectorize(sse_fmath_exp_ps, sse_fmath_exp_ps, mm_load_ps, mm_store_ps, 4)

proc benchSSE_fmath(a: Tensor[float32], nb_samples: int) =
  var output = newTensor[float32](a.shape)
  bench("SSE fmath"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    output.setZero() # We zero memory between computation
  do:
    # Main work
    sse_fmath_exp_ps(output.storage.raw_data, a.storage.raw_data, a.size)

{.compile: "lib_sse_exp.c".}
proc fast_exp_sse(x: m128): m128 {.importc.}
vectorize(fast_exp_sse, fast_exp_sse, mm_load_ps, mm_store_ps, 4)

proc benchSSE_fast_exp_sse(a: Tensor[float32], nb_samples: int) =
  var output = newTensor[float32](a.shape)
  bench("SSE fast_exp_sse (low order polynomial)"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    output.setZero() # We zero memory between computation
  do:
    # Main work
    fast_exp_sse(output.storage.raw_data, a.storage.raw_data, a.size)

import ../../laser/primitives/simd_math/exp_log_sse2
vectorize(exp, exp_float32x4_sse2, mm_load_ps, mm_store_ps, 4)

proc benchProdImplSSE2(a: Tensor[float32], nb_samples: int) =
  var output = newTensor[float32](a.shape)
  bench("SSE2 Prod implementation"):
    # Initialisation, not measured apart for the "Collected n samples in ... seconds"
    output.setZero() # We zero memory between computation
  do:
    # Main work
    exp_float32x4_sse2(output.storage.raw_data, a.storage.raw_data, a.size)

# ###########################################

when defined(fast_math):
  {.passC:"-ffast-math".}

when defined(march_native):
  {.passC:"-march=native".}


when isMainModule:
  randomize(42) # For reproducibility
  warmup()
  echo ""
  echo &"A - tensor shape: [{N}]"
  echo &"Required number of operations: {req_ops.float / float(10^6):>9.3f} millions"
  echo &"Required bytes:                {req_bytes.float / float(10^6):>9.3f} MB"
  echo &"Arithmetic intensity:          {req_ops.float / req_bytes.float:>9.3f} FLOP/byte"
  block:
    let a = randomTensor([N], -1.0'f32 .. 1.0'f32)
    echo "a[0]: " & $a[0]
    benchBaseline(a, NbSamples)
    benchSSEMathfun(a, NbSamples)
    benchSSE_fmath(a, NbSamples)
    benchSSE_fast_exp_sse(a, NbSamples)
    benchProdImplSSE2(a, NbSamples)

######################################################
## Bench on i5-5257U Broadwell - serial implementation

# Warmup: 1.2104 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A - tensor shape: [5000000]
# Required number of operations:     5.000 millions
# Required bytes:                   20.000 MB
# Arithmetic intensity:              0.250 FLOP/byte
# a[0]: -0.9999997019767761

# Baseline <math.h>
# Collected 100 samples in 2.625 seconds
# Average time: 24.991 ms
# Stddev  time: 0.675 ms
# Min     time: 24.576 ms
# Max     time: 29.511 ms
# Perf:         0.200 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# SSE mathfun
# Collected 100 samples in 1.131 seconds
# Average time: 10.069 ms
# Stddev  time: 0.253 ms
# Min     time: 9.948 ms
# Max     time: 11.963 ms
# Perf:         0.497 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# SSE fmath
# Collected 100 samples in 0.726 seconds
# Average time: 6.027 ms
# Stddev  time: 0.433 ms
# Min     time: 5.700 ms
# Max     time: 8.140 ms
# Perf:         0.830 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# SSE fast_exp_sse (low order polynomial)
# Collected 100 samples in 0.599 seconds
# Average time: 4.759 ms
# Stddev  time: 0.173 ms
# Min     time: 4.618 ms
# Max     time: 5.791 ms
# Perf:         1.051 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3682391047477722


######################################################
## Bench on i9-9980XE Skylake-X - serial implementation
## OC @ 4.1 GHz

# Warmup: 0.9044 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A - tensor shape: [5000000]
# Required number of operations:     5.000 millions
# Required bytes:                   20.000 MB
# Arithmetic intensity:              0.250 FLOP/byte
# a[0]: -0.9999997019767761

# Baseline <math.h>
# Collected 100 samples in 1.718 seconds
# Average time: 16.515 ms
# Stddev  time: 0.094 ms
# Min     time: 15.974 ms
# Max     time: 16.608 ms
# Perf:         0.303 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# SSE mathfun
# Collected 100 samples in 0.937 seconds
# Average time: 8.699 ms
# Stddev  time: 0.073 ms
# Min     time: 8.403 ms
# Max     time: 8.813 ms
# Perf:         0.575 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795695304871

# SSE fmath
# Collected 100 samples in 0.589 seconds
# Average time: 5.218 ms
# Stddev  time: 0.022 ms
# Min     time: 5.179 ms
# Max     time: 5.279 ms
# Perf:         0.958 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# SSE fast_exp_sse (low order polynomial)
# Collected 100 samples in 0.396 seconds
# Average time: 3.287 ms
# Stddev  time: 0.026 ms
# Min     time: 3.192 ms
# Max     time: 3.376 ms
# Perf:         1.521 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3682391047477722

# SSE2 Prod implementation
# Collected 100 samples in 0.673 seconds
# Average time: 6.064 ms
# Stddev  time: 0.024 ms
# Min     time: 6.017 ms
# Max     time: 6.158 ms
# Perf:         0.825 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 0.3678795397281647

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpuinfo, ../compiler_optim_hints,
  ./private/align_unroller

when defined(i386) or defined(amd_64):
  import ./sum_sse3

func sum_fallback(data: ptr UncheckedArray[float32], len: Natural): float32 =
  ## Fallback kernel for sum reduction
  ## We use 2 accumulators as they should exhibits the best compromise
  ## of speed and code-size across architectures
  ## especially when fastmath is turned on.
  ## Benchmarked: 2x faster than naive reduction and 2x slower than the SSE3 kernel

  withCompilerOptimHints()
  let data{.restrict.} = data

  # Loop peeling is left at the compiler discretion,
  # if optimizing for code size is desired
  const step = 2
  var accum1 = 0'f32
  let unroll_stop = len.round_step_down(step)
  for i in countup(0, unroll_stop - 1, 2):
    result += data[i]
    accum1 += data[i+1]
  result += accum1
  if unroll_stop != len:
    result += data[unroll_stop] # unroll_stop = len -1 last element

func sum_kernel*(data: ptr UncheckedArray[float32], len: Natural): float32 =
  ## Does a sum reduction on a contiguous range of float32
  ## Warning:
  ##   This kernel considers floating-point addition commutative
  ##   and will reorder additions.

  # Note that the kernel is memory-bandwith bound once the
  # CPU pipeline is saturated. Using AVX doesn't help
  # loading data from memory faster.
  when defined(i386) or defined(amd_64):
    if cpuinfo_has_x86_sse3():
      return sum_sse3(data, len)
  return sum_fallback(data, len)

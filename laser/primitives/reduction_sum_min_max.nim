# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpuinfo, ../compiler_optim_hints,
  ../private/align_unroller,
  ../openmp

when defined(i386) or defined(amd_64):
  import ./reduction_sum_min_max_sse3

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

proc sum_kernel*(data: ptr UncheckedArray[float32], len: Natural): float32 {.sideeffect.}=
  ## Does a sum reduction on a contiguous range of float32
  ## Warning:
  ##   This kernel considers floating-point addition associative
  ##   and will reorder additions.
  ## Due to parallel reduction and floating point rounding,
  ## same input can give different results depending on thread timings

  # Note that the kernel is memory-bandwith bound once the
  # CPU pipeline is saturated. Using AVX doesn't help
  # loading data from memory faster.
  when not defined(openmp):
    when defined(i386) or defined(amd_64):
      if cpuinfo_has_x86_sse3():
        return sum_sse3(data, len)
    return sum_fallback(data, len)
  else:
    # TODO: Fastest between a padded seq, a critical section, OMP atomics or CPU atomics?
    let
      max_threads = omp_get_max_threads()
      omp_condition = OMP_MEMORY_BOUND_GRAIN_SIZE * max_threads < len
      sse3 = cpuinfo_has_x86_sse3()

    {.emit: "#pragma omp parallel if (`omp_condition`)".}
    block:
      let
        nb_chunks = omp_get_num_threads()
        whole_chunk_size = len div nb_chunks
        thread_id = omp_get_thread_num()
        `chunk_offset`{.inject.} = whole_chunk_size * thread_id
        `chunk_size`{.inject.} =  if thread_id < nb_chunks - 1: whole_chunk_size
                                    else: len - chunk_offset
      block:
        let p_chunk{.restrict.} = cast[ptr UncheckedArray[float32]](
                                    data[chunk_offset].addr
                                  )
        when defined(i386) or defined(amd_64):
          let local_sum = if sse3: sum_sse3(p_chunk, chunk_size)
                          else: sum_fallback(p_chunk, chunk_size)
        else:
          let local_sum = sum_fallback(p_chunk, chunk_size)

        {.emit: "#pragma omp atomic".}
        {.emit: "`result` += `local_sum`;".}

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../cpuinfo, ../compiler_optim_hints,
  ../private/align_unroller,
  ../openmp

when defined(i386) or defined(amd_64):
  import ./simd_math/reductions_sse3

# Fallback reduction operations
# ----------------------------------------------------------------------------------

template reduction_op_fallback(op_name, initial_val, scalar_op, merge_op: untyped) =
  func `op_name`(data: ptr UncheckedArray[float32], len: Natural): float32 =
    ## Fallback kernel for reduction
    ## We use 2 accumulators as they should exhibits the best compromise
    ## of speed and code-size across architectures
    ## especially when fastmath is turned on.
    ## Benchmarked: 2x faster than naive reduction and 2x slower than the SSE3 kernel

    withCompilerOptimHints()
    let data{.restrict.} = data

    # Loop peeling is left at the compiler discretion,
    # if optimizing for code size is desired
    const step = 2
    var accum1 = initial_val
    let unroll_stop = len.round_step_down(step)
    for i in countup(0, unroll_stop - 1, 2):
      result = scalar_op(result, data[i])
      accum1 = scalar_op(accum1, data[i+1])
    result = scalar_op(result, accum1)
    if unroll_stop != len:
      # unroll_stop = len - 1 last element
      result = scalar_op(result, data[unroll_stop])

reduction_op_fallback(sum_fallback, 0'f32, `+`, `+`)
reduction_op_fallback(min_fallback, float32(Inf), min, min)
reduction_op_fallback(max_fallback, float32(-Inf), max, max)

# Reduction primitives
# ----------------------------------------------------------------------------------

template gen_reduce_kernel_f32(
          kernel_name: untyped{ident},
          initial_val: static float32,
          sse3_kernel, fallback_kernel: untyped{ident},
          merge_op: untyped
          ): untyped =

  proc `kernel_name`*(data: ptr (float32 or UncheckedArray[float32]), len: Natural): float32 {.sideeffect.}=
    ## Does a reduction on a contiguous range of float32
    ## Warning:
    ##   This kernel considers the reduction operation associative
    ##   and will reorder operations.
    ## Due to parallel reduction and floating point rounding,
    ## the same input can give different results depending on thread timings
    ## for some operations like addition

    # Note that the kernel is memory-bandwith bound once the
    # CPU pipeline is saturated. Using AVX doesn't help
    # loading data from memory faster.

    withCompilerOptimHints()
    let data{.restrict.} = cast[ptr UncheckedArray[float32]](data)

    when not defined(openmp):
      when defined(i386) or defined(amd_64):
        if cpuinfo_has_x86_sse3():
          return `sse3_kernel`(data, len)
      return `fallback_kernel`(data, len)
    else:
      result = initial_val

      let
        omp_condition = OMP_MEMORY_BOUND_GRAIN_SIZE * omp_get_max_threads() < len
        sse3 = cpuinfo_has_x86_sse3()

      omp_parallel_if(omp_condition):
        omp_chunks(len, chunk_offset, chunk_size):
          let local_ptr_chunk{.restrict.} = cast[ptr UncheckedArray[float32]](
                                      data[chunk_offset].addr
                                    )
          when defined(i386) or defined(amd_64):
            let local_accum = if sse3: `sse3_kernel`(local_ptr_chunk, chunk_size)
                            else: `fallback_kernel`(local_ptr_chunk, chunk_size)
          else:
            let local_accum = `fallback_kernel`(local_ptr_chunk, chunk_size)

          omp_critical:
            result = merge_op(result, local_accum)

gen_reduce_kernel_f32(
      reduce_sum,
      0'f32,
      sum_sse3, sum_fallback,
      `+`
    )

gen_reduce_kernel_f32(
      reduce_min,
      float32(Inf),
      min_sse3, min_fallback,
      min
    )

gen_reduce_kernel_f32(
      reduce_max,
      float32(-Inf),
      max_sse3, max_fallback,
      max
    )

# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ./omp_tuning, ./omp_mangling, ./omp_mangling
export omp_suffix

template omp_parallel_for*(
      index: untyped,
      length: Natural,
      omp_threshold: static Natural,
      omp_grain_size: static Positive,
      use_simd: static bool = true,
      body: untyped
      ) =
  ## Parallel loop
  ## Inputs:
  ##   - `index`, the iteration index, similar to
  ##     for `index` in 0 ..< length:
  ##       doSomething(`index`)
  ##   - `length`, the number of elements to iterate on
  ##   - `omp_threshold`, the minimal amount of total work before
  ##     a loop is parallelized. A value of 0 will always parallelize
  ##     the loop to all cores + hyperthreading, ignoring
  ##     the `omp_grain_size` parameter.
  ##   - `omp_grain_size`, the minimal amount of work per thread.
  ##   - `use_simd`, instruct the compiler to unroll the loops for `simd` use.
  ##     For example, for float32:
  ##     for i in 0..<16:
  ##       x[i] += y[i]
  ##     will be unrolled to take 128, 256 or 512-bit to use SSE, AVX or AVX512.
  ##     for 256-bit AVX:
  ##     for i in countup(0, 2, 8): # Step 8 by 8
  ##       x[i]   += y[i]
  ##       x[i+1] += y[i+1]
  ##       x[i+2] += y[i+2]
  ##       ...
  when not defined(openmp) or omp_threshold == 0:
    ## When OpenMP is not defined we use this simple loop as fallback
    ## This way, the compiler will still be provided "simd" vectorization hints
    when use_simd:
      for `index`{.inject.} in `||`(0, length - 1, "simd"):
        body
    else:
      for `index`{.inject.} in 0||(length-1):
        body
  else:
    const # Workaround to expose an unique symbol in C.
      ompsize_Csymbol = "ompsize_" & omp_suffix(genNew = true)
      nb_threads_Csymbol = "nb_threads_" & omp_suffix(genNew = false)

    let ompsize {.exportc: "ompsize_" & omp_suffix(genNew = false).} = length
    let nb_threads {.exportc: "nb_threads_" & omp_suffix(genNew = false).} = (
      min(omp_get_max_threads(), ompsize div omp_grain_size)
    )

    const omp_annotation = (when use_simd:"simd " else: "") &
      "num_threads(" & nb_threads_Csymbol & ") " &
      "if(" & $ompthreshold & " < " & ompsize_Csymbol & ")"

    for `index`{.inject.} in `||`(0, ompsize - 1, omp_annotation):
      body

template omp_parallel_for_default*(
      index: untyped,
      length: Natural,
      body: untyped
      ) =
  ## This will be renamed omp_parallel_for once
  ## https://github.com/nim-lang/Nim/issues/9414 is solved.
  ## Compared to omp_parallel_for the following are set by default
  ## - omp_threshold:
  ##     The default `OMP_MEMORY_BOUND_THRESHOLD` is 512 elements.
  ##     512x float32 (4kB) elements take 2MB which can s processed efficiently
  ##     in a CPU L2 cache.
  ## - omp_grain_size:
  ##     The default `OMP_MEMORY_BOUND_GRAIN_SIZE` is suitable for
  ##     contiguous copy or add operations. It's 1024 and can be changed
  ##     by passing `-d:OMP_MEMORY_BOUND_GRAIN_SIZE=123456` during compilation.
  ##     A value of 1 will always parallelize the loop.
  ## - simd is used by default
  omp_parallel_for(
    index,
    length,
    omp_threshold = OMP_MEMORY_BOUND_THRESHOLD,
    omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
    use_simd = true,
    body)

template omp_parallel_blocks*(
    length: Natural,
    block_offset, block_size: untyped,
    omp_threshold: static Natural,
    omp_grain_size: static Positive,
    use_simd: static bool = true,
    body: untyped): untyped =
  ## Create a block range for each threads. You can use:
  ## `for index in block_offset ..< block_size:` or
  ## `zeroMem(foo[block_offset].addr, block_size)`
  ##
  ##
  ## Splits the input `length` into blocks and do a parallel loop
  ## on each block. The number of blocks depends on the number of cores at runtime.
  ## `block_offset` and `block_size` should be passed as undeclared identifiers.
  ## Within the template block they will contain the start offset and the length
  ## of the current thread block. I.e. their value is thread-specific.
  ##
  ## This is useful for non-contiguous processing as a replacement to omp_parallel_for
  ## or when operating on (contiguous) ranges for example for memset or memcpy

  when not defined(openmp):
    const block_offset = 0
    let block_size = length
    body
  else:
    let ompsize = length # If length is the result of a proc, call the proc only once
    let nb_blocks = if omp_threshold < ompsize:
      min(
        omp_get_max_threads(),
        max(1, ompsize div omp_grain_size) # if ompsize < omp_grain_size
      )
      else: 1

    let block_size = ompsize div nb_blocks

    when use_simd:
      for block_index in `||`(0, nb_blocks-1, "simd"):
        let `block_offset`{.inject.} = block_size * block_index
        let `block_size`{.inject.} =  if block_index < nb_blocks - 1: block_size
                                      else: ompsize - block_offset
        block:
          body
    else:
      for block_index in 0||(nb_blocks-1):
        let `block_offset`{.inject.} = block_size * block_index
        let `block_size`{.inject.} =  if block_index < nb_blocks - 1: block_size
                                      else: ompsize - block_offset
        block:
          body

template omp_parallel_blocks_default*(
    length: Natural,
    block_offset, block_size: untyped,
    body: untyped): untyped =
  ## This will be renamed omp_parallel_blocks once
  ## https://github.com/nim-lang/Nim/issues/9414 is solved.
  ## Compared to omp_parallel_for the following are set by default
  ## - omp_threshold:
  ##     The default `OMP_MEMORY_BOUND_THRESHOLD` is 512 elements.
  ##     512x float32 (4kB) elements take 2MB which can s processed efficiently
  ##     in a CPU L2 cache.
  ## - omp_grain_size:
  ##     The default `OMP_MEMORY_BOUND_GRAIN_SIZE` is suitable for
  ##     contiguous copy or add operations. It's 1024 and can be changed
  ##     by passing `-d:OMP_MEMORY_BOUND_GRAIN_SIZE=123456` during compilation.
  ##     A value of 1 will always parallelize the loop.
  ## - simd is used by default
  omp_parallel_blocks(
    length,
    block_offset, block_size,
    omp_threshold = OMP_MEMORY_BOUND_THRESHOLD,
    omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
    use_simd = true,
    body
  )

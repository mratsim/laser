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
  when omp_threshold == 0:
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

    const omp_simd =  when use_simd:"simd "
                      else: ""

    let nb_threads {.exportc: "nb_threads_" & omp_suffix(genNew = false).} = (
      min(omp_get_max_threads(), ompsize div omp_grain_size)
    )

    const omp_annotation = omp_simd &
      "num_threads(" & nb_threads_Csymbol & ") " &
      "if(" & $ompthreshold & " < " & ompsize_Csymbol & ")"

    for `index`{.inject.} in `||`(0, length - 1, omp_annotation):
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

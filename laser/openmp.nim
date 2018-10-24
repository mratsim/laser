# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ###############################################################
# Compile-time name mangling for OpenMP thresholds
# Workaround https://github.com/nim-lang/Nim/issues/9365
# and https://github.com/nim-lang/Nim/issues/9366
import random
from strutils import toHex

var mangling_rng {.compileTime.} = initRand(0x1337DEADBEEF)
var current_suffix {.compileTime.} = ""

proc omp_suffix*(genNew: static bool = false): static string =
  ## genNew:
  ##   if false, return the last suffix
  ##   else return a fresh one
  # This is exported because you cannot bind the symbol early enough
  # for exportc

  if genNew:
    current_suffix = mangling_rng.rand(high(uint32)).toHex
  result = current_suffix

# ################################################################
# Tuning

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

  {.pragma: omp, header:"omp.h".}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.} # This takes hyperthreading into account
  proc omp_get_thread_num*(): cint {.omp.}

else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0

# TODO tuning for architectures
# https://github.com/zy97140/omp-benchmark-for-pytorch
# https://github.com/zy97140/omp-benchmark-for-pytorch/blob/master/benchmark-data/IntelR-XeonR-CPU-E5-2669-v4.md
# https://github.com/zy97140/omp-benchmark-for-pytorch/blob/master/benchmark-data/IntelR-XeonR-Platinum-8180-CPU.md


const OMP_MEMORY_BOUND_GRAIN_SIZE*{.intdefine.} = 1024
  ## This is the minimum amount of work per physical cores
  ## for memory-bound processing.
  ## - "copy" and "addition" are considered memory-bound
  ## - "float division" can be considered 2x~4x more complex
  ##   and should be scaled down accordingly
  ## - "exp" and "sin" operations are compute-bound and
  ##   there is a perf boost even when processing
  ##   only 1000 items on 28 cores
  ##
  ## Launching 2 threads per core (HyperThreading) is probably desirable:
  ##   - https://medium.com/data-design/destroying-the-myth-of-number-of-threads-number-of-physical-cores-762ad3919880
  ##
  ## Raising the following parameters can have the following impact:
  ##   - number of sockets: higher, more over memory fetch
  ##   - number of memory channel: lower, less overhead per memory fetch
  ##   - RAM speed: lower, less overhead per memory fetch
  ##   - Private L2 cache: higher, feed more data per CPU
  ##   - Hyperthreading and cache associativity
  ##   - Cores, shared L3 cache: Memory contention

const OMP_NON_CONTIGUOUS_SCALE_FACTOR*{.intdefine.} = 4
  ## Due to striding computation, we can use a lower grainsize
  ## for non-contiguous tensors

const OMP_MEMORY_BOUND_THRESHOLD*{.intdefine.} = 512
  ## Minimum number of elements before reverting to serial processing
  ## Below this threshold the data can always stay in the L2 or L3 cache
  ## of a modern x86 processor: 512 * 4kB (float32) = 2MB
  ## Change this on ARM cores


# ################################################################

template attachGC(): untyped =
  discard
  # TODO: this creates too strange error messages
  # when --threads is not on: https://github.com/nim-lang/Nim/issues/9489
  # if(omp_get_thread_num()!=0):
  #     setupForeignThreadGc()

template detachGC(): untyped =
  discard
  # TODO: this creates too strange error messages
  # when --threads is not on: https://github.com/nim-lang/Nim/issues/9489
  # if(omp_get_thread_num()!=0):
  #     teardownForeignThreadGc()

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
        block: body
    else:
      for `index`{.inject.} in 0||(length-1):
        block: body
  else:
    const # Workaround to expose an unique symbol in C.
      ompsize_Csymbol = "ompsize_" & omp_suffix(genNew = true)
      nb_threads_Csymbol = "nb_threads_" & omp_suffix(genNew = false)

    let ompsize {.exportc: "ompsize_" & omp_suffix(genNew = false).} = length
    let nb_threads {.exportc: "nb_threads_" & omp_suffix(genNew = false).} = (
      min(omp_get_max_threads(), max(1, ompsize div omp_grain_size))
    )

    const omp_annotation = (when use_simd:"simd " else: "") &
      "num_threads(" & nb_threads_Csymbol & ") " &
      "if(" & $ompthreshold & " < " & ompsize_Csymbol & ")"

    for `index`{.inject.} in `||`(0, ompsize - 1, omp_annotation):
      attachGC()
      block: body
      detachGC()

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

template omp_parallel_chunks*(
    length: Natural, nb_chunks: var Natural,
    chunk_offset, chunk_size: untyped,
    omp_threshold: static Natural,
    omp_grain_size: static Positive,
    use_simd: static bool = true,
    body: untyped): untyped =
  ## Create a chunk for each thread. You can use:
  ## `for index in chunk_offset ..< chunk_size:` or
  ## `zeroMem(foo[chunk_offset].addr, chunk_size)`
  ##
  ##
  ## Splits the input `length` into chunks and do a parallel loop
  ## on each chunk. The number of chunks depends on the number of cores at runtime.
  ## `chunk_offset` and `chunk_size` should be passed as undeclared identifiers.
  ## Within the template scope they will contain the start offset and the length
  ## of the current thread chunk. I.e. their value is thread-specific.
  ##
  ## Use omp_get_thread_num() to get the current thread number
  ##
  ## This is useful for non-contiguous processing as a replacement to omp_parallel_for
  ## or when operating on (contiguous) ranges for example for memset or memcpy
  when not defined(openmp):
    nb_chunks = 1
    const `chunk_offset`{.inject.} = 0
    let `chunk_size`{.inject.} = length
    block: body
  else:
    let ompsize = length # If length is the result of a proc, call the proc only once
    nb_chunks = if omp_threshold < ompsize:
      min(
        omp_get_max_threads(),
        max(1, ompsize div omp_grain_size) # if ompsize < omp_grain_size
      )
      else: 1
    let whole_chunk_size = ompsize div nb_chunks

    when use_simd:
      for chunk_id in `||`(0, nb_chunks-1, "simd"):
        let `chunk_offset`{.inject.} = whole_chunk_size * chunk_id
        let `chunk_size`{.inject.} =  if chunk_id < nb_chunks - 1: whole_chunk_size
                                      else: ompsize - chunk_offset
        # attachGC()
        block: body
        # detachGC()
    else:
      for chunk_id in 0||(nb_chunks-1):
        let `chunk_offset`{.inject.} = whole_chunk_size * chunk_id
        let `chunk_size`{.inject.} =  if chunk_id < nb_chunks - 1: whole_chunk_size
                                      else: ompsize - chunk_offset
        attachGC()
        block: body
        detachGC()

template omp_parallel_chunks_default*(
    length: Natural, nb_chunks: var Natural,
    chunk_offset, chunk_size: untyped,
    body: untyped): untyped =
  ## This will be renamed omp_parallel_chunks once
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
  omp_parallel_chunks(
    length, nb_chunks,
    chunk_offset, chunk_size,
    omp_threshold = OMP_MEMORY_BOUND_THRESHOLD,
    omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
    use_simd = true,
    body
  )

template omp_parallel*(body: untyped): untyped =
  {.emit: "#pragma omp parallel".}
  block:
    attachGC()
    body
    detachGC()


template omp_critical*(body: untyped): untyped =
  {.emit: "#pragma omp critical".}
  block:
    body

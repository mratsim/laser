# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import math

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

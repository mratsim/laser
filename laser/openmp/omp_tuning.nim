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
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}

else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0

# TODO tuning for architectures
# https://github.com/zy97140/omp-benchmark-for-pytorch

when true:
  const
    # i7-5960X (Haswell-E, 2014)
    # 8 cores @3Ghz, 2MB L2, 20MB L3
    OMP_BASE_THRESHOLD* {.intdefine.} = 8192
    OMP_COMPLEXITY_SCALE_FACTOR* {.intdefine.} = 4
elif false:
  const
    # Xeon CPU E5-2699v4 (Broadwell EP, 2016)
    # 22 cores @2.2Ghz, 5.5MB L2, 50MB L3
    OMP_BASE_THRESHOLD* {.intdefine.} = 16384
    OMP_COMPLEXITY_SCALE_FACTOR* {.intdefine.} = 2
elif false:
  const
    # Xeon Platinum 8180 (Skylake SP, 2017)
    # 28 cores @2.5Ghz, 28MB L2, 38.5MB L3
    OMP_BASE_THRESHOLD* {.intdefine.} = 65535
    OMP_COMPLEXITY_SCALE_FACTOR* {.intdefine.} = 2

const OMP_NON_CONTIGUOUS_SCALE_FACTOR*{.intdefine.} = 4

const
  OMP_LOW_COMPLEXITY_THRESHOLD* {.intdefine.} = OMP_BASE_THRESHOLD
  OMP_MID_COMPLEXITY_THRESHOLD* {.intdefine.} = OMP_BASE_THRESHOLD div OMP_COMPLEXITY_SCALE_FACTOR
  OMP_HIGH_COMPLEXITY_THRESHOLD* {.intdefine.} = 1000

# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# https://github.com/pytorch/pytorch/pull/5584
# and https://github.com/zy97140/omp-benchmark-for-pytorch
const
  OMP_BASE_THRESHOLD {.intdefine.} = 80_000
  OMP_LOW_COMPLEXITY_THRESHOLD {.intdefine.} = OMP_BASE_THRESHOLD
  OMP_MID_COMPLEXITY_THRESHOLD {.intdefine.} = OMP_BASE_THRESHOLD div 8
  OMP_HIGH_COMPLEXITY_THRESHOLD {.intdefine.} = OMP_BASE_THRESHOLD div 80

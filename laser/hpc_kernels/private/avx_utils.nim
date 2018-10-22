# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../simd, ./sse3_helpers

template lo*(vec: m256): m128 =
  ## Extract the low part of packed 8x float32 vector
  mm256_castps256_ps128(vec)

template hi*(vec: m256): m128 =
  ## Extract the low part of packed 8x float32 vector
  mm256_extractf128_ps(vec, 1)

func sum_ps_avx*(vec: m256): float32 {.inline.}=
  ## Sum packed 8x float32
  mm_add_ps(vec.lo, vec.hi).sum_ps_sse3()

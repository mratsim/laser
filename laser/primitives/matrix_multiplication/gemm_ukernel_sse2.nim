# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

template float64x2_muladd_unfused(a, b, c: m128d): m128d =
  mm_add_pd(mm_mul_pd(a, b), c)

ukernel_generator(
      x86_SSE2,
      typ = float64,
      vectype = m128d,
      nb_scalars = 2,
      simd_setZero = mm_setzero_pd,
      simd_broadcast_value = mm_set1_pd,
      simd_load_aligned = mm_load_pd,
      simd_load_unaligned = mm_loadu_pd,
      simd_fma = float64x2_muladd_unfused,
      simd_store_unaligned = mm_storeu_pd,
      simd_mul = mm_mul_pd,
      simd_add = mm_add_pd
    )

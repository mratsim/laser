# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../simd

func sum_ps_sse3*(vec: m128): float32 {.inline.}=
  ## Sum packed 4xfloat32
  let shuf = mm_movehdup_ps(vec)
  let sums = mm_add_ps(vec, shuf)
  let shuf2 = mm_movehl_ps(sums, sums)
  result = mm_add_ss(sums, shuf2).mm_cvtss_f32

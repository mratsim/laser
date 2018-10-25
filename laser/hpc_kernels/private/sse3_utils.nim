# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../simd

template m128_reduction(op_name, scalar_op, vector_op: untyped) =
  func op_name*(vec: m128): m128 {.inline.}=
    ## Reduce packed packed 4xfloat32
    let shuf = mm_movehdup_ps(vec)
    let sums = vector_op(vec, shuf)
    let shuf2 = mm_movehl_ps(sums, sums)
    result = scalar_op(sums, shuf2) # .mm_cvtss_f32

m128_reduction(sum_ps_sse3, mm_add_ss, mm_add_ps)
m128_reduction(max_ps_sse3, mm_max_ss, mm_max_ps)
m128_reduction(min_ps_sse3, mm_min_ss, mm_min_ps)

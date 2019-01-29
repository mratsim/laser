# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# See:
#  - exp_and_log_optimisation_resources.md
#  - https://github.com/herumi/fmath
#
#  - http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm
#  - http://soc.knu.ac.kr/video_lectures/15_4.pdf
#  - http://arith.cs.ucla.edu/publications/LogExp-TC04.pdf
#    - Range reduction + LookUp Table approximation
#
#  - https://nic.schraudolph.org/pubs/Schraudolph99.pdf
#    - Fast exp approximation using IEEE754 logarithmic nature

import
  math,
  ./exp_log_common,
  ../../simd

# ############################################################
#
#                     Float32 Exponentiation
#
# ############################################################

func fast_clamp(x: m128, lo, hi: static float32): m128 {.inline, noInit.} =

  # This is slower
  # result = mm_min_ps(x, hi.mm_set1_ps)
  # result = mm_max_ps(result, lo.mm_set1_ps)

  # This is faster
  # --------------------------
  const Lo32Mask = (not 0'i32) shr 1 # 2^31 - 1 - 0b0111...1111
  
  let # We could skip those but min/max are slow and there is a carried dependency that limits throughput
    limit = mm_and_si128(x.mm_castps_si128, Lo32Mask.mm_set1_epi32)
    over = mm_cmpgt_epi32(limit, mm_set1_epi32(static(hi.uint32))).mm_movemask_epi8
  
  if over != 0:
    result = mm_min_ps(x, hi.mm_set1_ps)
    result = mm_max_ps(result, lo.mm_set1_ps)
  else:
    result = x

template sse2_gather_explut_cvtps(t0: untyped, v: m128i) = 
  ## Gather from ExpLUT according to v
  ## Gather result (int32) is casted to packed float32
  # A template or inline proc that returns
  # `t0` will slow down compute by 25%

  let
    v0 = mm_cvtsi128_si32(v)
    # mm_extract_epi32 only in SSE4.1, we can work around with epi16
    v1 = mm_extract_epi16(v, 2)
    v2 = mm_extract_epi16(v, 4)
    v3 = mm_extract_epi16(v, 6)
  
  # mm_insert_epi32 only in SSE4.1, we cannot work around with epi16
  var t0{.inject.} = mm_castsi128_ps(mm_set1_epi32(ExpLUT[v0]))
  var t1 = mm_castsi128_ps(mm_set1_epi32(ExpLUT[v1]))
  let t2 = mm_castsi128_ps(mm_set1_epi32(ExpLUT[v2]))
  let t3 = mm_castsi128_ps(mm_set1_epi32(ExpLUT[v3]))

  t1 = mm_movelh_ps(t1, t3)
  t1 = mm_castsi128_ps(mm_slli_epi64(mm_castps_si128(t1), 32))

  t0 = mm_movelh_ps(t0, t2)
  t0 = mm_castsi128_ps(mm_srli_epi64(mm_castps_si128(t0), 32))
  t0 = mm_or_ps(t0, t1)

proc exp*(x: m128): m128 {.inline, noInit.} =
  let clamped = x.fast_clamp(ExpMin.float32, ExpMax.float32)

  let r = mm_cvtps_epi32(mm_mul_ps(clamped, mm_set1_ps(ExpA)))
  var t = mm_sub_ps(clamped, mm_mul_ps(mm_cvtepi32_ps(r), mm_set1_ps(ExpB)))
  t = mm_add_ps(t, mm_set1_ps(1'f32))

  var v = mm_and_si128(r, mm_set1_epi32(ExpBitsMask - 1))
  var u = mm_add_epi32(r, mm_set1_epi32(127 shl ExpBits))

  when false: # AVX2 only
    let ti = mm_i32gather_epi32(ExpLUT[0].unsafeAddr, v, 4)
    var t0 = mm_castsi128_ps(ti)
  else:
    sse2_gather_explut_cvtps(t0, v)

  u = mm_srli_epi32(u, ExpBits)
  u = mm_slli_epi32(u, MantissaBits)

  t0 = mm_or_ps(t0, mm_castsi128_ps(u))
  result = mm_mul_ps(t, t0)

when isMainModule:

  let a = mm_set1_ps(0.5'f32)
  let exp_a = exp(a)

  var scalar: array[4, float32]
  mm_store_ps(scalar[0].addr,exp_a)

  echo scalar

  echo exp(0.5'f32)

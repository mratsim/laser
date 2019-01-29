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
  ../../simd, ../../simd_extras

# ############################################################
#
#                     Float32 Exponentiation
#
# ############################################################

func fast_clamp(x: m512, lo, hi: static float32): m512 {.inline.} =
  const Lo32Mask = (not 0'i32) shr 1 # 2^31 - 1 - 0b0111...1111

  let # We could skip those but min/max are slow and there is a carried dependency that limits throughput
    limit = mm512_and_si512(x.mm512_castps_si512, Lo32Mask.mm512_set1_epi32)
    over = laser_mm512_cmpgt_epi32(limit, mm512_set1_epi32(static(hi.uint32))).laser_mm512_movemask_epi8

  if over != 0:
    result = mm512_min_ps(x, hi.mm512_set1_ps)
    result = mm512_max_ps(result, lo.mm512_set1_ps)
  else:
    result = x

proc exp*(x: m512): m512 {.inline.}=
  let clamped = x.fast_clamp(ExpMin.float32, ExpMax.float32)

  let r = mm512_cvtps_epi32(mm512_mul_ps(clamped, mm512_set1_ps(ExpA)))
  var t = mm512_sub_ps(clamped, mm512_mul_ps(mm512_cvtepi32_ps(r), mm512_set1_ps(ExpB)))
  t = mm512_add_ps(t, mm512_set1_ps(1'f32))

  var v = mm512_and_si512(r, mm512_set1_epi32(ExpBitsMask - 1))
  var u = mm512_add_epi32(r, mm512_set1_epi32(127 shl ExpBits))

  u = mm512_srli_epi32(u, ExpBits)
  u = mm512_slli_epi32(u, MantissaBits)

  let ti = mm512_i32gather_epi32(v, ExpLUT[0].unsafeAddr, 4)
  var t0 = mm512_castsi512_ps(ti)
  t0 = mm512_or_ps(t0, mm512_castsi512_ps(u))
  result = mm512_mul_ps(t, t0)

when isMainModule:

  let a = mm512_set1_ps(0.5'f32)
  let exp_a = exp(a)

  var scalar: array[16, float32]
  mm512_store_ps(scalar[0].addr,exp_a)

  echo scalar

  echo exp(0.5'f32)


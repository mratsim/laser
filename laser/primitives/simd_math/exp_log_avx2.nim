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

func fast_clamp(x: m256, lo, hi: static float32): m256 {.inline, noInit.} =

  # This is slower
  # result = mm256_min_ps(x, hi.mm256_set1_ps)
  # result = mm256_max_ps(result, lo.mm256_set1_ps)

  # This is faster
  # --------------------------
  const Lo32Mask = (not 0'i32) shr 1 # 2^31 - 1 - 0b0111...1111

  let # We could skip those but min/max are slow and there is a carried dependency that limits throughput
    limit = mm256_and_si256(x.mm256_castps_si256, Lo32Mask.mm256_set1_epi32)
    over = mm256_cmpgt_epi32(limit, mm256_set1_epi32(static(hi.uint32))).mm256_movemask_epi8

  if over != 0:
    result = mm256_min_ps(x, hi.mm256_set1_ps)
    result = mm256_max_ps(result, lo.mm256_set1_ps)
  else:
    result = x

proc exp*(x: m256): m256 {.inline, noInit.} =
  let clamped = x.fast_clamp(ExpMin.float32, ExpMax.float32)

  let r = mm256_cvtps_epi32(mm256_mul_ps(clamped, mm256_set1_ps(ExpA)))
  var t = mm256_sub_ps(clamped, mm256_mul_ps(mm256_cvtepi32_ps(r), mm256_set1_ps(ExpB)))
  t = mm256_add_ps(t, mm256_set1_ps(1'f32))

  var v = mm256_and_si256(r, mm256_set1_epi32(ExpBitsMask - 1))
  var u = mm256_add_epi32(r, mm256_set1_epi32(127 shl ExpBits))

  u = mm256_srli_epi32(u, ExpBits)
  u = mm256_slli_epi32(u, MantissaBits)

  let ti = mm256_i32gather_epi32(ExpLUT[0].unsafeAddr, v, 4)
  var t0 = mm256_castsi256_ps(ti)
  t0 = mm256_or_ps(t0, mm256_castsi256_ps(u))
  result = mm256_mul_ps(t, t0)

when isMainModule:

  let a = mm256_set1_ps(0.5'f32)
  let exp_a = exp(a)

  var scalar: array[8, float32]
  mm256_store_ps(scalar[0].addr,exp_a)

  echo scalar

  echo exp(0.5'f32)

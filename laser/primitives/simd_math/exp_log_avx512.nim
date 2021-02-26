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

func fast_clamp(x: m512, lo, hi: static float32): m512 {.inline, noInit.} =

  # This is slower
  # result = mm512_min_ps(x, hi.mm512_set1_ps)
  # result = mm512_max_ps(result, lo.mm512_set1_ps)

  # This is faster
  # --------------------------
  const Lo32Mask = (not 0'i32) shr 1 # 2^31 - 1 - 0b0111...1111

  let # We could skip those but min/max are slow and there is a carried dependency that limits throughput
    limit = mm512_and_si512(x.mm512_castps_si512, Lo32Mask.mm512_set1_epi32)
    over = laser_mm512_cmpgt_epi32(limit, mm512_set1_epi32(static(hi.uint32))).laser_mm512_movemask_epi8

  if over != 0:
    result = mm512_min_ps(x, hi.mm512_set1_ps)
    result = mm512_max_ps(result, lo.mm512_set1_ps)
  else:
    result = x

proc exp*(x: m512): m512 {.inline, noInit.}=
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

######################################################
## Bench on i9-9980XE Skylake-X - serial implementation
## OC @ 4.1 GHz, AVX 3.8 GHz, AVX512 @ 3.6 Ghz
# Warmup: 0.9069 s, result 224 (displayed to avoid compiler optimizing warmup away)

# A - tensor shape: [5000000]
# Required number of operations:     5.000 millions
# Required bytes:                   20.000 MB
# Arithmetic intensity:              0.250 FLOP/byte
# Theoretical peak single-core:     86.400 GFLOP/s
# Theoretical peak multi:          172.800 GFLOP/s
# a[0]: -9.999997138977051

# Baseline <math.h>
# Collected 300 samples in 5.021 seconds
# Average time: 16.081 ms
# Stddev  time: 0.047 ms
# Min     time: 15.618 ms
# Max     time: 16.124 ms
# Perf:         0.311 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540005829767324e-05

# SSE mathfun
# Collected 300 samples in 1.922 seconds
# Average time: 5.752 ms
# Stddev  time: 0.014 ms
# Min     time: 5.716 ms
# Max     time: 5.874 ms
# Perf:         0.869 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540006193565205e-05

# SSE fast_exp_sse (low order polynomial)
# Collected 300 samples in 1.070 seconds
# Average time: 2.909 ms
# Stddev  time: 0.013 ms
# Min     time: 2.883 ms
# Max     time: 2.996 ms
# Perf:         1.719 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.545032061287202e-05

# AVX2 fmath
# Collected 300 samples in 0.969 seconds
# Average time: 2.567 ms
# Stddev  time: 0.024 ms
# Min     time: 2.512 ms
# Max     time: 2.697 ms
# Perf:         1.948 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540006193565205e-05

# AVX2 FMA Minimax
# Collected 300 samples in 0.948 seconds
# Average time: 2.500 ms
# Stddev  time: 0.036 ms
# Min     time: 2.428 ms
# Max     time: 2.624 ms
# Perf:         2.000 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.539992369245738e-05

# AVX2 mathfun
# Collected 300 samples in 1.182 seconds
# Average time: 3.280 ms
# Stddev  time: 0.023 ms
# Min     time: 3.232 ms
# Max     time: 3.390 ms
# Perf:         1.524 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540006193565205e-05

# AVX+FMA Schraudolph-approx
# Collected 300 samples in 0.766 seconds
# Average time: 1.892 ms
# Stddev  time: 0.065 ms
# Min     time: 1.834 ms
# Max     time: 2.307 ms
# Perf:         2.643 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.625692963600159e-05

# Bench SIMD Math Prims
# Collected 300 samples in 4.338 seconds
# Average time: 13.804 ms
# Stddev  time: 0.232 ms
# Min     time: 13.573 ms
# Max     time: 14.121 ms
# Perf:         0.362 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.539986548479646e-05

# AVX2 Prod implementation
# Collected 300 samples in 0.941 seconds
# Average time: 2.475 ms
# Stddev  time: 0.027 ms
# Min     time: 2.425 ms
# Max     time: 2.622 ms
# Perf:         2.021 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540006193565205e-05

# AVX512 Prod implementation
# Collected 300 samples in 0.865 seconds
# Average time: 2.223 ms
# Stddev  time: 0.025 ms
# Min     time: 2.169 ms
# Max     time: 2.317 ms
# Perf:         2.249 GEXPOP/s

# Display output[0] to make sure it's not optimized away
# 4.540006193565205e-05

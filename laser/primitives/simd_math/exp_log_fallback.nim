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

import ./exp_log_common

# ############################################################
#
#          Float32 AVX2 vectorized Exponentiation
#
# ############################################################

proc exp(x: float32): float32 =

  static: assert cpuEndian == littleEndian, "Exp has not been tested on big-endian architectures"

  # TODO: clamp?

  let r = int32(x * ExpA) # ->f32->i32->f32 Rounding is important
  let t = x - r.float32 * ExpB + 1

  let v = r and (ExpBitsMask - 1)
  var u = r + (127 shl ExpBits)
  u = u shr ExpBits
  u = u shl MantissaBits

  let ti = ExpLut[v] or u
  result = t * cast[float32](ti)

when isMainModule:
  import math
  
  echo math.exp(0.5'f32)
  echo exp(0.5'f32)

  echo math.exp(-0.5'f32)
  echo exp(-0.5'f32)

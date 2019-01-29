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

from math import ln, pow

# ############################################################
#
#                     Float32 Exponentiation
#
# ############################################################

const
  ExpBits* = 10'i32
  ExpBitsMask* = 1'i32 shl ExpBits

  MantissaBits* = 23'i32
  MantissaBitsMask* = 1'i32 shl 23

  ln2 = ln(2'f32)

  ExpMax* = 88'i32
  ExpMin* = -88'i32
  ExpA* = ExpBitsMask.float32 / ln2
  ExpB* = ln2 / ExpBitsMask.float32

func initExpLUT(): array[ExpBitsMask, int32] =
  for i, val in result.mpairs:
    let y = pow(2'f32, i.float32 / ExpBitsMask.float32)
    val = cast[int32](y) and (MantissaBitsMask - 1)

# We need ExpLUT in the BSS so that we can take it's address so it can't be const
let ExpLUT* = initExpLUT()
# TODO debug multiplication with negative numbers

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../simd, math

# ############################################################
#
#                 AVX2: shuffling control
#
# ############################################################

func avx2_shufControl*(a0, a1, a2, a3, a4, a5, a6, a7: static uint8): cint =
  ## Human readable control for AVX2 _mm256_shuffle_epi32
  static:
    assert a0 + 4 == a4
    assert a1 + 4 == a5
    assert a2 + 4 == a6
    assert a3 + 4 == a7

  result = cint(a0)
  result = result or cint(a1 shl 2)
  result = result or cint(a2 shl 4)
  result = result or cint(a3 shl 6)

# ############################################################
#
#                              Tests
#
# ############################################################

when isMainModule:
  block:
    var scalars = [int32 1, 2, 3, 4, 5, 6, 7, 8]
    let vec = mm256_loadu_si256(scalars[0].addr)

    var dst: array[8, int32]

    block:
      const ctl = avx2_shufControl(0, 1, 2, 3, 4, 5, 6, 7)
      let shuffled = vec.mm256_shuffle_epi32(ctl)
      dst[0].addr.mm256_storeu_si256(shuffled)

      doAssert dst == [int32 1, 2, 3, 4, 5, 6, 7, 8]

    block:
      const ctl = avx2_shufControl(0, 1, 3, 2, 4, 5, 7, 6)
      let shuffled = vec.mm256_shuffle_epi32(ctl)
      dst[0].addr.mm256_storeu_si256(shuffled)

      doAssert dst == [int32 1, 2, 4, 3, 5, 6, 8, 7]

    block:
      const ctl = avx2_shufControl(0, 0, 0, 0, 4, 4, 4, 4)
      let shuffled = vec.mm256_shuffle_epi32(ctl)
      dst[0].addr.mm256_storeu_si256(shuffled)

      doAssert dst == [int32 1, 1, 1, 1, 5, 5, 5, 5]

    block:
      const ctl = avx2_shufControl(1, 1, 3, 3, 5, 5, 7, 7)
      let shuffled = vec.mm256_shuffle_epi32(ctl)
      dst[0].addr.mm256_storeu_si256(shuffled)

      doAssert dst == [int32 2, 2, 4, 4, 6, 6, 8, 8]

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../simd

# ############################################################
#
#                 SSE2: shuffling control
#
# ############################################################

func sse2_shufControl*(a0, a1, a2, a3: static uint8): cint =
  ## Human readable control for SSE2 _mm_shuffle_epi32
  result = cint(a0)
  result = result or cint(a1 shl 2)
  result = result or cint(a2 shl 4)
  result = result or cint(a3 shl 6)

# ############################################################
#
#             SSE2: int32 multiplication fallback
#
# ############################################################

func int32x4_mul_sse2_fallback*(a, b: m128i): m128i {.inline.}=
  ## SSE2 fallback for int32 multiplication
  ## Use mm_mullo_epi32 if SSE4.1 is available

  const ctl = sse2_shufControl(1, 1, 3, 3)

  let
    a_1133 = mm_shuffle_epi32(a, 0xF5)         # [a1, a1, a3, a3]
    b_1133 = mm_shuffle_epi32(b, 0xF5)         # [b1, b1, b3, b3]
    mul_02 = mm_mul_epu32(a, b)               # [lo: a0 * b0, hi, lo: a2 * b2, hi]
    mul_13 = mm_mul_epu32(a_1133, b_1133)     # [lo: a1 * b1, hi, lo: a3 * b3, hi]

    # Reminder, the mask applies in reverse order (big-endian)
    # Also: -1 = 0xFFFFFFFF for masking purposes
    mask = mm_set_epi32(e3 = 0, e2 = -1, e1 = 0, e0 = -1) # Discard pos 3 and pos 1, keep pos 2 and pos 0
    mulmasked_02 = mm_and_si128(mul_02, mask) # [lo: a0 * b0, 0, lo: a2 * b2, 0]

    # Reminder, logic left shift (`shl`) shifts to the right in little-endian
    # --> Interpret mul_13 as 2xint64 and shift right, discarding the overflowed high parts
    mulshifted_13 = mm_slli_epi64(mul_13, 32) # [0, a1 * b1, 0, a3 * b3]

  result = mm_or_si128(mulmasked_02, mulshifted_13)


# ############################################################
#
#                              Tests
#
# ############################################################


when isMainModule:
  block:
    var scalars = [int32 1, 2, 3, 4]
    let vec = mm_loadu_si128(scalars[0].addr)

    var dst: array[4, int32]

    block:
      const ctl = sse2_shufControl(0, 1, 2, 3)
      let shuffled = vec.mm_shuffle_epi32(ctl)
      dst[0].addr.mm_storeu_si128(shuffled)

      doAssert dst == [int32 1, 2, 3, 4]

    block:
      const ctl = sse2_shufControl(0, 1, 3, 2)
      let shuffled = vec.mm_shuffle_epi32(ctl)
      dst[0].addr.mm_storeu_si128(shuffled)

      doAssert dst == [int32 1, 2, 4, 3]

    block:
      const ctl = sse2_shufControl(0, 0, 0, 0)
      let shuffled = vec.mm_shuffle_epi32(ctl)
      dst[0].addr.mm_storeu_si128(shuffled)

      doAssert dst == [int32 1, 1, 1, 1]

    block:
      const ctl = sse2_shufControl(1, 1, 3, 3)
      let shuffled = vec.mm_shuffle_epi32(ctl)
      dst[0].addr.mm_storeu_si128(shuffled)

      doAssert dst == [int32 2, 2, 4, 4]

when isMainModule:
  block:
    let a = [int32 1, 2, 3, 4]
    let b = [int32 10, 100, 1000, 10000]
    let c = [int32 -1, -2, -3, -4]

    let va = mm_loadu_si128(a[0].unsafeaddr)
    let vb = mm_loadu_si128(b[0].unsafeaddr)
    let vc = mm_loadu_si128(c[0].unsafeaddr)

    var dst: array[4, int32]

    block:
      let mul = int32x4_mul_sse2_fallback(va, vb)
      dst[0].addr.mm_storeu_si128(mul)

      doAssert dst == [int32 10, 200, 3000, 40000]

    block:
      let mul = int32x4_mul_sse2_fallback(va, vc)
      dst[0].addr.mm_storeu_si128(mul)

      doAssert dst == [int32 -1, -4, -9, -16]

    block:
      let mul = int32x4_mul_sse2_fallback(vb, vc)
      dst[0].addr.mm_storeu_si128(mul)

      doAssert dst == [int32 -10, -200, -3000, -40000]

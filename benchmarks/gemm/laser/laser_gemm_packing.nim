# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./laser_gemm_matrix,
  ./laser_gemm_utils,
  ./laser_gemm_tiling

# ##############
#
# Packing A
#
# ##############

proc pack_mr_kc[T](buffer: ptr UncheckedArray[T], kc: int, ukernel: static MicroKernel[T], A: var MatrixView[T]) {.sideeffect.}=
  ## Packs micro-panel mr*kc for the bufA mc*kc (half-L2 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const MR = ukernel.mr

  for j in 0 ..< kc:
    for i in `||`(0, MR-1, "simd"): # TODO can use _mm_i32gather_ps on AVX
      buffer[i] = A[i, 0]
    buffer += MR
    A.incCol()

proc pack_A_mc_kc[T](buffer: ptr UncheckedArray[T], mc, kc: int, ukernel: static MicroKernel[T], A: var MatrixView[T]) =
  ## Packs panel mc*kc for the bufA (half-L2 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.

  # Copy panels of A into the mc*kc buffer
  # Copy uses a tile of dimension kc*MR
  const MR = ukernel.mr

  var mr = 0
  doWhile 0 < A.nrows:
    if A.nrows < MR:
      mr = A.nrows

    pack_mr_kc[MR](buffer, kc, A)
    buffer += kc*MR
    A.incCol(MR)

  # Process the tail
  if mr != 0:
    for j in 0 ..< kc:
      for i in 0 ..< mr:
        buffer[i] = A[i, 0]
      zeroMem(buffer, (MR-mr)*T.sizeof)
      buffer += MR
      A.incCol()

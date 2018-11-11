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

proc pack_mr_kc[T](
      buffer: ptr UncheckedArray[T],
      kc: int, ukernel: static MicroKernel,
      A: var MatrixView[T]) {.sideeffect.}=
  ## Packs micro-panel [mr, kc] for buffer Ã[mc, kc] (half-L2 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const MR = ukernel.mr

  for j in 0 ..< kc:
    for i in `||`(0, MR-1, "simd"): # TODO can use _mm_i32gather_ps on AVX
      buffer[i] = A[i, 0]
    buffer += MR
    A.incCol()

proc pack_A_mc_kc*[T](
      buffer: ptr UncheckedArray[T],
      kc: int, ukernel: static MicroKernel,
      A: var MatrixView[T]) =
  ## Packs panel [mc, kc] into buffer Ã (size ~half-L2 cache)
  ## Pads if needed
  ## Buffer uses Z-ordering so that the ukernel can access contiguous
  ## chunks of memory for the dot product

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.

  # Copy panels of A into the mc*kc buffer
  # Copy uses a tile of dimension kc*MR
  const MR = ukernel.mr

  var mr = MR
  doWhile 0 < A.nrows:
    if A.nrows < MR: # last iteration
      mr = A.nrows   # tail to process

    pack_mr_kc(buffer, kc, ukernel, A)
    buffer += kc*MR
    A.incCol(MR)

  # Process the tail
  if mr != 0:
    for j in 0 ..< kc:
      for i in 0 ..< mr:
        buffer[i] = A[i, 0]
      for i in mr ..< MR:
        # Padding
        buffer[i] = 0.T
      buffer += MR
      A.incCol()

proc pack_kc_nr[T](
      buffer: ptr UncheckedArray[T],
      kc: int, ukernel: static MicroKernel,
      B: var MatrixView[T]) {.sideeffect.}=
  ## Packs micro-panel kc*nr for ~B (half-L1 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const NR = ukernel.nr

  for i in 0 ..< kc:
    for j in `||`(0, NR-1, "simd"):
      buffer[j] = B[0, j]
    buffer += NR
    B.incRow()

proc pack_B_kc_nc*[T](
      buffer: ptr UncheckedArray[T],
      kc: int, ukernel: static MicroKernel,
      B: var MatrixView[T]) =
  ## Packs panel [kc, nc] for ~B (half-L1 cache)
  ## Pads if needed
  ## Buffer uses Z-ordering so that the ukernel can access contiguous
  ## chunks of memory for the dot product

  const NR = ukernel.nr

  var nr = NR
  doWhile 0 < B.ncols:
    if B.ncols < NR: # last iteration
      nr = B.ncols   # tail to process

    pack_kc_nr(buffer, kc, ukernel, B)
    buffer += kc*NR
    B.incCol(NR)

  # Process the tail
  if nr != 0:
    for i in 0 ..< kc:
      for j in 0 ..< nr:
        buffer[j] = B[0, j]
      for j in nr ..< NR:
        # Padding
        buffer[j] = 0.T
      buffer += NR
      B.incRow()

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Due to issue with "static MicroKernel" as parameter
# as of 0.19.9 we pass it as a generic param
#   - 1. undeclared identifier mr/nr, when accessing ukernel
#   - 2. object constructor needs and object type when workaround first issue with macro

import
  ./laser_gemm_matrix,
  ./laser_gemm_utils,
  ./laser_gemm_tiling

# ##############
#
# Packing A
#
# ##############

proc pack_mr_kc[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      A: var MatrixView[T]) {.sideeffect.}=
  ## Packs micro-panel [mr, kc] for buffer Ã[mc, kc] (half-L2 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const MR = ukernel.extract_mr

  assert A.ncols == kc
  while true:
    for i in `||`(0, MR-1, "simd"): # TODO can use _mm_i32gather_ps on AVX
      buffer[i] = A[i, 0]
    buffer += MR
    if 0 < A.ncols:
      break
    A.incCol()

proc pack_A_mc_kc*[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      A: var MatrixView[T]) =
  ## Packs panel [mc, kc] into buffer Ã (size ~half-L2 cache)
  ## Pads if needed
  ## Buffer uses Z-ordering so that the ukernel can access contiguous
  ## chunks of memory for the dot product

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.

  # Copy panels of A into the mc*kc buffer
  # Copy uses a tile of dimension mr*kc
  const MR = ukernel.extract_mr

  while MR <= A.nrows:
    pack_mr_kc[T, ukernel](buffer, kc, A)
    buffer += kc*MR
    A.incRow(MR)

  # Process the tail
  if A.nrows != 0:
    let mr = A.nrows
    for j in 0 ..< kc:
      for i in 0 ..< mr:
        buffer[i] = A[i, 0]
      for i in mr ..< MR:
        # Padding
        buffer[i] = 0.T
      buffer += MR
      A.incCol()

proc pack_kc_nr[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      B: var MatrixView[T]) {.sideeffect.}=
  ## Packs micro-panel kc*nr for ~B (half-L1 cache)

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const NR = ukernel.extract_nr

  assert B.nrows == kc
  while true:
    for j in `||`(0, NR-1, "simd"):
      buffer[j] = B[0, j]
    buffer += NR
    if 0 < B.nrows:
      break
    B.incRow()

proc pack_B_kc_nc*[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      B: var MatrixView[T]) =
  ## Packs panel [kc, nc] for ~B (half-L1 cache)
  ## Pads if needed
  ## Buffer uses Z-ordering so that the ukernel can access contiguous
  ## chunks of memory for the dot product

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.

  # Copy panels of B into the kc*nc buffer
  # Copy uses a tile of dimension kc*nr
  const NR = ukernel.extract_nr

  while NR <= B.ncols:
    pack_kc_nr[T, ukernel](buffer, kc, B)
    buffer += kc*NR
    B.incCol(NR)

  # Process the tail
  if B.ncols != 0:
    let nr = B.ncols
    for i in 0 ..< kc:
      for j in 0 ..< nr:
        buffer[j] = B[0, j]
      for j in nr ..< NR:
        # Padding
        buffer[j] = 0.T
      buffer += NR
      B.incRow()

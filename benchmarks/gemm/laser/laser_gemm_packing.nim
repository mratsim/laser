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

# TODO - part of align_unroller
func round_step_down(x: Natural, step: static Natural): int {.inline.} =
  ## Round the input to the previous multiple of "step"
  when (step and (step - 1)) == 0:
    # Step is a power of 2. (If compiler cannot prove that x>0 it does not make the optim)
    result = x and not(step - 1)
  else:
    result = x - x mod step

# ##############
#
# Packing A
#
# ##############

proc pack_A_mc_kc*[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      A: MatrixView[T]) =
  ## Packs panel [mc, kc] into buffer Ã (size ~half-L2 cache)
  ## Pads if needed
  ## Buffer uses Z-ordering so that the ukernel can access contiguous
  ## chunks of memory for the dot product

  ## ⚠ Warning, the buffer pointer will be moved even though it's not a var.
  ##     Unfortunately if it's made into a var, Nim will use a hidden pointer
  ##     and it's the hidden pointer that will be moved :/.
  const MR = ukernel.extract_mr
  let mblock = A.nrows div MR
  let tail = A.nrows mod MR

  for i in 0 ..< mblock:
    let offset = i * MR
    for k in 0 ..< kc:
      for ii in 0 ..< MR:
        buffer[0] = A[ii + offset, k]
        buffer += 1

  if tail > 0:
    let offset = mblock * MR
    for k in 0 ..< kc:
      for i in 0 ..< A.nrows:
        buffer[i] = A[i + offset, k]
      for i in A.nrows ..< MR:
        buffer[i] = 0
      buffer += 1


# ##############
#
# Packing B
#
# ##############

proc pack_B_kc_nc*[T; ukernel: static MicroKernel](
      buffer: ptr UncheckedArray[T],
      kc: int,
      B: MatrixView[T]) =
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
  let nblock = B.ncols div NR
  let tail = B.ncols mod NR
  let packedB{.restrict.} = buffer

  assert kc == B.nrows # TODO remove kc param

  # TODO: packing B can be parallel. N/NC is large.
  for j in 0 ..< nblock:
    let offset = j * NR
    for k in 0 ..< kc:
      for jj in 0 ..< NR:
        packedB[0] = B[k, jj+offset]
        packedB += 1

  if tail > 0:
    let offset = nblock * NR
    for k in 0 ..< kc:
      for j in 0 ..< B.ncols:
        packedB[j] = B[k, j+offset]
      for j in B.ncols ..< NR:
        packedB[j] = 0
      packedB += 1

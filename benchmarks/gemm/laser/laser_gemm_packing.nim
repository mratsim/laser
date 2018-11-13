# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Due to issue with "static MicroKernel" as parameter
# as of 0.19.9 we pass it as a generic param
#   - 1. undeclared identifier mr/nr, when accessing ukernel
#   - 2. object constructor needs and object type when workaround first issue with macro

import
  ../../../laser/compiler_optim_hints,
  ./laser_gemm_matrix,
  ./laser_gemm_utils,
  ./laser_gemm_tiling

withCompilerOptimHints()

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
      tiles: Tile[T],
      mc, kc: int,
      A: MatrixView[T]) =
  ## Packs panel [kc, mc] into buffer Ã (size ~half-L2 cache)
  ## Pads if needed
  ## Note that A is of shape [M, K] so it is transposed.
  ##
  ## Concretely the outer dimension of packed matrices
  ## is k so that C[i, j] = A[i, k] * B[k, j]
  ## does not require strided access
  let buffer{.restrict.} = assume_aligned tiles.a
  const MR = ukernel.extract_mr()
  let unroll_stop = mc.round_step_down(MR)

  # 1. Pack mc/mr matrices of size kc*mr
  {.emit:"""
      #pragma omp parallel for
      for (int i = 0; i < `unroll_stop`; i+=`MR`)
        for (int k = 0; k < `kc`; k++)
          for (int ii = 0; ii < `MR`; ii++)
            `buffer`[i*`kc`+k*`MR`+ii] = `A`.buffer[(i+ii)*`A`.rowStride + k*`A`.colStride];
  """.}

  # block:
  #   for i in countup(0, unroll_stop-1, MR):
  #     for k in 0 ..< kc:
  #       for ii in 0 ..< MR:
  #         echo "bufA: ", i*kc+k*MR+ii
  #         buffer[i*kc+k*MR+ii] = A[i+ii, k]

  # 2. Process the tail
  let remainder = mc - unroll_stop
  if remainder > 0:
    let offBuf = buffer + kc*unroll_stop
    for k in 0 ..< kc:
      for i in 0 ..< remainder:
        offBuf[k*MR + i] = A[unroll_stop+i, k]
      for i in remainder ..< MR:
        # Pad with 0 if packing over the edge
        offBuf[k*MR + i] = 0.T

  # block:
  #   echo "[kc, mc]: [", kc, ", ", mc, "] = ", kc*mc
  #   var mA = newSeq[T](mc*kc)
  #   var bufA = newSeq[T](mc*kc)
  #   for i in 0 ..< mc*kc:
  #     mA[i] = A.buffer[i]
  #     bufA[i] = buffer[i]
  #   echo "A view: ", mA
  #   echo "A buffer: ", bufA
  #   echo "###############"

# ##############
#
# Packing B
#
# ##############

proc pack_B_kc_nc*[T; ukernel: static MicroKernel](
      tiles: Tile[T],
      kc, nc: int,
      B: MatrixView[T]) =
  ## Packs panel [kc, nc] for ~B (half-L1 cache)
  ## Pads if needed
  ##
  ## Concretely the outer dimension of packed matrices
  ## is k so that C[i, j] = A[i, k] * B[k, j]
  ## does not require strided access
  let buffer{.restrict.} = tiles.b # TODO align B for SIMD
  const NR = ukernel.extract_nr()
  let unroll_stop = nc.round_step_down(NR)

  # 1. Pack nc/nr matrices of size kc*nr
  {.emit:"""
      #pragma omp parallel for
      for (int j = 0; j < `unroll_stop`; j+=`NR`)
        for (int k = 0; k < `kc`; k++)
          // #pragma omp simd // NR is always of vector size - special case unit stride for SIMD
          for (int jj = 0; jj < `NR`; jj++)
            `buffer`[j*`kc`+k*`NR`+jj] = `B`.buffer[k*`B`.rowStride + (j+jj)*`B`.colStride];
  """.}

  # block:
  #   for j in countup(0, unroll_stop-1, NR):
  #     for k in 0 ..< kc:
  #       for jj in 0 ..< NR:
  #         echo "bufB: ", j*kc+k*NR+jj
  #         buffer[j*kc+k*NR+jj] = B[k, j+jj]

  # 2. Process the tail
  let remainder = nc - unroll_stop
  if remainder > 0:
    let offBuf = buffer + kc*unroll_stop
    for k in 0 ..< kc:
      for j in 0 ..< remainder:
        offBuf[k*NR + j] = B[k, unroll_stop+j]
      for j in remainder ..< NR:
        # Pad with 0 if packing over the edge
        offBuf[k*NR + j] = 0.T

  # block:
  #   echo "[kc, nc]: [", kc, ", ", nc, "] = ", kc*nc
  #   var mB = newSeq[T](kc*nc)
  #   var bufB = newSeq[T](kc*nc)
  #   for i in 0 ..< kc*nc:
  #     mB[i] = B.buffer[i]
  #     bufB[i] = buffer[i]
  #   echo "B view: ", mB
  #   echo "B buffer: ", bufB
  #   echo "###############"

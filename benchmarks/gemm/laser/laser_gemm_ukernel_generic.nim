# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Generic microkernel for matrix multiplication

import
  ./laser_gemm_tiling, ./laser_gemm_matrix,
  ../../../laser/[cpuinfo, compiler_optim_hints],
  macros

# TODO: vzeroupper for AVX version.
withCompilerOptimHints()

macro unroll_ukernel[mr, nr: static int, T](
      AB: ptr array[mr, array[nr, T]],
      A, B: ptr
    ): untyped =

  result = newStmtList()
  for i in 0 .. mr - 1:
    for j in 0 .. nr - 1:
      result.add quote do:
        `AB`[i][j] += `A`[`i`] * `B`[`j`]

func gemm_ukernel_generic*[mr, nr: static int, T](
      AB: var array[mr, array[nr, T]],
      tiles: Tile[T],
      ukernel: static MicroKernel
    ) =

  # The inner loop must be the SIMD vector size
  # Also since we work mostly with row-major tensors
  # it's better if it's the nr dimension to avoid transposing AB
  static: assert nr == ukernel.vec_size

  let pAB{.restrict.} = assume_aligned AB[0][0].addr
  var  A {.restrict.} = assume_aligned tiles.a # [mr, kc]
  var  B {.restrict.} = tiles.b                # [kc, nr]

  for k in 0 ..< tiles.kc:
    prefetch(B + nr    , Read) # TODO: temporal locality?
    prefetch(B + nr + 8, Read) # AVX SIMD with is 8 and we issue 2 of them

    # ⚠ Warning AB result is transposed
    unroll_ukernel(pAB, A, B)   # 95% of GEMM time is here

    A += mr
    B += nr

# ########################
# Epilogue
#
# Cases
# 1. C *=   β, starting default
# 2. C  =  AB, if β = 0 and α = 1
# 3. C  = αAB, if β = 0 and α = 1
# 4. C +=  AB, if α = 1
# 5. C += αAB, if α = 1
#
# TODO: Fused operations like relu/sigmoid/tanh
#       should be done here as well

func gemm_ukernel_epilogue*[MR, NR: static int, T](
      alpha: T, AB: array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T]
    ) =

  let pAB{.restrict.} = assume_aligned AB[0][0].addr

  if beta == 0.T:
    if alpha == 1.T:                   # C = AB
      for i in 0 ..< MR:
        for j in `||`(0, NR-1, "simd"):
          vC[i, j] = pAB[i][j]
    else:                              # C = αAB
      for i in 0 ..< MR:
        for j in `||`(0, NR-1, "simd"):
          vC[i, j] = alpha * pAB[i][j]
  else:                                # C *= β
    for i in 0 ..< MR:
      for j in `||`(0, NR-1, "simd"):
        vC[i, j] *= beta

    if alpha == 1.T:                   # C += AB
      for i in 0 ..< MR:
        for j in `||`(0, NR-1, "simd"):
          vC[i, j] += pAB[i][j]
    else:                              # C += αAB
      for i in 0 ..< MR:
        for j in `||`(0, NR-1, "simd"):
          vC[i, j] += alpha * pAB[i][j]

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

func gemm_ukernel_edge_epilogue*[MR, NR: static int, T](
      alpha: T, AB: array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T],
      mr, nr: int # Tail to process
    ) =

  let pAB{.restrict.} = assume_aligned AB[0][0].addr

  if beta == 0.T:
    if alpha == 1.T:                   # C = AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = pAB[i][j]
    else:                              # C = αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = alpha * pAB[i][j]
  else:                                # C *= β
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        vC[i, j] *= beta

    if alpha == 1.T:                   # C += AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += pAB[i][j]
    else:                              # C += αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += alpha * pAB[i][j]

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

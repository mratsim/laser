# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Generic microkernel for matrix multiplication

import
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils,
  ../../../laser/[cpuinfo, compiler_optim_hints],
  macros

# TODO: vzeroupper for AVX version.
withCompilerOptimHints()

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


proc gebb_ukernel_epilogue*[MR, NR: static int, T](
      alpha: T, AB: array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T]
    ) =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB.unsafeAddr)

  # Beta always = 1 after the first pass on the current C micro-tile
  # so even if beta = 1 we need to accumulate with `+=`
  if beta == 0.T:
    for i in 0 ..< MR:
      for j in `||`(0, NR-1, "simd"):
        vC[i, j] = 0.T
  elif beta != 1.T:                  # C *= β
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

func gebb_ukernel_edge_epilogue*[MR, NR: static int, T](
      alpha: T, AB: array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T],
      mr, nr: int # Tail to process
    ) =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB.unsafeAddr)

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

macro unroll_ukernel[MR, NR: static int, T](
      AB: array[MR, array[NR, T]],
      A, B: ptr
    ): untyped =

  result = newStmtList()
  for i in 0 .. MR - 1:
    for j in 0 .. NR - 1:
      result.add quote do:
        `AB`[`i`][`j`] += `A`[`i`] * `B`[`j`]

template ukernel_impl(){.dirty.} =
  const
    MR = ukernel.extract_mr()
    NR = ukernel.extract_nr()
    vec_size = ukernel.extract_vecsize
    simd = ukernel.extract_cpu_simd

  var AB{.align_variable.}: array[MR, array[NR, T]]
  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  for k in 0 ..< kc:
    # TODO prefetch
    unroll_ukernel(AB, A, B)   # 95% of GEMM time is here

    A += MR
    B += NR

proc gebb_ukernel_generic*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_impl()
  gebb_ukernel_epilogue(alpha, AB, beta, vC)

proc gebb_ukernel_edge*[T; ukernel: static MicroKernel](
      mr, nr, kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_impl()
  gebb_ukernel_edge_epilogue(alpha, AB, beta, vC, mr, nr)

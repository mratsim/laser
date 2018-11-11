# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Generic microkernel for matrix multiplication

import
  ./laser_gemm_tiling, ./laser_gemm_matrix,
  ../../../laser/[cpuinfo, compiler_optim_hints],
  macros

macro unroll_ukernel[mr, nr: static int, T](
      AB: array[nr, array[mr, T]],
      A, B: ptr
    ): untyped =

  result = newStmtList()
  for j in 0 .. nr - 1:
    for i in 0 .. mr - 1:
      result.add quote do:
        # Note the transposition for contiguous AB access
        # mr is a multiple of the SIMD size
        `AB`[j][i] += `A`[`i`] * `B`[`j`]

func gemm_ukernel_generic*[T](
      alpha, beta: T,
      vC: MatrixView[T],
      tiles: Tile[T],
      ukernel: static MicroKernel
    ) =

  const mr = ukernel.mr
  const nr = ukernel.nr

  withCompilerOptimHints()
  var AB{.align_variable.}: array[mr, array[nr, T]]
  var A {.restrict.} = assume_aligned tiles.a # [mr, kc]
  var B {.restrict.} = tiles.b                # [kc, nr]

  for k in 0 ..< tiles.kc:
    prefetch(B + nr    , Read) # TODO: temporal locality?
    prefetch(B + nr + 8, Read) # AVX SIMD with is 8 and we issue 2 of them

    # ⚠ Warning AB result is transposed
    unroll_ukernel(AB, A, B)   # 95% of GEMM time is here

    A += mr
    B += nr

  # ########################
  # Epilogue
  #
  # Cases
  # 1. C *=   β, starting default
  # 2. C  =  AB, if Beta = 0 and alpha = 1
  # 3. C  = αAB, if Beta = 0 and alpha = 1
  # 4. C +=  AB, if alpha = 1
  # 5. C += αAB, if alpha = 1
  #
  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

  # We need to untranspose here
  if beta == 0.T:
    if alpha == 1.T:                   # C = AB
      for j in 0 ..< nr:
        for i in `||`(0, mr-1, "simd"):
          vC[i, j] = AB[j][i]
    else:                              # C = αAB
      for j in 0 ..< nr:
        for i in `||`(0, mr-1, "simd"):
          vC[i, j] = alpha * AB[j][i]
  else:                                # C *=   β
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        # We assume that C is row Major
        # or row-Major tilted
        vC[i, j] *= beta

    if alpha == 1.T:                   # C +=  AB
      for j in 0 ..< nr:
        for i in `||`(0, mr-1, "simd"):
          vC[i, j] += AB[j][i]
    else:                              # C += αAB
      for j in 0 ..< nr:
        for i in `||`(0, mr-1, "simd"):
          vC[i, j] += alpha * AB[j][i]

  # TODO: arbitrary function like relu/tanh/sigmoid

func gemm_edge_epilogue*[mr, nr: static int, T](
        alpha, beta: T,
        vC: MatrixView,
        bufC: array[mr, array[nr, T]]
    ) =
  # To avoids overwriting memory over the edge of the matrix
  # C updates are placed first in a buffer of size [nr, mr]
  #
  # and then copied into matrix C edges (i.e. tile area < nr*mr)

  assert vC.nrows < mr and vc.ncols < nr
  

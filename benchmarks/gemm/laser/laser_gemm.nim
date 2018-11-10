# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../../laser/cpuinfo,
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils

proc gemm_impl[T](
      alpha: T, vA: MatrixView, vB: MatrixView,
      beta: T, vC: MatrixView,
      tiles: Tile[T],
      ukernel: static MicroKernel[T]
    )=
  # Loop concerns:
  #   - Which one to parallelize
  #   - Handling edge cases where tile does not divide the dimension
  #   - Reducing indexing overhead. For example Arraymancer fallback BLAS
  #     requires to multiply the iteration index by the dimension stride.
  #     We can increment the index by the stride directly instead.
  #     (But that makes edge cases harder)
  #   - Reducing register pressure. We stress registers a lot, we could

  var incB: int    # stride B, deal with edge cases
  if tiles.mc < vA.nrows:
    incB = tiles.kc

  # 1. for jc = 0,...,n−1 in steps of nc   # Often 1 iteration
  # 2.   for pc = 0,...,k−1 in steps of kc
  var kc = tiles.kc
  doWhile 0 < vA.ncols:
    if vA.ncols < kc: # last iteration
      kc = vA.ncols

    var pc_a{.restrict.} = vA.sliceCol(kc) # vA[0:M, 0:kc]
    let pc_b{.restrict.} = vB
    let pc_c{.restrict.} = vC

    # 3. for ic = 0,...,m−1 in steps of mc
    var mc = tiles.mc
    doWhile 0 < pc_a.nrows:
      if pc_a.nrows < mc: # last iteration
        mc = pc_a.nrows

    # increment stride

proc gemm_strided*[T: SomeNumber](
      M, N, K: int,
      alpha: T,
      A: ptr T,
      incRowA, incColA: int,
      B: ptr T,
      incRowB, incColB: int,
      beta: T,
      C: ptr T,
      incRowC, incColc: int) =

    let tiles = newTiles(M, N, K, T)
    # buffer A: mc*kc L2 cache
    # buffer B: kc*nc L3 cache
    # buffer C: mr*nr registers
    #
    # and kc*nr panel in L1 cache

    # TODO detect colMajor
    # TODO shortcut alpha = 0 or K = 0

    # Create a view to abstract deling with strides
    # and passing those in each proc
    var vA = A.toMatrixView(M, K, incRowA, incColA)
    var vB = B.toMatrixView(K, N, incRowB, incColB)
    var vC = C.toMatrixView(M, N, incRowC, incColC)

    # Dispatch - TODO, support for element-wise epilogue like relu or tanh
    if cpuinfo_has_x86_avx512f(): x86_ukernel(T, x86_AVX512)
    elif cpuinfo_has_x86_avx2(): x86_ukernel(T, x86_AVX2)
    elif cpuinfo_has_x86_avx(): x86_ukernel(T, x86_AVX)
    elif cpuinfo_has_x86_sse2(): x86_ukernel(T, x86_SSE2)
    elif cpuinfo_has_x86_sse(): x86_ukernel(T, x86_SSE)
    else: x86_ukernel(T, x86_Generic)


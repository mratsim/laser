# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../../laser/[cpuinfo, compiler_optim_hints],
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils,
  ./laser_gemm_packing, ./laser_gemm_ukernel_generic

# TODO: vzeroupper?

proc gemm_mkernel[T](
      alpha, beta: T,
      vC: MatrixView[T],
      tiles: Tile[T],
      ukernel: static MicroKernel
    ) =
  ## Macro kernel around a mr * nr tile of C

  # Since nr is small this the the good place to parallelize
  # See: Anatomy of High-Performance Many-Threaded Matrix Multiplication
  #      Smith et al
  #      - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf

  # ⚠ We need to ensure that loop variables and pointers
  # are private to each thread

  # Nim doesn't support arbitrary increment with OpenMP
  # So we store indexing/edge case data in tiles

  const mr = ukernel.mr

  # #####################################
  # 4. for jr = 0,...,nc−1 in steps of nr
  var tileC = vC
  var nr = ukernel.nr
  for jrb in 0||(tiles.jr_num_nr_tiles - 1):
    if jrb == tiles.jr_num_nr_tiles - 1: # last iteration
      nr = tileC.ncols

    # ###################################
    # 5. for ir = 0,...,m−1 in steps of mr
    doWhile 0 < tileC.nrows:
      # TODO save addr of next panel of A for prefetch
      # and if last iter, save addr of next panel of B

      if nr <= tileC.nrows and mr <= tileC.ncols:
        # General case
        gemm_ukernel_generic(
          alpha, beta, tileC, tiles, ukernel
        )
      else:
        # Matrix edges
        var bufC{.align_variable.}: array[mr, array[nr, T]]
        let pbufC = bufC[0][0].addr.toMatrixView(mr, nr, nr, 1)

        gemm_ukernel_generic(
          alpha, beta, pbufC, tiles, ukernel
        )


      tileC.incRow(nr)
    # ###################################
    tileC.incCol(mr)

proc gemm_impl[T](
      alpha: T, vA: MatrixView, vB: MatrixView,
      beta: T, vC: MatrixView,
      tiles: Tile[T],
      ukernel: static MicroKernel
    ) =
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

  # ##################################################################
  # 1. for jc = 0,...,n−1 in steps of nc   # not partitioned currently
  # ######################################
  # 2.   for pc = 0,...,k−1 in steps of kc
  var kc = tiles.kc
  doWhile 0 < vA.ncols:
    if vA.ncols < kc: # last iteration
      kc = vA.ncols

    var pc_a{.restrict.} = vA.sliceCol(kc) # vA[0:M, 0:kc]
    var pc_b{.restrict.} = vB
    var pc_c{.restrict.} = vC

    let bufB{.restrict.} = tile.b
    pack_B_kc_nc(bufB, kc, ukernel, pc_b)

    # ####################################
    # 3. for ic = 0,...,m−1 in steps of mc
    var mc = tiles.mc
    doWhile 0 < pc_a.nrows:
      if pc_a.nrows < mc: # last iteration
        mc = pc_a.nrows

      let bufA{.restrict.} = assume_aligned tile.a
      pack_A_mc_kc(bufA, kc, ukernel, pc_a)




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


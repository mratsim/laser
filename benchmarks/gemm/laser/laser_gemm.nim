# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../../laser/[cpuinfo, compiler_optim_hints],
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils,
  ./laser_gemm_ukernel_generic, ./laser_gemm_packing

withCompilerOptimHints()

# Terminology
#   - M, Matrix: Both dimension are large or unknown
#   - P, Panel: one of the dimension is small
#   - B, Block: both dimension are small
#
#   - GEMM: GEneralized Matrix-Matrix multiplication
#   - GEPP: GEneralized Panel-Panel multiplication
#   - GEBP: Generalized Block-Panel multiplication
#   ...

proc gebp_mkernel[T; ukernel: static MicroKernel](
      alpha, beta: T,
      mcncC: MatrixView[T],
      tiles: Tile[T]
    ) =
  ## Macro kernel, multiply:
  ##  - a block A[mc, kc] * panel B[kc, N]

  # Since nr is small this the the good place to parallelize
  # See: Anatomy of High-Performance Many-Threaded Matrix Multiplication
  #      Smith et al
  #      - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf

  # ⚠ We need to ensure that loop variables and pointers
  # are private to each thread

  # Nim doesn't support arbitrary increment with OpenMP
  # So we store indexing/edge case data in tiles

  const MR = ukernel.extract_mr
  const NR = ukernel.extract_nr

  # #####################################
  # 4. for jr = 0,...,nc−1 in steps of nr
  var nr = NR
  var jr_mcncC = mcncC
  for jrb in 0||(tiles.jr_num_nr_tiles - 1):
    if jrb == tiles.jr_num_nr_tiles - 1: # last iteration
      nr = jr_mcncC.ncols

    var ir_mcnrC = jr_mcncC.sliceCols(nr)          # C[ic:ic+mc, jc+jr:jc+jr+nr]
    # ###################################
    # 5. for ir = 0,...,m−1 in steps of mr
    var mr = MR
    doWhile 0 < ir_mcnrC.nrows:
      if ir_mcnrC.nrows < MR: # last iteration
        mr = ir_mcnrC.nrows

      var mrnrC = ir_mcnrC.sliceRows(mr)           # C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr]

      # TODO save addr of next panel of A for prefetch
      # and if last iter, save addr of next panel of B
      var AB{.align_variable.}: array[MR, array[NR, T]]

      gemm_ukernel_generic(AB, tiles)              # GEBB microkernel + epilogue
      if nr == NR and mr == MR:                    #   C[ic+ir:ic+ir+mr, jc+jr:jc+jr+nr] =
        # General case                             #    αA[ic+ir:ic+ir+mr, pc:pc+kc] *
        gemm_ukernel_epilogue(                     #     B[pc:pc+kc, jc+jr:jc+jr+nr] +
          alpha, AB, beta, mrnrC                   #    βC[ic:ic+mc, jc:jc+nc]
        )
      else:
        # Matrix edges
        gemm_ukernel_edge_epilogue(
          alpha, AB, beta, mrnrC,
          mr, nr
        )

      ir_mcnrC.incRow(mr)
    # ###################################
    jr_mcncC.incCol(nr)
  # #####################################

proc gemm_impl[T; ukernel: static MicroKernel](
      alpha: T, vA: MatrixView[T], vB: MatrixView[T],
      beta: T, vC: MatrixView[T],
      tiles: Tile[T]
    ) =
  # Loop concerns:
  #   - Which one to parallelize
  #   - Handling edge cases where tile does not divide the dimension
  #   - Reducing indexing overhead. For example Arraymancer fallback BLAS
  #     requires to multiply the iteration index by the dimension stride.
  #     We can increment the index by the stride directly instead.
  #     (But that makes edge cases harder)
  #   - Reducing register pressure. We stress registers a lot, we could

  # ##################################################################
  # 1. for jc = 0,...,n−1 in steps of nc
  # not partitioned currently nc = N
  var jc_kncB = vB                                  # B[0:K, jc:jc+nc]
  var jc_mncC = vC                                  # C[0:M, jc:jc+nc]
  # ######################################
  # 2.   for pc = 0,...,k−1 in steps of kc
  var kc = tiles.kc
  var pc_mkA = vA                                   # A[0:M, 0:K]
  doWhile 0 < pc_mkA.ncols:
    if pc_mkA.ncols < tiles.kc: # last iteration
      kc = pc_mkA.ncols

    var pc_kcncB = jc_kncB.sliceRows(kc)            # B[pc:pc+kc, jc:jc+nc]
    let bufB{.restrict.} = tiles.b                  # panel [kc, nc] (nc is large or unknown)
    pack_B_kc_nc[T, ukernel](bufB, kc, pc_kcncB)    # mutate bufB and pc_kcncB

    # ####################################
    # 3. for ic = 0,...,m−1 in steps of mc
    var mc = tiles.mc
    var ic_mkcA = pc_mkA.sliceCols(kc)              # A[0:M, pc:pc+kc]
    doWhile 0 < ic_mkcA.nrows:
      if ic_mkcA.nrows < tiles.mc: # last iteration
        mc = ic_mkcA.nrows
      let jr_mckcA = ic_mkcA.sliceRows(mc)          # A[ic:ic+mc, pc:pc+kc]
      let bufA{.restrict.} = assume_aligned tiles.a # block [mc, kc]
      pack_A_mc_kc[T, ukernel](bufA, kc, jr_mckcA)

      echo "mc * kc: ", mc, " * ", kc
      var v: seq[T]
      let vA2 = ic_mkcA.sliceRows(mc)
      for i in 0 ..< mc:
        for j in 0 ..< kc:
          v.add vA2[i, j]
      echo "cur view: ", v

      var s: seq[T]
      echo "A[0, 1]: ", vA[0, 1]
      for i in 0 ..< mc*kc:
        s.add tiles.a[i]
      echo "buffer: ", s

      let jr_mcncC = jc_mncC.sliceRows(mc)          # C[ic:ic+mc, jc:jc+nc]
      gebp_mkernel[T, ukernel](                     # GEBP macrokernel:
          alpha, beta, jr_mcncC,                    #   C[ic:ic+mc, jc:jc+nc] =
          tiles                                     #    αA[ic:ic+mc, pc:pc+kc] * B[pc:pc+kc, jc:jc+nc] +
        )                                           #    βC[ic:ic+mc, jc:jc+nc]

      jc_mncC.incRow(mc)
      ic_mkcA.incRow(mc)
    # ####################################
    jc_kncB.incRow(kc)
    pc_mkA.incCol(kc)
  # ######################################

proc gemm_strided*[T: SomeNumber](
      M, N, K: int,
      alpha: T,
      A: ptr T,
      rowStrideA, colStrideA: int,
      B: ptr T,
      rowStrideB, colStrideB: int,
      beta: T,
      C: ptr T,
      rowStrideC, colStrideC: int) =

    # TODO detect colMajor / transpose for contiguous iteration
    # TODO shortcut alpha = 0 or K = 0
    # TODO: custom epilogue fusion like relu/tanh/sigmoid
    # TODO: shortcut for small gemm

    # Create a view to abstract deling with strides
    # and passing those in each proc
    let vA = A.toMatrixView(M, K, rowStrideA, colStrideA)
    let vB = B.toMatrixView(K, N, rowStrideB, colStrideB)
    let vC = C.toMatrixView(M, N, rowStrideC, colStrideC)

    # Cache hierarchy:
    #   - block C: mr*nr registers
    #   - block B: kc*nr L1 cache
    #   - block A: mc*kc L2 cache
    #   - panel B: kc*nc L3 cache

    # Dispatch - TODO, support for element-wise epilogue like relu or tanh
    template dispatch(cpu_features: static CPUFeatureX86){.dirty.} =
      # const ukernel = cpu_features.x86_ukernel(T)
      const ukernel = MicroKernel(mr: 6, nr: 1)
      let tiles = ukernel.newTiles(T, M, N, K)
      gemm_impl[T, ukernel](
        alpha, vA, vB,
        beta, vC,
        tiles
      )
      return

    when defined(i386) or defined(amd64):
      when T is SomeFloat:
        if cpuinfo_has_x86_avx512f(): dispatch(x86_AVX512)
        elif cpuinfo_has_x86_avx():   dispatch(x86_AVX) # Handles AVX2 as well, only diff is that AVX2 can issue 2xFMA
        elif cpuinfo_has_x86_sse2():   dispatch(x86_SSE2)
        elif cpuinfo_has_x86_sse():   dispatch(x86_SSE)
      else: # Integers are taking advantage of wider registers later (in SSE2 and AVX2)
        if cpuinfo_has_x86_avx512f(): dispatch(x86_AVX512)
        elif cpuinfo_has_x86_avx2():   dispatch(x86_AVX2)
        elif cpuinfo_has_x86_sse2():   dispatch(x86_SSE2)
    dispatch(x86_Generic)

# ########################################################################################
when isMainModule:
  # Tests
  # block:
  #   let a = [[1.0, 2, 3],
  #            [1.0, 1, 1],
  #            [1.0, 1, 1]]

  #   let b = [[1.0, 1],
  #            [1.0, 1],
  #            [1.0, 1]]

  #   let ab = [[6.0, 6],
  #             [3.0, 3],
  #             [3.0, 3]]

  #   var res_ab: array[3, array[2, float]]
  #   gemm_strided(
  #     3, 2, 3,
  #     1.0,  a[0][0].unsafeAddr, 3, 1,
  #           b[0][0].unsafeAddr, 2, 1,
  #     0.0,  res_ab[0][0].addr,  2, 1
  #     )

  #   echo "expected: ", ab
  #   echo "result: ", res_ab

  #   # doAssert res_ab == ab

  block:
    let a = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    let b = [[1.0, 1],
             [1.0, 1],
             [1.0, 1]]

    let ab = [[ 6.0,  6],
              [15.0, 15],
              [24.0, 24]]

    var res_ab: array[3, array[2, float]]
    gemm_strided(
      3, 2, 3,
      1.0,  a[0][0].unsafeAddr, 3, 1,
            b[0][0].unsafeAddr, 2, 1,
      0.0,  res_ab[0][0].addr,  2, 1
      )

    echo "expected: ", ab
    echo "result: ", res_ab

    # doAssert res_ab == ab

  # block:
  #   let a = [[1.0,2,3],
  #            [4.0,5,6]]

  #   let b = [[7.0,  8],
  #            [9.0, 10],
  #            [11.0,12]]

  #   let ab = [[ 58.0, 64],
  #             [139.0,154]]

  #   var res_ab: array[2, array[2, float]]
  #   gemm_strided(
  #     2, 2, 3,
  #     1.0,  a[0][0].unsafeAddr, 3, 1,
  #           b[0][0].unsafeAddr, 2, 1,
  #     0.0,  res_ab[0][0].addr,  2, 1
  #     )

  #   doAssert res_ab == ab

  # block:
  #   # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
  #   # (M x K) * (K x N) with M < N
  #   let a = [[-2,-3,-1],
  #            [ 3, 0, 4]]
  #   let b = [[ 1, 5, 2,-1],
  #            [-3, 0, 3, 4],
  #            [ 6,-2, 7,-4]]

  #   let ab = [[ 1,-8,-20, -6],
  #             [27, 7, 34,-19]]

  #   var res_ab: array[2, array[4, int]]
  #   gemm_strided(
  #     2, 4, 3,
  #     1,  a[0][0].unsafeAddr, 3, 1,
  #         b[0][0].unsafeAddr, 4, 1,
  #     0,  res_ab[0][0].addr,  4, 1
  #     )

  #   doAssert res_ab == ab

  # block:
  #   # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
  #   # (M x K) * (K x N) with M > N and M > block-size (4x4)
  #   let a =  [[5,6,5,8],
  #             [8,2,8,8],
  #             [0,5,4,0],
  #             [4,0,5,6],
  #             [4,5,0,3]]
  #   let b =  [[5,3,6,0],
  #             [5,2,3,3],
  #             [8,8,2,0],
  #             [7,7,0,0]]

  #   let ab = [[151,123,58,18],
  #             [170,148,70, 6],
  #             [ 57, 42,23,15],
  #             [102, 94,34, 0],
  #             [ 66, 43,39,15]]

  #   var res_ab: array[5, array[4, int]]
  #   gemm_strided(
  #     5, 4, 4,
  #     1,  a[0][0].unsafeAddr, 4, 1,
  #         b[0][0].unsafeAddr, 4, 1,
  #     0,  res_ab[0][0].addr,  4, 1
  #     )

  #   doAssert res_ab == ab

  # block:
  #   let a =  [[2, 4,  3,  1,  3,  1,  3,  1],
  #             [4, 3,  2,  4,  1,  0,  0,  0]]


  #   let b =  [[2, 2],
  #             [2, 1],
  #             [0, 3],
  #             [0, 1],
  #             [0, 2],
  #             [4, 3],
  #             [3, 3],
  #             [2, 1]]

  #   let ab = [[27,37],
  #             [14,23]]

  #   var res_ab: array[2, array[2, int]]
  #   gemm_strided(
  #     2, 2, 8,
  #     1,  a[0][0].unsafeAddr, 8, 1,
  #         b[0][0].unsafeAddr, 2, 1,
  #     0,  res_ab[0][0].addr,  2, 1
  #     )

  #   doAssert res_ab == ab

  # block:
  #   let a =  [[2, 1],
  #             [1, 3],
  #             [2, 1],
  #             [1, 0],
  #             [3, 4],
  #             [2, 4],
  #             [3, 1],
  #             [4, 0]]


  #   let b =  [[2, 2,  0,  4,  0,  0,  4,  2],
  #             [2, 1,  2,  1,  2,  4,  4,  1]]

  #   let ab = [[ 6,  5,  2,  9,  2,  4, 12,  5],
  #             [ 8,  5,  6,  7,  6, 12, 16,  5],
  #             [ 6,  5,  2,  9,  2,  4, 12,  5],
  #             [ 2,  2,  0,  4,  0,  0,  4,  2],
  #             [14, 10,  8, 16,  8, 16, 28, 10],
  #             [12,  8,  8, 12,  8, 16, 24,  8],
  #             [ 8,  7,  2, 13,  2,  4, 16,  7],
  #             [ 8,  8,  0, 16,  0,  0, 16,  8]]

  #   var res_ab: array[8, array[8, int]]
  #   gemm_strided(
  #     8, 2, 8,
  #     1,  a[0][0].unsafeAddr, 2, 1,
  #         b[0][0].unsafeAddr, 8, 1,
  #     0,  res_ab[0][0].addr,  8, 1
  #     )

  #   echo "expected: ", ab
  #   echo "result: ",   res_ab

  #   # doAssert res_ab == ab

  # block:
  #   # from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%222%22,%224%22,%223%22,%221%22,%223%22,%221%22,%223%22,%221%22],[%221%22,%222%22,%221%22,%221%22,%222%22,%220%22,%224%22,%223%22],[%222%22,%220%22,%220%22,%223%22,%220%22,%224%22,%224%22,%221%22],[%221%22,%221%22,%224%22,%220%22,%223%22,%221%22,%223%22,%220%22],[%223%22,%224%22,%221%22,%221%22,%224%22,%222%22,%223%22,%224%22],[%222%22,%224%22,%220%22,%222%22,%223%22,%223%22,%223%22,%224%22],[%223%22,%220%22,%220%22,%223%22,%221%22,%224%22,%223%22,%221%22],[%224%22,%223%22,%222%22,%224%22,%221%22,%220%22,%220%22,%220%22]]&matrix2=[[%222%22,%222%22,%220%22,%224%22,%220%22,%220%22,%224%22,%222%22],[%222%22,%220%22,%220%22,%221%22,%221%22,%221%22,%223%22,%221%22],[%220%22,%222%22,%222%22,%220%22,%222%22,%222%22,%223%22,%223%22],[%220%22,%220%22,%221%22,%220%22,%224%22,%222%22,%224%22,%221%22],[%220%22,%220%22,%221%22,%223%22,%224%22,%222%22,%224%22,%222%22],[%224%22,%223%22,%224%22,%221%22,%224%22,%224%22,%220%22,%223%22],[%223%22,%223%22,%220%22,%222%22,%221%22,%222%22,%223%22,%223%22],[%222%22,%221%22,%222%22,%221%22,%222%22,%224%22,%224%22,%221%22]]&operator=*
  #   # (N x N) * (N x N) with N multiple of block size

  #   let a =  [[2, 4,  3,  1,  3,  1,  3,  1],
  #             [1, 2,  1,  1,  2,  0,  4,  3],
  #             [2, 0,  0,  3,  0,  4,  4,  1],
  #             [1, 1,  4,  0,  3,  1,  3,  0],
  #             [3, 4,  1,  1,  4,  2,  3,  4],
  #             [2, 4,  0,  2,  3,  3,  3,  4],
  #             [3, 0,  0,  3,  1,  4,  3,  1],
  #             [4, 3,  2,  4,  1,  0,  0,  0]]


  #   let b =  [[2, 2,  0,  4,  0,  0,  4,  2],
  #             [2, 0,  0,  1,  1,  1,  3,  1],
  #             [0, 2,  2,  0,  2,  2,  3,  3],
  #             [0, 0,  1,  0,  4,  2,  4,  1],
  #             [0, 0,  1,  3,  4,  2,  4,  2],
  #             [4, 3,  4,  1,  4,  4,  0,  3],
  #             [3, 3,  0,  2,  1,  2,  3,  3],
  #             [2, 1,  2,  1,  2,  4,  4,  1]]

  #   let ab = [[27,23,16,29,35,32,58,37],
  #             [24,19,11,23,26,30,49,27],
  #             [34,29,21,21,34,34,36,32],
  #             [17,22,15,21,28,25,40,33],
  #             [39,27,23,40,45,46,72,41],
  #             [41,26,25,34,47,48,65,38],
  #             [33,28,22,26,37,34,41,33],
  #             [14,12, 9,22,27,17,51,23]]

  #   var res_ab: array[8, array[8, int]]
  #   gemm_strided(
  #     8, 8, 8,
  #     1,  a[0][0].unsafeAddr, 8, 1,
  #         b[0][0].unsafeAddr, 8, 1,
  #     0,  res_ab[0][0].addr,  8, 1
  #     )

  #   echo "expected: ", ab
  #   echo "result: ",   res_ab

  #   # doAssert res_ab == ab

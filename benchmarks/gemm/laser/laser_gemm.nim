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
    if pc_mkA.ncols < kc: # last iteration
      kc = pc_mkA.ncols

    var pc_kcncB = jc_kncB.sliceRows(kc)            # B[pc:pc+kc, jc:jc+nc]
    let bufB{.restrict.} = tiles.b                  # panel [kc, nc] (nc is large or unknown)
    pack_B_kc_nc[T, ukernel](bufB, kc, pc_kcncB)    # mutate bufB and pc_kcncB

    # ####################################
    # 3. for ic = 0,...,m−1 in steps of mc
    var mc = tiles.mc
    var ic_mkcA = pc_mkA.sliceCols(kc)              # A[0:M, pc:pc+kc]
    doWhile 0 < ic_mkcA.nrows:
      if ic_mkcA.nrows < mc: # last iteration
        mc = ic_mkcA.nrows

      var jr_mckcA = ic_mkcA.sliceRows(mc)          # A[ic:ic+mc, pc:pc+kc]
      let bufA{.restrict.} = assume_aligned tiles.a # block [mc, kc]
      pack_A_mc_kc[T, ukernel](bufA, kc, jr_mckcA)  # mutate bufA and jr_mckcA

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
    # template dispatch(cpu_features: static CPUFeatureX86){.dirty.} =
    #   const ukernel = MicroKernel(mr: 6, nr: 16, vec_size:32, cpu_simd: x86_AVX2) # cpu_features.x86_ukernel(T)
    #   let tiles = ukernel.newTiles(T, M, N, K)
    #   gemm_impl(
    #     alpha, vA, vB,
    #     beta, vC,
    #     tiles,
    #     ukernel
    #   )
    #   return

    # when defined(i386) or defined(amd64):
    #   when T is SomeFloat:
    #     if cpuinfo_has_x86_avx512f(): dispatch(x86_AVX512)
    #     elif cpuinfo_has_x86_avx():   dispatch(x86_AVX) # Handles AVX2 as well, only diff is that AVX2 can issue 2xFMA
    #     elif cpuinfo_has_x86_sse2():   dispatch(x86_SSE2)
    #     elif cpuinfo_has_x86_sse():   dispatch(x86_SSE)
    #   else: # Integers are taking advantage of wider registers later (in SSE2 and AVX2)
    #     if cpuinfo_has_x86_avx512f(): dispatch(x86_AVX512)
    #     elif cpuinfo_has_x86_avx2():   dispatch(x86_AVX2)
    #     elif cpuinfo_has_x86_sse2():   dispatch(x86_SSE2)
    # dispatch(x86_Generic)

    const ukernel = MicroKernel(mr: 6, nr: 16, vec_size:32, cpu_simd: x86_AVX2) # cpu_features.x86_ukernel(T)
    let tiles = ukernel.newTiles(T, M, N, K)
    gemm_impl[T, ukernel](
      alpha, vA, vB,
      beta, vC,
      tiles
    )

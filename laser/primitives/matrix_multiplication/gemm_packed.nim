# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#        Matrix multiplication with prepacked matrices
#
# ############################################################

import
  ../../compiler_optim_hints, ../../openmp,
  ../../private/align_unroller,
  ../../private/memory,
  ./gemm_utils, ./gemm_tiling, ./gemm_packing

withCompilerOptimHints()

type
  PackedA[T; MR: static int] = ref object
    a*: ptr UncheckedArray[T]
    M*, K*: int
    mc*, kc*: int
    pc_num_kc_tiles*: int
    ic_num_mc_tiles*: int

  PackedB[T; NR: static int] = ref object
    b*: ptr UncheckedArray[T]
    K*, N*: int
    kc*, nc*: int

proc deallocPackedA[T, MR](packedA: PackedA[T, MR]) =
  if not packedA.a.isNil:
    deallocShared packedA.a

proc deallocPackedB(packedB: PackedB) =
  if not packedB.b.isNil:
    deallocShared packedB.b

proc gemm_allocPackedA*(
      M, K: int,
      T: type,
      MR: static int
    ): PackedA[T, MR] =

  new result, deallocPackedA[T, MR]
  result.M = M
  result.K = K
  result.mc = min( 768 div T.sizeof, M)
  result.kc = min(2048 div T.sizeof, K)

  result.ic_num_mc_tiles = (M+result.mc-1) div result.mc

  let bufA_size = T.sizeof * K * (M+MR)
  let a_mem = allocShared(bufA_size + 63)
  result.a = assume_aligned align_raw_data(T, a_mem)

proc gemm_packA*[T, MR](
        ukernel: static Microkernel,
        packedA: PackedA[T, MR],
        A: ptr T,
        rowStrideA, colStrideA: int,
    ) =
  ## Warning: use the same number of threads
  ## for matrix multiplication

  static: assert MR == ukernel.mr

  let M = packedA.M
  let K = packedA.K
  let vA = A.toMatrixView(rowStrideA, colStrideA)

  for pc in countup(0, K-1, packedA.kc):
    let kc = min(K - pc, packedA.kc)
    for icb in 0||(packedA.ic_num_mc_tiles - 1):
      let thread_id = omp_get_thread_num()

      # In the prepack case we don't override the
      # kc*mc matrix at each kc step, we pack the complete
      # kc*mc * num_kc_mc tiles. pc = index * kc
      let pA = packedA.a + thread_id * packedA.mc * pc
      prefetch(pA, Write, LowTemporalLocality)

      let ic = icb * packedA.mc
      let mc = min(M-ic, packedA.mc)

      let mckcA = vA.stride(ic, pc)
      pack_A_mc_kc[T, ukernel](pA, mc, kc, mckcA)

when isMainModule:
  # Tests
  block:
    let a = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    let packedA = gemm_allocPackedA(3, 3, float64, 6)
    const ukernel = x86_ukernel(x86_AVX2, float64, true)

    ukernel.gemm_packA(
      packedA, a[0][0].unsafeAddr,
      3, 1
    )

    var buf: seq[float]
    for i in 0 ..< (packedA.M+ukernel.mr) * packedA.K:
      buf.add packedA.a[i]

    echo a
    echo buf

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../cpuinfo, ../../compiler_optim_hints, ../../openmp,
  ../../private/align_unroller,
  ./gemm_tiling, ./gemm_utils, ./gemm_packing,
  ./gemm_ukernel_dispatch

withCompilerOptimHints()

# ############################################################
#
#            GEMM Prepacked Matrices A and B
#
# ############################################################

template dispatch(
    return_void: static bool,
    func_call: untyped): untyped {.dirty.} =
  ## Warning: statements after dispatch are unreachable
  template dispatch_opt(cpu_features: static CPUFeatureX86): untyped {.dirty.} =
    ## Dispatch depending on detected CPU features.
    type A = T # workaround "Cannot evaluate at compile-time
    # c_unit_stride is not relevant here
    const ukernel = cpu_features.x86_ukernel(A, c_unit_stride = false)

    when return_void:
      func_call
      return
    else:
      return func_call 

  when defined(i386) or defined(amd64):
    when T is float32:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_fma3():    dispatch_opt(x86_AVX_FMA)
      elif cpuinfo_has_x86_avx():     dispatch_opt(x86_AVX)
      elif cpuinfo_has_x86_sse():     dispatch_opt(x86_SSE)
    elif T is float64:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_fma3():    dispatch_opt(x86_AVX_FMA)
      elif cpuinfo_has_x86_avx():     dispatch_opt(x86_AVX)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
    elif T is int32 or T is uint32:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_avx2():    dispatch_opt(x86_AVX2)
      elif cpuinfo_has_x86_sse41():   dispatch_opt(x86_SSE4_1)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
    elif T is int64:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
  dispatch_opt(x86_Generic)

# ############################################################
#
#                    Packing B
#
# ############################################################

func gemm_prepackB_mem_required_impl*(
  ukernel: static MicroKernel,
  T: typedesc,
  M, N, K: int): int =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const NR = ukernel.nr

  let pc_num_iter = get_num_tiles(K, KC)
  let upanelB_size = KC * round_step_up(NC, NR)

  result = T.sizeof * upanelB_size * pc_num_iter

func gemm_prepackB_mem_required*(
  T: typedesc,
  M, N, K: int): int =
  ## Returns the amount of memory that needs to be preallocated
  ## to pack matrix B.

  dispatch(return_void = false):
    gemm_prepackB_mem_required_impl(
      ukernel, T, M, N, K
    )

proc gemm_prepackB_impl[T; ukernel: static MicroKernel](
        dst: ptr UncheckedArray[T],
        M, N, K: int,
        vB: MatrixView[T]
      ) =
  
  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  let pc_num_iter = get_num_tiles(K, KC)
  let upanelB_size = KC * round_step_up(NC, ukernel.nr)
  for pcb in 0||(pc_num_iter-1):
    let packB = dst + pcb * upanelB_size
    prefetch(packB, Write, LowTemporalLocality)

    let pc = pcb * KC
    let kc = min(K - pc, KC)
    let kcncB = vB.stride(pc, 0)

    # Note: pack_B also creates a parallel region
    #       this will cause issues if omp_get_nested = 1
    pack_B_kc_nc[T, ukernel](
      packB,
      KC, NC, kcncB
    ) 
  
proc gemm_prepackB*[T](
        dst_packedB: ptr (T or UncheckedArray[T]),
        M, N, K: int,
        src_B: ptr T, rowStrideB, colStrideB: int) =
  ## Prepack matrix B of shape KxN
  ## and strides rowStrideB and colStrideB
  ## for matrix multiplication.
  ## B must be 64-bit aligned.
  ##
  ## For optimal performance packing is machine and architecture dependent
  ## i.e. it depends on detected features like AVX and number of cores
  ## and may depend on your machine cache sizes in the future.
  ## It is unsafe to store or serialize it.

  let vB = src_B.toMatrixView(rowStrideB, colStrideB)
  let dst = cast[ptr UncheckedArray[T]](dst_packedB)

  dispatch(return_void = true):
    gemm_prepackB_impl[T, ukernel](
      dst,
      M, N, K,
      vB
    )

# ############################################################
#
#                    Packing A
#
# ############################################################

func gemm_prepackA_mem_required_impl*(
  ukernel: static MicroKernel,
  T: typedesc,
  M, N, K: int): int =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const MR = ukernel.mr

  let pc_num_iter = get_num_tiles(K, KC)
  let ic_num_iter = get_num_tiles(M, MC)
  let upanelA_size = KC * round_step_up(MC, MR)

  result = T.sizeof * upanelA_size * pc_num_iter * ic_num_iter

func gemm_prepackA_mem_required*(
  T: typedesc,
  M, N, K: int): int =
  ## Returns the amount of memory that needs to be preallocated
  ## to pack matrix B.

  dispatch(return_void = false):
    gemm_prepackA_mem_required_impl(
      ukernel, T, M, N, K
    )

proc gemm_prepackA_impl[T; ukernel: static MicroKernel](
        dst: ptr UncheckedArray[T],
        M, N, K: int,
        vA: MatrixView[T]
      ) =
  
  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const MR = ukernel.mr

  let pc_num_iter = get_num_tiles(K, KC)
  let ic_num_iter = get_num_tiles(M, MC)
  let upanelA_size = KC * round_step_up(MC, MR)

  for pcb in 0||(pc_num_iter-1):
    let pc = pcb * KC
    let kc = min(K - pc, KC)

    for icb in 0 ..< ic_num_iter:
      let packA = dst + pc*pc_num_iter + icb*upanelA_size
      prefetch(packA, Write, LowTemporalLocality)
      let ic = icb * MC
      let mc = min(M-ic, MC)

      let mckcA = vA.stride(ic, pc)
      pack_A_mc_kc[T, ukernel](packA, mc, kc, mckcA)

proc gemm_prepackA*[T](
        dst_packedA: ptr (T or UncheckedArray[T]),
        M, N, K: int,
        src_A: ptr T, rowStrideA, colStrideA: int) =
  ## Prepack matrix A of shape MxK
  ## and strides rowStrideA and colStrideA
  ## for matrix multiplication.
  ## B must be 64-bit aligned.
  ##
  ## For optimal performance packing is machine and architecture dependent
  ## i.e. it depends on detected features like AVX and number of cores
  ## and may depend on your machine cache sizes in the future.
  ## It is unsafe to store or serialize it.

  let vA = src_A.toMatrixView(rowStrideA, colStrideA)
  let dst = cast[ptr UncheckedArray[T]](dst_packedA)

  dispatch(return_void = true):
    gemm_prepackA_impl[T, ukernel](
      dst,
      M, N, K,
      vA
    )


# ############################################################
#
#                       Private tests
#
# ############################################################

when isMainModule:
  # Tests
  block:
    let
      M = 3
      N = 3
      K = 3

    let packedB_size = gemm_prepackB_mem_required(
      float, M, N, K
    )

    let b = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    var packB = newSeq[float](packedB_size)

    gemm_prepackB(
      packB[0].addr,
      M, N, K,
      b[0][0].unsafeAddr,
      3, 1
    )

    echo packB


    let packedA_size = gemm_prepackA_mem_required(
      float, M, N, K
    )

    let a = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    var packA = newSeq[float](packedA_size)

    gemm_prepackA(
      packA[0].addr,
      M, N, K,
      a[0][0].unsafeAddr,
      3, 1
    )
    echo packA
# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Specialized microkernels for matrix multiplication

import
  ./laser_gemm_tiling, ./laser_gemm_matrix, ./laser_gemm_utils,
  ../../../laser/[cpuinfo, compiler_optim_hints],
  macros,
  ./laser_gemm_ukernel_generic, ./laser_gemm_ukernel_aux

macro B_load(mB, B: untyped, NbVecs: static int, k: int): untyped =
  result = newStmtList()
  for z in 0 ..< NbVecs:
    result.add quote do:
      `mB`[`z`] = mm256_load_ps(`B`[`k`*NR+`z`*MR].addr)

macro A_set1(mA, A: untyped, NbVecs: static int, k, i: int): untyped =
  result = newStmtList()
  for z in 0 ..< NbVecs:
    result.add quote do:
      `mA`[`z`] = mm256_set1_ps(`A`[`k`*MR+`z`*NR+`i`])

proc gebb_ukernel_f32_avx*[ukernel: static MicroKernel](
      kc: int,
      alpha: float32, packedA, packedB: ptr UncheckedArray[float32],
      beta: float32, vC: MatrixView[float32]
    ) =
  const
    MR = ukernel.extract_mr()
    NR = ukernel.extract_nr()
    vec_size = ukernel.extract_vecsize
    simd = ukernel.extract_cpu_simd
    NbElems = 8
    NbVecs = NR div NbElems

  static:
    assert vecsize == 32
    assert simd in {x86_AVX, x86_AVX2, x86_AVX512}
    assert NR div 8 == 0 # Unrolling checks
    assert MR div 2 == 0

  # var AB{.align_variable.}: array[MR, array[NR, float32]]
  var AB{.align_variable.}: array[MR, array[NbVecs, m256]]
  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  var mA, mB: array[NbVecs, m256]

  for k in 0 ..< kc:
    B_load(mB, B, NbVecs, k)
    for i in 0 ..< MR:
      A_set1(mA, A, NbVecs, k, i)
      for z in 0 ..< NbVecs:
        when simd == x86_AVX:
          AB[i][z] = mm256_add_ps(mm256_mul_ps(mA[z], mB[z]), AB[i][z])
        else:
          AB[i][z] = mm256_fmadd_ps(mA[z], mB[z], AB[i][z])

  gebb_ukernel_epilogue(
    alpha, cast[array[MR, array[NR, float32]]](AB),
    beta, vC)


# #################################################
  # Reference loop -  AB: array[MR, array[NR, float32]]
  # for k in 0 ..< kc:
  #   for i in 0 ..< MR:
  #     for j in 0 ..< NR:
  #       AB[i][j] += A[k*MR+i] * B[k*NR+j]

  # Reference AVX - AB: array[MR, m256] and NR == 8
  # for k in 0 ..< kc:
  #   for z in 0 ..< NbVecs:
  #     mB[z] = mm256_load_ps(B[k*NR+MR*z].addr)
  #   for i in 0 ..< MR:
  #     mA = mm256_set1_ps(A[k*MR+i])
  #     when simd == x86_AVX:
  #       AB[i] = mm256_add_ps(mm256_mul_ps(mA, mB), AB[i])
  #     else:
  #       AB[i] = mm256_fmadd_ps(mA, mB, AB[i])

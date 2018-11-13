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

# TODO: the generic template in laser_gemm_ukernel_aux
# causes crashes.

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

  static:
    assert vecsize == 32
    assert simd in {x86_AVX, x86_AVX2, x86_AVX512}
    assert NR div 8 == 0 # Unrolling checks
    assert MR div 2 == 0

  var AB{.align_variable.}: array[MR, array[NR div NbElems, m256]]
  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  var mA: array[2, m256]
  template A0 = mA[0]
  template A1 = mA[1]

  var mB: array[NR div NbElems, m256]

  for i, bval in mB.mpairs:
    bval = mm256_load_ps(B[i*NbElems].addr)

  for k in 0 ..< kc:
    # TODO prefetch
    for i in countup(0, MR-1, 2):
      let A0 = mm256_set1_ps(A[0])
      let A1 = mm256_set1_ps((A+1)[0])
      for j in 0 ..< NR div NbElems:
        when simd == x86_AVX:
          AB[i  ][j] = mm256_add_ps(mm256_mul_ps(A0, mB[j]), AB[i  ][j])
          AB[i+1][j] = mm256_add_ps(mm256_mul_ps(A1, mB[j]), AB[i+1][j])
        else:
          AB[i  ][j] = mm256_fmadd_ps(A0, mB[j], AB[i  ][j])
          AB[i+1][j] = mm256_fmadd_ps(A1, mB[j], AB[i+1][j])
      A += MR

  gebb_ukernel_epilogue(
    alpha, cast[array[MR, array[NR, float32]]](AB),
    beta, vC)

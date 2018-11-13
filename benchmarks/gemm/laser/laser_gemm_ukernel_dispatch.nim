# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./laser_gemm_tiling, ./laser_gemm_matrix,
  ./laser_gemm_ukernel_generic,
  ./laser_gemm_ukernel_specialised

export gebb_ukernel_edge

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =
  const simd = ukernel.extract_cpu_simd()

  when T is float32 and simd in {x86_AVX, x86_AVX2, x86_AVX512}:
    gebb_ukernel_f32_avx[ukernel](
            kc,
      alpha, packedA, packedB,
      beta, vC
    )
  else:
    gebb_ukernel_generic[T, ukernel](
            kc,
      alpha, packedA, packedB,
      beta, vC
    )

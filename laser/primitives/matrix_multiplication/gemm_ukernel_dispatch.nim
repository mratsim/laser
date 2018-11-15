# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic,
  ./gemm_ukernel_avx,
  ../../compiler_optim_hints

export gebb_ukernel_edge

# ############################################################
#
#         Dispatch according to runtime CPU detection
#
# ############################################################

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =
  const simd = ukernel.extract_cpu_simd()
  const MR = ukernel.extract_MR()
  const NbElems = 8 # TODO

  prefetch(packedB, Read, LowTemporalLocality)
  prefetch(packedB + NbElems, Read, LowTemporalLocality)
  for i in 0 ..< MR:
    prefetch(vC[i, 0].addr, Write, HighTemporalLocality)

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

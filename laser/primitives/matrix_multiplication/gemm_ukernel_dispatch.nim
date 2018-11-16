# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic,
  ./gemm_ukernel_avx,
  ../../compiler_optim_hints,
  macros

export gebb_ukernel_edge

# ############################################################
#
#         Dispatch according to runtime CPU detection
#
# ############################################################

{.experimental: "dynamicBindSym".}
macro dispatch(
    ukernel: static MicroKernel,
    kc: int,
    alpha: typed, packedA, packedB: ptr UncheckedArray[typed],
    beta: typed, vC: MatrixView[typed]
  ): untyped =

  let simd = ukernel.cpu_simd
  let MR = ukernel.mr
  let nb_scalars = ukernel.nb_scalars

  result = newStmtList()

  # 1. Prefetch packedB (used in microkernel)
  #         and C (used in epilogue update)
  result.add quote do:
    prefetch(`packedB`, Read, LowTemporalLocality)
    prefetch(`packedB` + `nb_scalars`, Read, LowTemporalLocality)
    for i in 0 ..< `MR`:
      prefetch(`vC`[i, 0].addr, Write, HighTemporalLocality)

  # 2. Dispatch according to type and SIMD support
  let symT = getTypeInst(alpha)             # Retrieve float32/int64 ...

  # 2.1. No SIMD case
  if simd == x86_Generic:
    result.add quote do:
      gebb_ukernel_generic[`symT`, ukernel]( # Hack: ukernel is generic from the calling proc
              `kc`,
        `alpha`, `packedA`, `packedB`,
        `beta`, `vC`
      )
    return

  # 2.2. SIMD case
  let simdTag = $simd
  let ukernel_name = bindSym("gebb_ukernel_" & $symT & "_" & simdTag)
  result.add quote do:
    `ukernel_name`[ukernel]( # Hack: ukernel is generic from the calling proc
      `kc`,
      `alpha`, `packedA`, `packedB`,
      `beta`, `vC`
    )

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =

  ukernel.dispatch(kc, alpha, packedA, packedB, beta, vC)

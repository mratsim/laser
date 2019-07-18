# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# TODO: merge with laser/primitives/matrix_multiplication/gemm_tiling

import
  # Standard library
  macros,
  # Internal
  ./platform_common,
  ../../simd

# ###########################################
#
#    SIMD configuration for x86 and x86_64
#
# ###########################################

type
  SimdArch* = enum
    ArchGeneric,
    x86_SSE,
    # x86_SSE2,
    # x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    # x86_AVX2,
    # x86_AVX512

const SimdWidth*:array[SimdArch, int] = [
  ArchGeneric:   0,
  x86_SSE:     128 div 8,
  # x86_SSE2:    128 div 8,
  # x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  # x86_AVX2:    256 div 8,
  # x86_AVX512:  512 div 8
]

const SimdAlignment*:array[SimdArch, int] = [
  ArchGeneric:  0,
  x86_SSE:     16,
  x86_AVX:     32,
  x86_AVX_FMA: 32,
]

template sse_fma_fallback(a, b, c: m128): m128 =
  mm_add_ps(mm_mul_ps(a, b), c)

template avx_fma_fallback(a, b, c: m256): m256 =
  mm256_add_ps(mm256_mul_ps(a, b), c)

template sse_fma_fallback(a, b, c: m128d): m128d =
  mm_add_pd(mm_mul_pd(a, b), c)

template avx_fma_fallback(a, b, c: m256d): m256d =
  mm256_add_pd(mm256_mul_pd(a, b), c)

proc simdX86Float32(): array[SimdArch, array[SimdPrimitives, NimNode]] =
  let sse = [
    simdSetZero:   bindSym"mm_setzero_ps",
    simdBroadcast: bindSym"mm_set1_ps",
    simdLoadA:     bindSym"mm_load_ps",
    simdLoadU:     bindSym"mm_loadu_ps",
    simdStoreA:    bindSym"mm_store_ps",
    simdStoreU:    bindSym"mm_storeu_ps",
    simdAdd:       bindSym"mm_add_ps",
    simdMul:       bindSym"mm_mul_ps",
    simdFma:       bindSym"sse_fma_fallback",
    simdType:      bindSym"m128"
  ]

  let avx: array[SimdPrimitives, NimNode] = [
    simdSetZero:   bindSym"mm256_setzero_ps",
    simdBroadcast: bindSym"mm256_set1_ps",
    simdLoadA:     bindSym"mm256_load_ps",
    simdLoadU:     bindSym"mm256_loadu_ps",
    simdStoreA:    bindSym"mm256_store_ps",
    simdStoreU:    bindSym"mm256_storeu_ps",
    simdAdd:       bindSym"mm256_add_ps",
    simdMul:       bindSym"mm256_mul_ps",
    simdFma:       bindSym"avx_fma_fallback",
    simdType:      bindSym"m256"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = bindSym"mm256_fmadd_ps"

  result = [
    ArchGeneric: genericPrimitives(),
    x86_SSE: sse,
    x86_AVX: avx,
    x86_AVX_FMA: avx_fma
  ]

proc simdX86Float64(): array[SimdArch, array[SimdPrimitives, NimNode]] =
  let sse = [
    simdSetZero:   bindSym"mm_setzero_pd",
    simdBroadcast: bindSym"mm_set1_pd",
    simdLoadA:     bindSym"mm_load_pd",
    simdLoadU:     bindSym"mm_loadu_pd",
    simdStoreA:    bindSym"mm_store_pd",
    simdStoreU:    bindSym"mm_storeu_pd",
    simdAdd:       bindSym"mm_add_pd",
    simdMul:       bindSym"mm_mul_pd",
    simdFma:       bindSym"sse_fma_fallback",
    simdType:      bindSym"m128d"
  ]

  let avx: array[SimdPrimitives, NimNode] = [
    simdSetZero:   bindSym"mm256_setzero_pd",
    simdBroadcast: bindSym"mm256_set1_pd",
    simdLoadA:     bindSym"mm256_load_pd",
    simdLoadU:     bindSym"mm256_loadu_pd",
    simdStoreA:    bindSym"mm256_store_pd",
    simdStoreU:    bindSym"mm256_storeu_pd",
    simdAdd:       bindSym"mm256_add_pd",
    simdMul:       bindSym"mm256_mul_pd",
    simdFma:       bindSym"avx_fma_fallback",
    simdType:      bindSym"m256d"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = bindSym"mm256_fmadd_pd"

  result = [
    ArchGeneric: genericPrimitives(),
    x86_SSE: sse,
    x86_AVX: avx,
    x86_AVX_FMA: avx_fma
  ]

let MapX86Float32{.compileTime.} = simdX86Float32()
let MapX86Float64{.compileTime.} = simdX86Float64()

proc SimdMap*(arch: SimdArch, T: NimNode, p: SimdPrimitives): NimNode =
  if T.eqIdent"float32":
    result = MapX86Float32[arch][p]
  elif T.eqIdent"float64":
    result = MapX86Float64[arch][p]
  elif T.eqIdent"int":
    # TODO - hack for integer array accesses
    assert arch == ArchGeneric
    result = MapX86Float32[ArchGeneric][p]
  else:
    error "Unsupported type: \"" & T.repr & '\"'

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# TODO: merge with laser/primitives/matrix_multiplication/gemm_tiling

import
  # Standard library
  macros,
  # Internal
  ../../simd

type
  SimdPrimitives* = enum
    simdSetZero
    simdBroadcast
    simdLoadA
    simdLoadU
    simdStoreA
    simdStoreU
    simdAdd
    simdMul
    simdFma
    simdType

  SimdArch* = enum
    ArchGeneric,
    x86_SSE,
    # x86_SSE2,
    # x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    # x86_AVX2,
    # x86_AVX512

const SimdWidth* = [
  x86_SSE:     128 div 8,
  # x86_SSE2:    128 div 8,
  # x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  # x86_AVX2:    256 div 8,
  # x86_AVX512:  512 div 8
]

const SimdAlignment* = [
  x86_SSE:     16,
  x86_AVX:     32,
  x86_AVX_FMA: 32,
]

template sse_fma_fallback(a, b, c: m128): m128 =
  mm_add_ps(mm_mul_ps(a, b), c)

template avx_fma_fallback(a, b, c: m128): m128 =
  mm256_add_ps(mm256_mul_ps(a, b), c)

proc genSimdTableX86(): array[SimdArch, array[SimdPrimitives, NimNode]] =

  let sse: array[SimdPrimitives, NimNode] = [
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
    ArchGeneric: default(array[SimdPrimitives, NimNode]),
    x86_SSE: sse,
    x86_AVX: avx,
    x86_AVX_FMA: avx_fma
  ]

let SimdTable*{.compileTime.} = genSimdTableX86()

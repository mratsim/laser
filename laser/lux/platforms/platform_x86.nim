# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# TODO: merge with laser/primitives/matrix_multiplication/gemm_tiling

import
  macros

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

proc genSimdTableX86(): array[SimdArch, array[SimdPrimitives, NimNode]] =

  let sse: array[SimdPrimitives, NimNode] = [
    simdSetZero:   ident"mm_setzero_ps",
    simdBroadcast: ident"mm_set1_ps",
    simdLoadA:     ident"mm_load_ps",
    simdLoadU:     ident"mm_loadu_ps",
    simdStoreA:    ident"mm_store_ps",
    simdStoreU:    ident"mm_storeu_ps",
    simdAdd:       ident"mm_add_ps",
    simdMul:       ident"mm_mul_ps",
    simdFma:       ident"sse_fma_fallback",
    simdType:      ident"m128"
  ]

  let avx: array[SimdPrimitives, NimNode] = [
    simdSetZero:   ident"mm256_setzero_ps",
    simdBroadcast: ident"mm256_set1_ps",
    simdLoadA:     ident"mm256_load_ps",
    simdLoadU:     ident"mm256_loadu_ps",
    simdStoreA:    ident"mm256_store_ps",
    simdStoreU:    ident"mm256_storeu_ps",
    simdAdd:       ident"mm256_add_ps",
    simdMul:       ident"mm256_mul_ps",
    simdFma:       ident"avx_fma_fallback",
    simdType:      ident"m256"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = ident"mm256_fmadd_ps"

  result = [
    x86_SSE: sse,
    x86_AVX: avx,
    x86_AVX_FMA: avx_fma
  ]

let SimdTable*{.compileTime.} = genSimdTableX86()

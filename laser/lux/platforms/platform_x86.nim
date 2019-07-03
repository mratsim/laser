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
    x86_SSE,
    # x86_SSE2,
    # x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    # x86_AVX2,
    # x86_AVX512

const VecWidth = [
  x86_SSE:     128 div 8,
  # x86_SSE2:    128 div 8,
  # x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  # x86_AVX2:    256 div 8,
  # x86_AVX512:  512 div 8
]

proc genSimdTableX86(): array[SimdArch, array[SimdPrimitives, NimNode]] =

  let sse: array[SimdPrimitives, NimNode] = [
    simdSetZero:   newIdentNode"mm_setzero_ps",
    simdBroadcast: newIdentNode"mm_set1_ps",
    simdLoadA:     newIdentNode"mm_load_ps",
    simdLoadU:     newIdentNode"mm_loadu_ps",
    simdStoreA:    newIdentNode"mm_store_ps",
    simdStoreU:    newIdentNode"mm_storeu_ps",
    simdAdd:       newIdentNode"mm_add_ps",
    simdMul:       newIdentNode"mm_mul_ps",
    simdFma:       newIdentNode"sse_fma_fallback",
    simdType:      newIdentNode"m128"
  ]

  let avx: array[SimdPrimitives, NimNode] = [
    simdSetZero:   newIdentNode"mm256_setzero_ps",
    simdBroadcast: newIdentNode"mm256_set1_ps",
    simdLoadA:     newIdentNode"mm256_load_ps",
    simdLoadU:     newIdentNode"mm256_loadu_ps",
    simdStoreA:    newIdentNode"mm256_store_ps",
    simdStoreU:    newIdentNode"mm256_storeu_ps",
    simdAdd:       newIdentNode"mm256_add_ps",
    simdMul:       newIdentNode"mm256_mul_ps",
    simdFma:       newIdentNode"avx_fma_fallback",
    simdType:      newIdentNode"m256"
  ]

  var avx_fma = avx
  avx_fma[simdFma] = newIdentNode"mm256_fmadd_ps"

  result = [
    x86_SSE: sse,
    x86_AVX: avx,
    x86_AVX_FMA: avx_fma
  ]

let SimdTable*{.compileTime.} = genSimdTableX86()

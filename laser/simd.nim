# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

when defined(i386) or defined(amd64):
  # SIMD throughput and latency:
  #   - https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  #   - https://www.agner.org/optimize/instruction_tables.pdf

  when defined(vcc):
    {.pragma: x86_type, byCopy, header:"<intrin.h>".}
    {.pragma: x86, noDecl, header:"<intrin.h>".}
  else:
    {.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
    {.pragma: x86, noDecl, header:"<x86intrin.h>".}
  type
    m128* {.importc: "__m128", x86_type.} = object
      raw: array[4, float32]
    m128d* {.importc: "__m128d", x86_type.} = object
      raw: array[2, float64]

  ## SSE
  # Reminder: x86 is little-endian, order is [low part, high part]
  func mm_setzero_ps*(): m128 {.importc: "_mm_setzero_ps", x86.}
    ## [float32 0, 0, 0, 0]
  func mm_set1_ps*(a: float32): m128 {.importc: "_mm_set1_ps", x86.}
    ## [float32 a, a, a, a]
  func mm_load_ps*(aligned_data: ptr float32): m128 {.importc: "_mm_load_ps", x86.}
    ## Load 4 packed float32 in __m128. They must be aligned on 16-byte boundary.
  func mm_load_ss*(aligned_data: ptr float32): m128 {.importc: "_mm_load_ss", x86.}
    ## Load 1 float32 in __m128. in the lower word and zero the rest.
  func mm_add_ps*(a, b: m128): m128 {.importc: "_mm_add_ps", x86.}
    ## Vector addition
  func mm_add_ss*(a, b: m128): m128 {.importc: "_mm_add_ss", x86.}
    ## Low part addition + copy of a
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { A0 + B0, A1, A2, A3 }
  func mm_max_ps*(a, b: m128): m128 {.importc: "_mm_max_ps", x86.}
    ## Vector maximum
  func mm_max_ss*(a, b: m128): m128 {.importc: "_mm_max_ss", x86.}
    ## Low part max + copy of a
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { max(A0,B0), A1, A2, A3 }
  func mm_min_ps*(a, b: m128): m128 {.importc: "_mm_min_ps", x86.}
    ## Vector min
  func mm_min_ss*(a, b: m128): m128 {.importc: "_mm_min_ss", x86.}
    ## Low part min + copy of a
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { min(A0,B0), A1, A2, A3 }

  ##SSE3
  func mm_movehdup_ps*(a: m128): m128 {.importc: "_mm_movehdup_ps", x86.}
    ## Duplicates high parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A1, A1, A3, A3 }
  func mm_moveldup_ps*(a: m128): m128 {.importc: "_mm_moveldup_ps", x86.}
    ## Duplicates low parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A0, A0, A2, A2 }
  func mm_movehl_ps*(a, b: m128): m128 {.importc: "_mm_movehl_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { B2, B3, A2, A3 }
  func mm_cvtss_f32*(a: m128): float32 {.importc: "_mm_cvtss_f32", x86.}
    ## Extract the low part of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   A0

  ## AVX
  type
    m256* {.importc: "__m256", x86_type.} = object
      raw: array[8, float32]
    m256d* {.importc: "__m256d", x86_type.} = object
      raw: array[4, float64]

  func mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps", x86.}
    ## [float32 0, 0, 0, 0, 0, 0, 0, 0]
  func mm256_set1_ps*(a: float32): m128 {.importc: "_mm256_set1_ps", x86.}
    ## [float32 a, a, a, a, a, a, a, a]
  func mm256_load_ps*(aligned_data: ptr float32): m256 {.importc: "_mm256_load_ps", x86.}
    ## Load 8 packed float32 in __m128. They must be aligned on 16-byte boundary.
  func mm256_add_ps*(a, b: m256): m256 {.importc: "_mm256_add_ps", x86.}
    ## Vector addition
  func mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps", x86.}
    ## Vector multiplication
  func mm256_castps256_ps128*(a: m256): m128 {.importc: "_mm256_castps256_ps128", x86.}
    ## Returns the lower part of a m256 in a m128
  func mm256_extractf128_ps*(v: m256, m: cint{lit}): m128 {.importc: "_mm256_extractf128_ps", x86.}
    ## Extracts the low part (m = 0) or high part (m = 1) of a m256 into a m128
    ## m must be a literal

  ## FMA
  func mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps", x86.}
    ## a*b + c

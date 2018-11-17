# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

when defined(i386) or defined(amd64):
  # SIMD throughput and latency:
  #   - https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  #   - https://www.agner.org/optimize/instruction_tables.pdf

  # Reminder: x86 is little-endian, order is [low part, high part]
  # Documentation at https://software.intel.com/sites/landingpage/IntrinsicsGuide/

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
    m128i* {.importc: "__m128i", x86_type.} = object
      raw: array[16, byte]
    m256* {.importc: "__m256", x86_type.} = object
      raw: array[8, float32]
    m256d* {.importc: "__m256d", x86_type.} = object
      raw: array[4, float64]
    m256i* {.importc: "__m128i", x86_type.} = object
      raw: array[32, byte]

  # ############################################################
  #
  #                   SSE - float32 - packed
  #
  # ############################################################

  func mm_setzero_ps*(): m128 {.importc: "_mm_setzero_ps", x86.}
  func mm_set1_ps*(a: float32): m128 {.importc: "_mm_set1_ps", x86.}
  func mm_load_ps*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ps", x86.}
  func mm_loadu_ps*(data: ptr float32): m128 {.importc: "_mm_loadu_ps", x86.}
  func mm_storeu_ps*(mem_addr: ptr float32, a: m128) {.importc: "_mm_storeu_ps", x86.}
  func mm_add_ps*(a, b: m128): m128 {.importc: "_mm_add_ps", x86.}
  func mm_mul_ps*(a, b: m128): m128 {.importc: "_mm_mul_ps", x86.}
  func mm_max_ps*(a, b: m128): m128 {.importc: "_mm_max_ps", x86.}
  func mm_min_ps*(a, b: m128): m128 {.importc: "_mm_min_ps", x86.}

  # ############################################################
  #
  #                    SSE - float32 - scalar
  #
  # ############################################################

  func mm_load_ss*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ss", x86.}
  func mm_add_ss*(a, b: m128): m128 {.importc: "_mm_add_ss", x86.}
  func mm_max_ss*(a, b: m128): m128 {.importc: "_mm_max_ss", x86.}
  func mm_min_ss*(a, b: m128): m128 {.importc: "_mm_min_ss", x86.}

  func mm_cvtss_f32*(a: m128): float32 {.importc: "_mm_cvtss_f32", x86.}
    ## Extract the low part of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   A0

  func mm_movehl_ps*(a, b: m128): m128 {.importc: "_mm_movehl_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { B2, B3, A2, A3 }

  # ############################################################
  #
  #                    SSE2 - float64 - packed
  #
  # ############################################################

  func mm_setzero_pd*(): m128d {.importc: "_mm_setzero_pd", x86.}
  func mm_set1_pd*(a: float64): m128d {.importc: "_mm_set1_pd", x86.}
  func mm_load_pd*(aligned_mem_addr: ptr float64): m128d {.importc: "_mm_load_pd", x86.}
  func mm_loadu_pd*(mem_addr: ptr float64): m128d {.importc: "_mm_loadu_pd", x86.}
  func mm_storeu_pd*(mem_addr: ptr float64, a: m128d) {.importc: "_mm_storeu_pd", x86.}
  func mm_add_pd*(a, b: m128d): m128d {.importc: "_mm_add_pd", x86.}
  func mm_mul_pd*(a, b: m128d): m128d {.importc: "_mm_mul_pd", x86.}

  # ############################################################
  #
  #                    SSE2 - integer - packed
  #
  # ############################################################

  func mm_setzero_si128*(): m128i {.importc: "_mm_setzero_si128", x86.}
  func mm_set1_epi8*(a: int8): m128i {.importc: "_mm_set1_epi8", x86.}
  func mm_set1_epi16*(a: int8): m128i {.importc: "_mm_set1_epi16", x86.}
  func mm_set1_epi32*(a: int8): m128i {.importc: "_mm_set1_epi32", x86.}
  func mm_set1_epi64x*(a: int8): m128i {.importc: "_mm_set1_epi64x", x86.}
  func mm_load_si128*(mem_addr: ptr SomeInteger): m128i {.importc: "_mm_load_si128", x86.}
  func mm_loadu_si128*(mem_addr: ptr SomeInteger): m128i {.importc: "_mm_loadu_si128", x86.}
  func mm_storeu_si128*(mem_addr: ptr SomeInteger, a: m128i) {.importc: "_mm_storeu_si128", x86.}
  func mm_add_epi8*(a, b: m128i): m128i {.importc: "_mm_add_epi8", x86.}
  func mm_add_epi16*(a, b: m128i): m128i {.importc: "_mm_add_epi16", x86.}
  func mm_add_epi32*(a, b: m128i): m128i {.importc: "_mm_add_epi32", x86.}
  func mm_add_epi64*(a, b: m128i): m128i {.importc: "_mm_add_epi64", x86.}

  func mm_mullo_epi16*(a, b: m128i): m128i {.importc: "_mm_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 8 16-bit ints
    ## into intermediate 8 32-bit ints, and keep the low 16-bit parts

  # ############################################################
  #
  #                    SSE3 - float32
  #
  # ############################################################

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

  # ############################################################
  #
  #                    SSE4.1 - integer - packed
  #
  # ############################################################

  func mm_mullo_epi32*(a, b: m128i): m128i {.importc: "_mm_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 4 32-bit ints
    ## into intermediate 4 64-bit ints, and keep the low 32-bit parts

  # ############################################################
  #
  #                    AVX - float32 - packed
  #
  # ############################################################

  func mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps", x86.}
  func mm256_set1_ps*(a: float32): m256 {.importc: "_mm256_set1_ps", x86.}
  func mm256_load_ps*(aligned_mem_addr: ptr float32): m256 {.importc: "_mm256_load_ps", x86.}
  func mm256_loadu_ps*(mem_addr: ptr float32): m256 {.importc: "_mm256_loadu_ps", x86.}
  func mm256_storeu_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_storeu_ps", x86.}
  func mm256_add_ps*(a, b: m256): m256 {.importc: "_mm256_add_ps", x86.}
  func mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps", x86.}

  func mm256_castps256_ps128*(a: m256): m128 {.importc: "_mm256_castps256_ps128", x86.}
    ## Returns the lower part of a m256 in a m128
  func mm256_extractf128_ps*(v: m256, m: cint{lit}): m128 {.importc: "_mm256_extractf128_ps", x86.}
    ## Extracts the low part (m = 0) or high part (m = 1) of a m256 into a m128
    ## m must be a literal

  # ############################################################
  #
  #                   AVX - float64 - packed
  #
  # ############################################################

  func mm256_setzero_pd*(): m256d {.importc: "_mm256_setzero_pd", x86.}
  func mm256_set1_pd*(a: float64): m256d {.importc: "_mm256_set1_pd", x86.}
  func mm256_load_pd*(aligned_mem_addr: ptr float64): m256d {.importc: "_mm256_load_pd", x86.}
  func mm256_loadu_pd*(mem_addr: ptr float64): m256d {.importc: "_mm256_loadu_pd", x86.}
  func mm256_storeu_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_storeu_pd", x86.}
  func mm256_add_pd*(a, b: m256d): m256d {.importc: "_mm256_add_pd", x86.}
  func mm256_mul_pd*(a, b: m256d): m256d {.importc: "_mm256_mul_pd", x86.}

  # ############################################################
  #
  #                 AVX + FMA - float32/64 - packed
  #
  # ############################################################

  func mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps", x86.}
  func mm256_fmadd_pd*(a, b, c: m256d): m256d {.importc: "_mm256_fmadd_pd", x86.}

  # ############################################################
  #
  #                   AVX - integers - packed
  #
  # ############################################################

  func mm256_setzero_si256*(): m256i {.importc: "_mm256_setzero_si256", x86.}
  func mm256_set1_epi8*(a: int8): m256i {.importc: "_mm256_set1_epi8", x86.}
  func mm256_set1_epi16*(a: int8): m256i {.importc: "_mm256_set1_epi16", x86.}
  func mm256_set1_epi32*(a: int8): m256i {.importc: "_mm256_set1_epi32", x86.}
  func mm256_set1_epi64x*(a: int8): m256i {.importc: "_mm256_set1_epi64x", x86.}
  func mm256_load_si256*(mem_addr: ptr SomeInteger): m256i {.importc: "_mm256_load_si256", x86.}
  func mm256_loadu_si256*(mem_addr: ptr SomeInteger): m256i {.importc: "_mm256_loadu_si256", x86.}
  func mm256_storeu_si256*(mem_addr: ptr SomeInteger, a: m256i) {.importc: "_mm256_storeu_si256", x86.}

  # ############################################################
  #
  #                   AVX2 - integers - packed
  #
  # ############################################################

  func mm256_add_epi8*(a, b: m256i): m256i {.importc: "_mm256_add_epi8", x86.}
  func mm256_add_epi16*(a, b: m256i): m256i {.importc: "_mm256_add_epi16", x86.}
  func mm256_add_epi32*(a, b: m256i): m256i {.importc: "_mm256_add_epi32", x86.}
  func mm256_add_epi64*(a, b: m256i): m256i {.importc: "_mm256_add_epi64", x86.}

  func mm256_mullo_epi16*(a, b: m256i): m256i {.importc: "_mm_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 16 16-bit ints
    ## into intermediate 16 32-bit ints, and keep the low 16-bit parts

  func mm256_mullo_epi32*(a, b: m256i): m256i {.importc: "_mm_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 8x 32-bit ints
    ## into intermediate 8x 64-bit ints, and keep the low 32-bit parts

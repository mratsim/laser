import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

template int32x4_muladd_unfused(a, b, c: m128i): m128i =
  mm_add_epi32(mm_mullo_epi32(a, b), c)

ukernel_generator(
      x86_SSE4_1,
      typ = int32,
      vectype = m128i,
      nb_scalars = 4,
      simd_setZero = mm_setzero_si128,
      simd_broadcast_value = mm_set1_epi32,
      simd_load_aligned = mm_load_si128,
      simd_load_unaligned = mm_loadu_si128,
      simd_fma = int32x4_muladd_unfused,
      simd_store_unaligned = mm_storeu_si128,
      simd_mul = mm_mullo_epi32,
      simd_add = mm_add_epi32
    )

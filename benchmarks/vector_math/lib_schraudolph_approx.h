#include <immintrin.h>

// Approximation for EXP(x) -- very fast, but not super accurate
static inline __m256 _mm256_expfaster_ps(const __m256 q) {

    const __m256 C1 = _mm256_set1_ps(1064872507.1541044f);
    const __m256 C2 = _mm256_set1_ps(12102203.161561485f);

    return _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_fmadd_ps(C2, q, C1)));
}

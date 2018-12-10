#include <immintrin.h>

// https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
// Cody Waite Scheme: W. J. Cody & W. Waite, "Software Manual for the Elementary Functions", Prentice Hall 1980.

#define USE_FMA 1

/* compute exp(x) for x in [-87.33654f, 88.72283]
   maximum relative error: 3.1575e-6 (USE_FMA = 0); 3.1533e-6 (USE_FMA = 1)
*/
__m256 faster_more_accurate_exp_avx2 (__m256 x)
{
    __m256 t, f, p, r;
    __m256i i, j;

    const __m256 l2e = _mm256_set1_ps (1.442695041f); /* log2(e) */
    const __m256 l2h = _mm256_set1_ps (-6.93145752e-1f); /* -log(2)_hi */
    const __m256 l2l = _mm256_set1_ps (-1.42860677e-6f); /* -log(2)_lo */
    /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
    const __m256 c0 =  _mm256_set1_ps (0.041944388f);
    const __m256 c1 =  _mm256_set1_ps (0.168006673f);
    const __m256 c2 =  _mm256_set1_ps (0.499999940f);
    const __m256 c3 =  _mm256_set1_ps (0.999956906f);
    const __m256 c4 =  _mm256_set1_ps (0.999999642f);

    /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
    t = _mm256_mul_ps (x, l2e);      /* t = log2(e) * x */
    r = _mm256_round_ps (t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */

#if USE_FMA
    f = _mm256_fmadd_ps (r, l2h, x); /* x - log(2)_hi * r */
    f = _mm256_fmadd_ps (r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
#else // USE_FMA
    p = _mm256_mul_ps (r, l2h);      /* log(2)_hi * r */
    f = _mm256_add_ps (x, p);        /* x - log(2)_hi * r */
    p = _mm256_mul_ps (r, l2l);      /* log(2)_lo * r */
    f = _mm256_add_ps (f, p);        /* f = x - log(2)_hi * r - log(2)_lo * r */
#endif // USE_FMA

    i = _mm256_cvtps_epi32(t);       /* i = (int)rint(t) */

    /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
    p = c0;                          /* c0 */
#if USE_FMA
    p = _mm256_fmadd_ps (p, f, c1);  /* c0*f+c1 */
    p = _mm256_fmadd_ps (p, f, c2);  /* (c0*f+c1)*f+c2 */
    p = _mm256_fmadd_ps (p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_fmadd_ps (p, f, c4);  /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#else // USE_FMA
    p = _mm256_mul_ps (p, f);        /* c0*f */
    p = _mm256_add_ps (p, c1);       /* c0*f+c1 */
    p = _mm256_mul_ps (p, f);        /* (c0*f+c1)*f */
    p = _mm256_add_ps (p, c2);       /* (c0*f+c1)*f+c2 */
    p = _mm256_mul_ps (p, f);        /* ((c0*f+c1)*f+c2)*f */
    p = _mm256_add_ps (p, c3);       /* ((c0*f+c1)*f+c2)*f+c3 */
    p = _mm256_mul_ps (p, f);        /* (((c0*f+c1)*f+c2)*f+c3)*f */
    p = _mm256_add_ps (p, c4);       /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */
#endif // USE_FMA

    /* exp(x) = 2^i * p */
    j = _mm256_slli_epi32 (i, 23); /* i << 23 */
    r = _mm256_castsi256_ps (_mm256_add_epi32 (j, _mm256_castps_si256 (p))); /* r = p * 2^i */

    return r;
}


// /* maximum relative error: 1.7428e-7 (USE_FMA = 0); 1.6586e-7 (USE_FMA = 1) */
// const __m256 c0 =  _mm256_set1_ps (0.008301110f);
// const __m256 c1 =  _mm256_set1_ps (0.041906696f);
// const __m256 c2 =  _mm256_set1_ps (0.166674897f);
// const __m256 c3 =  _mm256_set1_ps (0.499990642f);
// const __m256 c4 =  _mm256_set1_ps (0.999999762f);
// const __m256 c5 =  _mm256_set1_ps (1.000000000f);


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

// https://stackoverflow.com/questions/47025373/fastest-implementation-of-exponential-function-using-sse

/* max. rel. error = 1.72863156e-3 on [-87.33654, 88.72283] */
__m128 fast_exp_sse (__m128 x)
{
    __m128 t, f, e, p, r;
    __m128i i, j;
    __m128 l2e = _mm_set1_ps (1.442695041f);  /* log2(e) */
    __m128 c0  = _mm_set1_ps (0.3371894346f);
    __m128 c1  = _mm_set1_ps (0.657636276f);
    __m128 c2  = _mm_set1_ps (1.00172476f);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    t = _mm_mul_ps (x, l2e);             /* t = log2(e) * x */
#ifdef __SSE4_1__
    e = _mm_floor_ps (t);                /* floor(t) */
    i = _mm_cvtps_epi32 (e);             /* (int)floor(t) */
#else /* __SSE4_1__*/
    i = _mm_cvttps_epi32 (t);            /* i = (int)t */
    j = _mm_srli_epi32 (_mm_castps_si128 (x), 31); /* signbit(t) */
    i = _mm_sub_epi32 (i, j);            /* (int)t - signbit(t) */
    e = _mm_cvtepi32_ps (i);             /* floor(t) ~= (int)t - signbit(t) */
#endif /* __SSE4_1__*/
    f = _mm_sub_ps (t, e);               /* f = t - floor(t) */
    p = c0;                              /* c0 */
    p = _mm_mul_ps (p, f);               /* c0 * f */
    p = _mm_add_ps (p, c1);              /* c0 * f + c1 */
    p = _mm_mul_ps (p, f);               /* (c0 * f + c1) * f */
    p = _mm_add_ps (p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm_slli_epi32 (i, 23);          /* i << 23 */
    r = _mm_castsi128_ps (_mm_add_epi32 (j, _mm_castps_si128 (p))); /* r = p * 2^i*/
    return r;
}

// int main (void)
// {
//     union {
//         float f[4];
//         unsigned int i[4];
//     } arg, res;
//     double relerr, maxrelerr = 0.0;
//     int i, j;
//     __m128 x, y;

//     float start[2] = {-0.0f, 0.0f};
//     float finish[2] = {-87.33654f, 88.72283f};

//     for (i = 0; i < 2; i++) {

//         arg.f[0] = start[i];
//         arg.i[1] = arg.i[0] + 1;
//         arg.i[2] = arg.i[0] + 2;
//         arg.i[3] = arg.i[0] + 3;
//         do {
//             memcpy (&x, &arg, sizeof(x));
//             y = fast_exp_sse (x);
//             memcpy (&res, &y, sizeof(y));
//             for (j = 0; j < 4; j++) {
//                 double ref = exp ((double)arg.f[j]);
//                 relerr = fabs ((res.f[j] - ref) / ref);
//                 if (relerr > maxrelerr) {
//                     printf ("arg=% 15.8e  res=%15.8e  ref=%15.8e  err=%15.8e\n",
//                             arg.f[j], res.f[j], ref, relerr);
//                     maxrelerr = relerr;
//                 }
//             }
//             arg.i[0] += 4;
//             arg.i[1] += 4;
//             arg.i[2] += 4;
//             arg.i[3] += 4;
//         } while (fabsf (arg.f[3]) < fabsf (finish[i]));
//     }
//     printf ("maximum relative errror = %15.8e\n", maxrelerr);
//     return EXIT_SUCCESS;
// }

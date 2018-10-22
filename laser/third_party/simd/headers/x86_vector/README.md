# x86 instrinsics table

|     Header      |                                         Purpose|
|---------------- | ------------------------------------------------------------------------------------------|
| x86intrin.h     | Everything, including non-vector x86 instructions like _rdtsc().|
| mmintrin.h      | MMX (Pentium MMX!)|
| mm3dnow.h       | 3dnow! (K6-2) (deprecated)|
| xmmintrin.h     | SSE + MMX (Pentium 3, Athlon XP)|
| emmintrin.h     | SSE2 + SSE + MMX (Pentium 4, Athlon 64)|
| pmmintrin.h     | SSE3 + SSE2 + SSE + MMX (Pentium 4 Prescott, Athlon 64 San Diego)|
| tmmintrin.h     | SSSE3 + SSE3 + SSE2 + SSE + MMX (Core 2, Bulldozer)|
| popcntintrin.h  | POPCNT (Nehalem (Core i7), Phenom)|
| ammintrin.h     | SSE4A + SSE3 + SSE2 + SSE + MMX (AMD-only, starting with Phenom)|
| smmintrin.h     | SSE4_1 + SSSE3 + SSE3 + SSE2 + SSE + MMX (Penryn, Bulldozer)|
| nmmintrin.h     | SSE4_2 + SSE4_1 + SSSE3 + SSE3 + SSE2 + SSE + MMX (Nehalem (aka Core i7), Bulldozer)|
| wmmintrin.h     | AES (Core i7 Westmere, Bulldozer)|
| immintrin.h     | AVX, AVX2, AVX512, all SSE+MMX (except SSE4A and XOP), popcnt, BMI/BMI2, FMA|

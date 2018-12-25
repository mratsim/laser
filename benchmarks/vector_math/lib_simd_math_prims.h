/*
The MIT License (MIT)
Copyright (c) 2015 Jacques-Henri Jourdan <jourgun@gmail.com>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef SIMD_MATH_PRIMS_H
#define SIMD_MATH_PRIMS_H

#include<math.h>
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Workaround a lack of optimization in gcc */
float exp_cst1 = 2139095040.f;
float exp_cst2 = 0.f;

/* Relative error bounded by 1e-5 for normalized outputs
   Returns invalid outputs for nan inputs
   Continuous error */
inline float expapprox(float val) {
  union { int32_t i; float f; } xu, xu2;
  float val2, val3, val4, b;
  int32_t val4i;
  val2 = 12102203.1615614f*val+1065353216.f;
  val3 = val2 < exp_cst1 ? val2 : exp_cst1;
  val4 = val3 > exp_cst2 ? val3 : exp_cst2;
  val4i = (int32_t) val4;
  xu.i = val4i & 0x7F800000;
  xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
  b = xu2.f;

  /* Generated in Sollya with:
     > f=remez(1-x*exp(-(x-1)*log(2)),
               [|(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x|],
               [1.000001,1.999999], exp(-(x-1)*log(2)));
     > plot(exp((x-1)*log(2))/(f+x)-1, [1,2]);
     > f+x;
  */
  return
    xu.f * (0.509871020343597804469416f + b *
            (0.312146713032169896138863f + b *
             (0.166617139319965966118107f + b *
              (-2.19061993049215080032874e-3f + b *
               1.3555747234758484073940937e-2f))));
}

/* Absolute error bounded by 1e-6 for normalized inputs
   Returns a finite number for +inf input
   Returns -inf for nan and <= 0 inputs.
   Continuous error. */
inline float logapprox(float val) {
  union { float f; int32_t i; } valu;
  float exp, addcst, x;
  valu.f = val;
  exp = valu.i >> 23;
  /* 89.970756366f = 127 * log(2) - constant term of polynomial */
  addcst = val > 0 ? -89.970756366f : -(float)INFINITY;
  valu.i = (valu.i & 0x7FFFFF) | 0x3F800000;
  x = valu.f;


  /* Generated in Sollya using:
    > f = remez(log(x)-(x-1)*log(2),
            [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x,
              (x-1)*(x-2)*x*x*x|], [1,2], 1, 1e-8);
    > plot(f+(x-1)*log(2)-log(x), [1,2]);
    > f+(x-1)*log(2)
 */
  return
    x * (3.529304993f + x * (-2.461222105f +
      x * (1.130626167f + x * (-0.288739945f +
        x * 3.110401639e-2f))))
    + (addcst + 0.69314718055995f*exp);
}

/* Correct only in [-pi, pi]
   Absolute error bounded by 5e-5
   Continuous error */
inline float cosapprox(float val) {
  float val2 = val*val;
  /* Generated in Sollya using:
     > f = remez(cos(x)-1, [|x*x, x*x*x*x, x*x*x*x*x*x, x*x*x*x*x*x*x*x|],
                             [0.000001, pi], 1, 1e-8);
     > plot(f-cos(x)+1, [0, pi]);
     > f+1
 */
  return
    1.f + val2 *
    (-0.49985158205212948f + val2 *
     (4.151803521644389800e-2f + val2 *
      (-1.34229470252267675e-3f + val2 *
       1.89298648243010005378029e-5f)));
}

/* Correct only in [-pi, pi]
   Absolute error bounded by 6e-6
   Continuous error */
inline float sinapprox(float val) {
  float val2 = val*val;
  return
    val * (0.99997937679290771484375f + val2 *
           (-0.166624367237091064453125f + val2 *
            (8.30897875130176544189453125e-3f + val2 *
             (-1.92649182281456887722015380859375e-4f + val2 *
              2.147840177713078446686267852783203125e-6f))));
}

#ifdef __cplusplus
}
#endif

#endif

/* This program implements a show-case vector (vectorizable) double
   precision exponential with a 4 ulp error bound.

   Author: Christoph Lauter,

           Sorbonne Université - LIP6 - PEQUAN team.

   This program uses code generated using Sollya and Metalibm; see the
   licences and exception texts below.

   This program is

   Copyright 2014-2018 Christoph Lauter Sorbonne Université

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

   3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*

    This code was generated using non-trivial code generation commands
    of the Metalibm software program.

    Before using, modifying and/or integrating this code into other
    software, review the copyright and license status of this
    generated code. In particular, see the exception below.

    This generated program is partly or entirely based on a program
    generated using non-trivial code generation commands of the Sollya
    software program. See the copyright notice and exception text
    referring to that Sollya-generated part of this program generated
    with Metalibm below.

    Metalibm is

    Copyright 2008-2013 by

    Laboratoire de l'Informatique du Parallélisme,
    UMR CNRS - ENS Lyon - UCB Lyon 1 - INRIA 5668

    and by

    Laboratoire d'Informatique de Paris 6, equipe PEQUAN,
    UPMC Universite Paris 06 - CNRS - UMR 7606 - LIP6, Paris, France.

    Contributors: Christoph Quirin Lauter
                  (UPMC LIP6 PEQUAN formerly LIP/ENS Lyon)
                  christoph.lauter@lip6.fr

		  and

		  Olga Kupriianova
		  (UPMC LIP6 PEQUAN)
		  olga.kupriianova@lip6.fr

    Metalibm was formerly developed by the Arenaire project at Ecole
    Normale Superieure de Lyon and is now developed by Equipe PEQUAN
    at Universite Pierre et Marie Curie Paris 6.

    The Metalibm software program is free software; you can
    redistribute it and/or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option)
    any later version.

    Metalibm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with the Metalibm program; if not, write to the Free
    Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
    02111-1307, USA.

    This generated program is distributed WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    As a special exception, you may create a larger work that contains
    part or all of this software generated using Metalibm and
    distribute that work under terms of your choice, so long as that
    work isn't itself a numerical code generator using the skeleton of
    this code or a modified version thereof as a code skeleton.
    Alternatively, if you modify or redistribute this generated code
    itself, or its skeleton, you may (at your option) remove this
    special exception, which will cause this generated code and its
    skeleton and the resulting Metalibm output files to be licensed
    under the General Public licence (version 2) without this special
    exception.

    This special exception was added by the Metalibm copyright holders
    on November 20th 2013.

*/



/*
    This code was generated using non-trivial code generation commands of
    the Sollya software program.

    Before using, modifying and/or integrating this code into other
    software, review the copyright and license status of this generated
    code. In particular, see the exception below.

    Sollya is

    Copyright 2006-2013 by

    Laboratoire de l'Informatique du Parallelisme, UMR CNRS - ENS Lyon -
    UCB Lyon 1 - INRIA 5668,

    Laboratoire d'Informatique de Paris 6, equipe PEQUAN, UPMC Universite
    Paris 06 - CNRS - UMR 7606 - LIP6, Paris, France

    and by

    Centre de recherche INRIA Sophia-Antipolis Mediterranee, equipe APICS,
    Sophia Antipolis, France.

    Contributors Ch. Lauter, S. Chevillard, M. Joldes

    christoph.lauter@ens-lyon.org
    sylvain.chevillard@ens-lyon.org
    joldes@lass.fr

    The Sollya software is a computer program whose purpose is to provide
    an environment for safe floating-point code development. It is
    particularily targeted to the automatized implementation of
    mathematical floating-point libraries (libm). Amongst other features,
    it offers a certified infinity norm, an automatic polynomial
    implementer and a fast Remez algorithm.

    The Sollya software is governed by the CeCILL-C license under French
    law and abiding by the rules of distribution of free software.  You
    can use, modify and/ or redistribute the software under the terms of
    the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
    following URL "http://www.cecill.info".

    As a counterpart to the access to the source code and rights to copy,
    modify and redistribute granted by the license, users are provided
    only with a limited warranty and the software's author, the holder of
    the economic rights, and the successive licensors have only limited
    liability.

    In this respect, the user's attention is drawn to the risks associated
    with loading, using, modifying and/or developing or reproducing the
    software by the user in light of its specific status of free software,
    that may mean that it is complicated to manipulate, and that also
    therefore means that it is reserved for developers and experienced
    professionals having in-depth computer knowledge. Users are therefore
    encouraged to load and test the software's suitability as regards
    their requirements in conditions enabling the security of their
    systems and/or data to be ensured and, more generally, to use and
    operate it in the same conditions as regards security.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.

    The Sollya program is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.

    This generated program is distributed WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    As a special exception, you may create a larger work that contains
    part or all of this software generated using Sollya and distribute
    that work under terms of your choice, so long as that work isn't
    itself a numerical code generator using the skeleton of this code or a
    modified version thereof as a code skeleton.  Alternatively, if you
    modify or redistribute this generated code itself, or its skeleton,
    you may (at your option) remove this special exception, which will
    cause this generated code and its skeleton and the resulting Sollya
    output files to be licensed under the CeCILL-C licence without this
    special exception.

    This special exception was added by the Sollya copyright holders in
    version 4.1 of Sollya.

*/

#include <stdint.h>
#include "exp.h"

/* Two caster types */
typedef union _dblcast {
  double   d;
  uint64_t i;
} dblcast;

typedef union {
  int64_t l;
  double d;
} db_number;

/* Macro implementations of some double-double operations */
#define Add12(s, r, a, b)                       \
  {double _z, _a=a, _b=b;                       \
    s = _a + _b;                                \
    _z = s - _a;                                \
    r = _b - _z;   }

#define Mul12(rh,rl,u,v)                                \
  {                                                     \
    CONST double c  = 134217729.; /* 2^27 +1 */         \
    double up, u1, u2, vp, v1, v2;                      \
    double _u =u, _v=v;                                 \
                                                        \
    up = _u*c;        vp = _v*c;                        \
    u1 = (_u-up)+up;  v1 = (_v-vp)+vp;                  \
    u2 = _u-u1;       v2 = _v-v1;                       \
                                                        \
    *rh = _u*_v;                                        \
    *rl = (((u1*v1-*rh)+(u1*v2))+(u2*v1))+(u2*v2);      \
  }

#define Mul122(resh,resl,a,bh,bl)               \
  {                                             \
    double _t1, _t2, _t3, _t4;                  \
                                                \
    Mul12(&_t1,&_t2,(a),(bh));                  \
    _t3 = (a) * (bl);                           \
    _t4 = _t2 + _t3;                            \
    Add12((*(resh)),(*(resl)),_t1,_t4);         \
  }

#define Mul22(zh,zl,xh,xl,yh,yl)                        \
  {                                                     \
    double mh, ml;                                      \
                                                        \
    CONST double c = 134217729.;                        \
    double up, u1, u2, vp, v1, v2;                      \
                                                        \
    up = (xh)*c;        vp = (yh)*c;                    \
    u1 = ((xh)-up)+up;  v1 = ((yh)-vp)+vp;              \
    u2 = (xh)-u1;       v2 = (yh)-v1;                   \
                                                        \
    mh = (xh)*(yh);                                     \
    ml = (((u1*v1-mh)+(u1*v2))+(u2*v1))+(u2*v2);        \
                                                        \
    ml += (xh)*(yl) + (xl)*(yh);                        \
    *zh = mh+ml;                                        \
    *zl = mh - (*zh) + ml;                              \
  }

/* Need fabs */
double fabs(double);


/* Some constants */
#define LOG2_E    1.442695040888963407359924681001892137426645954153
#define LOG_2_HI  0.693147180559890330187045037746429443359375
#define LOG_2_LO  5.4979230187083711552420206887059365096458163346682e-14
#define SHIFTER   6755399441055744.0

/* A metalibm generated function for the callout */
#define f_approx_exp_arg_red_coeff_0h 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000e+00
#define f_approx_exp_arg_red_coeff_1h 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000e+00
#define f_approx_exp_arg_red_coeff_2h 5.00000000000032307490016592055326327681541442871093750000000000000000000000000000e-01
#define f_approx_exp_arg_red_coeff_3h 1.66666666664336909908783468381443526595830917358398437500000000000000000000000000e-01
#define f_approx_exp_arg_red_coeff_4h 4.16666661063678778198493546369718387722969055175781250000000000000000000000000000e-02
#define f_approx_exp_arg_red_coeff_5h 8.33337739276391979703628720699271070770919322967529296875000000000000000000000000e-03
#define f_approx_exp_arg_red_coeff_6h 1.39156772666044516173489142829566844739019870758056640625000000000000000000000000e-03


STATIC INLINE void f_approx_exp_arg_red(double * RESTRICT f_approx_exp_arg_red_resh, double * RESTRICT f_approx_exp_arg_red_resm, double x) {




  double f_approx_exp_arg_red_t_1_0h;
  double f_approx_exp_arg_red_t_2_0h;
  double f_approx_exp_arg_red_t_3_0h;
  double f_approx_exp_arg_red_t_4_0h;
  double f_approx_exp_arg_red_t_5_0h;
  double f_approx_exp_arg_red_t_6_0h;
  double f_approx_exp_arg_red_t_7_0h;
  double f_approx_exp_arg_red_t_8_0h;
  double f_approx_exp_arg_red_t_9_0h;
  double f_approx_exp_arg_red_t_10_0h;
  double f_approx_exp_arg_red_t_11_0h;
  double f_approx_exp_arg_red_t_12_0h;
  double f_approx_exp_arg_red_t_13_0h, f_approx_exp_arg_red_t_13_0m;



  f_approx_exp_arg_red_t_1_0h = f_approx_exp_arg_red_coeff_6h;
  f_approx_exp_arg_red_t_2_0h = f_approx_exp_arg_red_t_1_0h * x;
  f_approx_exp_arg_red_t_3_0h = f_approx_exp_arg_red_coeff_5h + f_approx_exp_arg_red_t_2_0h;
  f_approx_exp_arg_red_t_4_0h = f_approx_exp_arg_red_t_3_0h * x;
  f_approx_exp_arg_red_t_5_0h = f_approx_exp_arg_red_coeff_4h + f_approx_exp_arg_red_t_4_0h;
  f_approx_exp_arg_red_t_6_0h = f_approx_exp_arg_red_t_5_0h * x;
  f_approx_exp_arg_red_t_7_0h = f_approx_exp_arg_red_coeff_3h + f_approx_exp_arg_red_t_6_0h;
  f_approx_exp_arg_red_t_8_0h = f_approx_exp_arg_red_t_7_0h * x;
  f_approx_exp_arg_red_t_9_0h = f_approx_exp_arg_red_coeff_2h + f_approx_exp_arg_red_t_8_0h;
  f_approx_exp_arg_red_t_10_0h = f_approx_exp_arg_red_t_9_0h * x;
  f_approx_exp_arg_red_t_11_0h = f_approx_exp_arg_red_coeff_1h + f_approx_exp_arg_red_t_10_0h;
  f_approx_exp_arg_red_t_12_0h = f_approx_exp_arg_red_t_11_0h * x;
  Add12(f_approx_exp_arg_red_t_13_0h,f_approx_exp_arg_red_t_13_0m,f_approx_exp_arg_red_coeff_0h,f_approx_exp_arg_red_t_12_0h);
  *f_approx_exp_arg_red_resh = f_approx_exp_arg_red_t_13_0h; *f_approx_exp_arg_red_resm = f_approx_exp_arg_red_t_13_0m;


}

STATIC CONST double f_approx_twoPower_Index_Hi[32] = {
  1,
  1.021897148654116627,
  1.0442737824274137548,
  1.067140400676823697,
  1.0905077326652576897,
  1.114386742595892432,
  1.1387886347566915646,
  1.1637248587775774755,
  1.1892071150027210269,
  1.2152473599804689552,
  1.241857812073484002,
  1.2690509571917332199,
  1.2968395546510096406,
  1.3252366431597413232,
  1.3542555469368926513,
  1.3839098819638320226,
  1.4142135623730951455,
  1.4451808069770466503,
  1.4768261459394993462,
  1.5091644275934228414,
  1.542210825407940744,
  1.5759808451078864966,
  1.6104903319492542835,
  1.6457554781539649458,
  1.681792830507429004,
  1.718619298122477934,
  1.7562521603732994535,
  1.794709075003107168,
  1.8340080864093424307,
  1.8741676341102999626,
  1.9152065613971474,
  1.9571441241754001794
};

STATIC CONST double f_approx_twoPower_Index_Mi[32] = {
  0,
  5.109225028973443893e-17,
  8.551889705537964892e-17,
  -7.899853966841582122e-17,
  -3.046782079812471147e-17,
  1.0410278456845570955e-16,
  8.912812676025407777e-17,
  3.8292048369240934987e-17,
  3.982015231465646111e-17,
  -7.71263069268148813e-17,
  4.658027591836936791e-17,
  2.667932131342186095e-18,
  2.5382502794888314959e-17,
  -2.858731210038861373e-17,
  7.700948379802989461e-17,
  -6.770511658794786287e-17,
  -9.66729331345291345e-17,
  -3.023758134993987319e-17,
  -3.4839945568927957958e-17,
  -1.016455327754295039e-16,
  7.949834809697620856e-17,
  -1.013691647127830398e-17,
  2.470719256979788785e-17,
  -1.0125679913674772604e-16,
  8.19901002058149652e-17,
  -1.851380418263110988e-17,
  2.960140695448873307e-17,
  1.822745842791208677e-17,
  3.283107224245627203e-17,
  -6.122763413004142561e-17,
  -1.0619946056195962638e-16,
  8.960767791036667767e-17
};

#define f_approx_argred_log2_of_base_times_two_to_w 4.616624130844682838e1
#define f_approx_argred_minus_logbase_of_2_times_two_to_minus_w_hi -2.1660849392498290195e-2
#define f_approx_argred_minus_logbase_of_2_times_two_to_minus_w_mi -7.24702129326968612e-19
#define f_approx_argred_shifter 6755399441055744.0
#define f_approx_argred_w 5
#define f_approx_argred_idx_mask 31ull
#define f_approx_argred_lower_32_bits 0xffffffffull

STATIC INLINE void scalar_exp_callout_inner(double * RESTRICT res_resh, double * RESTRICT res_resm, double xh) {
  double zh;
  double poly_resh, poly_resm;

  double t;
  double shifted_t;
  double mAsDouble;
  db_number argRedCaster;
  int mAsInt;
  int E;
  int E1;
  int E2;
  int idx;
  double rescaled_m_hi;
  double rescaled_m_mi;
  double table_hi;
  double table_mi;
  double tableTimesPoly_hi;
  double tableTimesPoly_mi;
  db_number twoE1;
  db_number twoE2;
  double twoE1tablePoly_hi;
  double twoE1tablePoly_mi;


  t = xh * f_approx_argred_log2_of_base_times_two_to_w;
  shifted_t = t + f_approx_argred_shifter;
  mAsDouble = shifted_t - f_approx_argred_shifter;
  argRedCaster.d = shifted_t;
  mAsInt = (int) (argRedCaster.l & f_approx_argred_lower_32_bits);
  E = mAsInt >> f_approx_argred_w;
  E1 = E >> 1;
  E2 = E - E1;
  idx = mAsInt & f_approx_argred_idx_mask;
  Mul122(&rescaled_m_hi, &rescaled_m_mi, mAsDouble, f_approx_argred_minus_logbase_of_2_times_two_to_minus_w_hi, f_approx_argred_minus_logbase_of_2_times_two_to_minus_w_mi);
  zh = (xh + rescaled_m_hi) + rescaled_m_mi;

  f_approx_exp_arg_red(&poly_resh, &poly_resm, zh);

  table_hi = f_approx_twoPower_Index_Hi[idx];
  table_mi = f_approx_twoPower_Index_Mi[idx];
  Mul22(&tableTimesPoly_hi,&tableTimesPoly_mi,table_hi,table_mi,poly_resh,poly_resm);
  twoE1.l = E1 + 1023ll;
  twoE1.l <<= 52;
  twoE2.l = E2 + 1023ll;
  twoE2.l <<= 52;
  twoE1tablePoly_hi = twoE1.d * tableTimesPoly_hi;
  twoE1tablePoly_mi = twoE1.d * tableTimesPoly_mi;
  *res_resh = twoE2.d * twoE1tablePoly_hi;
  *res_resm = twoE2.d * twoE1tablePoly_mi;

}

/* A scalar exponential for the callout */
STATIC INLINE double scalar_exp_callout(double x) {
  dblcast xdb, xAbsdb;
  double yh, yl, twoM600, two600;

  xdb.d = x;
  xAbsdb.i = xdb.i & 0x7fffffffffffffffull;
  if (xAbsdb.i >= 0x7ff0000000000000ull) {
    /* If we are here, we have an Inf or a Nan */
    if (xAbsdb.i == 0x7ff0000000000000ull) {
      /* Here, the input is an Inf */
      if (xdb.i >> 63) {
	/* x = -Inf, return 0 */
	return 0.0;
      }
      /* x = +Inf, return +Inf */
      return x;
    }

    /* Here, the input is a NaN */
    return 1.0 + x;
  }

  /* Here, the input is real.

     Start by checking if we have evident under- or overflow.

     We have evident underflow if x <= -746.0
     and     evident overflow  if x >= 711.0.
  */
  if (x <= -746.0) {
    /* Return a completely underflowed result */
    twoM600 = 2.4099198651028841177407500347125089364310049545099e-181;

    return twoM600 * twoM600;
  }
  if (x >= 711.0) {
    /* Return a completely overflowed result */
    two600 = 4.1495155688809929585124078636911611510124462322424e180;

    return two600 * two600;
  }

  /* Here, the input will not provoke any huge overflow or underflow
     but there might still be some under- or overflow.

     Now check if x is that small in magnitude that returning 1.0 + x
     suffices to well approximate the exponential (up to a relative
     error of 2^-53). This is surely the case when abs(x) <= 0.75 *
     2^-26.
  */
  if (fabs(x) <= 1.11758708953857421875e-8) {
    return 1.0 + x;
  }

  /* Here, the input is real. There might still be some slight under-
     or overflow on output.

     Just use a metalibm generated function.

  */
  scalar_exp_callout_inner(&yh, &yl, x);

  return yh + yl;
}

/* A vector exponential callout */
STATIC INLINE void vector_exp_callout(double * RESTRICT y, CONST double * RESTRICT x) {
  int i;

  for (i=0;i<VECTOR_LENGTH;i++) {
    y[i] = scalar_exp_callout(x[i]);
  }
}

/* Generated polynomial for vector exponential */

#define vector_exp_poly_coeff_0h 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000e+00
#define vector_exp_poly_coeff_1h 1.00000000000000643929354282590793445706367492675781250000000000000000000000000000e+00
#define vector_exp_poly_coeff_2h 4.99999999999983513188084316425374709069728851318359375000000000000000000000000000e-01
#define vector_exp_poly_coeff_3h 1.66666666665578222517041240280377678573131561279296875000000000000000000000000000e-01
#define vector_exp_poly_coeff_4h 4.16666666679390979011188278491317760199308395385742187500000000000000000000000000e-02
#define vector_exp_poly_coeff_5h 8.33333338463836288678709962596258264966309070587158203125000000000000000000000000e-03
#define vector_exp_poly_coeff_6h 1.38888885906261988316401367882235717843286693096160888671875000000000000000000000e-03
#define vector_exp_poly_coeff_7h 1.98411714150174687447750199176255136990221217274665832519531250000000000000000000e-04
#define vector_exp_poly_coeff_8h 2.48018422092243362301385717350044046725088264793157577514648437500000000000000000e-05
#define vector_exp_poly_coeff_9h 2.76397570196414793205260534980638453816936817020177841186523437500000000000000000e-06
#define vector_exp_poly_coeff_10h 2.75111392508451855531313940197990497438240709016099572181701660156250000000000000e-07


STATIC INLINE void vector_exp_poly(double * RESTRICT vector_exp_poly_resh, double x) {




  double vector_exp_poly_t_1_0h;
  double vector_exp_poly_t_2_0h;
  double vector_exp_poly_t_3_0h;
  double vector_exp_poly_t_4_0h;
  double vector_exp_poly_t_5_0h;
  double vector_exp_poly_t_6_0h;
  double vector_exp_poly_t_7_0h;
  double vector_exp_poly_t_8_0h;
  double vector_exp_poly_t_9_0h;
  double vector_exp_poly_t_10_0h;
  double vector_exp_poly_t_11_0h;
  double vector_exp_poly_t_12_0h;
  double vector_exp_poly_t_13_0h;
  double vector_exp_poly_t_14_0h;
  double vector_exp_poly_t_15_0h;
  double vector_exp_poly_t_16_0h;
  double vector_exp_poly_t_17_0h;
  double vector_exp_poly_t_18_0h;
  double vector_exp_poly_t_19_0h;
  double vector_exp_poly_t_20_0h;
  double vector_exp_poly_t_21_0h;



  vector_exp_poly_t_1_0h = vector_exp_poly_coeff_10h;
  vector_exp_poly_t_2_0h = vector_exp_poly_t_1_0h * x;
  vector_exp_poly_t_3_0h = vector_exp_poly_coeff_9h + vector_exp_poly_t_2_0h;
  vector_exp_poly_t_4_0h = vector_exp_poly_t_3_0h * x;
  vector_exp_poly_t_5_0h = vector_exp_poly_coeff_8h + vector_exp_poly_t_4_0h;
  vector_exp_poly_t_6_0h = vector_exp_poly_t_5_0h * x;
  vector_exp_poly_t_7_0h = vector_exp_poly_coeff_7h + vector_exp_poly_t_6_0h;
  vector_exp_poly_t_8_0h = vector_exp_poly_t_7_0h * x;
  vector_exp_poly_t_9_0h = vector_exp_poly_coeff_6h + vector_exp_poly_t_8_0h;
  vector_exp_poly_t_10_0h = vector_exp_poly_t_9_0h * x;
  vector_exp_poly_t_11_0h = vector_exp_poly_coeff_5h + vector_exp_poly_t_10_0h;
  vector_exp_poly_t_12_0h = vector_exp_poly_t_11_0h * x;
  vector_exp_poly_t_13_0h = vector_exp_poly_coeff_4h + vector_exp_poly_t_12_0h;
  vector_exp_poly_t_14_0h = vector_exp_poly_t_13_0h * x;
  vector_exp_poly_t_15_0h = vector_exp_poly_coeff_3h + vector_exp_poly_t_14_0h;
  vector_exp_poly_t_16_0h = vector_exp_poly_t_15_0h * x;
  vector_exp_poly_t_17_0h = vector_exp_poly_coeff_2h + vector_exp_poly_t_16_0h;
  vector_exp_poly_t_18_0h = vector_exp_poly_t_17_0h * x;
  vector_exp_poly_t_19_0h = vector_exp_poly_coeff_1h + vector_exp_poly_t_18_0h;
  vector_exp_poly_t_20_0h = vector_exp_poly_t_19_0h * x;
  vector_exp_poly_t_21_0h = vector_exp_poly_coeff_0h + vector_exp_poly_t_20_0h;
  *vector_exp_poly_resh = vector_exp_poly_t_21_0h;


}

/* A vector exponential */
void vector_exp(double * RESTRICT yArg, CONST double * RESTRICT xArg) {
  int i;
  int okaySlots;
  double * RESTRICT y;
  CONST double * RESTRICT x;
  double shiftedXTLog2e, eDouble, t, r;
  int E;
  double p;
  dblcast twoE;

  /* Assume alignment */
#ifdef NO_ASSUME_ALIGNED
  x = xArg;
  y = yArg;
#else
  x = __builtin_assume_aligned(xArg, VECTOR_LENGTH * __alignof__(double));
  y = __builtin_assume_aligned(yArg, VECTOR_LENGTH * __alignof__(double));
#endif

  /* Check if we can handle all inputs */
  okaySlots = 0;
  for (i=0;i<VECTOR_LENGTH;i++) {
    okaySlots += (fabs(x[i]) < 700.0);
  }

  /* Perform a callout if we cannot handle the input in one slot */
  if (okaySlots != VECTOR_LENGTH) {
    vector_exp_callout(yArg, xArg);
    return;
  }

  /* Here we know that all inputs are real and do not provoke under-
     or overflow in output
  */
  for (i=0;i<VECTOR_LENGTH;i++) {
    shiftedXTLog2e = x[i] * LOG2_E + SHIFTER;
    eDouble = shiftedXTLog2e - SHIFTER;
    E = (int) eDouble;
    t = x[i] - eDouble * LOG_2_HI; /* exact: trailing bits of constant 0, Sterbenz */
    r = t - eDouble * LOG_2_LO;
    vector_exp_poly(&p,r);
    twoE.i = E + 1023;
    twoE.i <<= 52;
    y[i] = twoE.d * p;
  }
}

Optimising exponential and logarithm functions is critical for machine learning as many activations and loss functions are relying on those especially:

  - Negative log-likelihood and cross-entropy loss
  - sigmoid
  - softmax
  - softmax cross-entropy (using the log-sum-exp techniques)

The default implementation in `<math.h>` are very slow. The usual way to implement them is via polynomial approximation.

## Literature

- Taylor Expansion:
  - https://en.wikipedia.org/wiki/Taylor_series#Exponential_function
  - http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm
- Remez algorithm and Chebyshev approximation:
  - https://en.wikipedia.org/wiki/Remez_algorithm
  - http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm
- Euler's continued fractions: https://en.wikipedia.org/wiki/Euler%27s_continued_fraction_formula#The_exponential_function
- Padé approximant: https://en.wikipedia.org/wiki/Padé_table#An_example_–_the_exponential_function
- Using the fact that `e^x = 2^x/ln(2)` and split x into an integer portion to be computed with shifts and a fractional part
- Range reduction
  Example using the previous 2 to implement exp(x) on FPGA: http://perso.citi-lab.fr/fdedinec/recherche/publis/2005-FPT.pdf
- Argument reduction `e^x = 2^E * 2^(k-E) * e ^r` with `2^(k-E)` coming from a table of size `2^w` and `e^r` to approximate (`r= x log2 e − k`): https://hal.sorbonne-universite.fr/hal-01084726/document
- Tang's Exponential function:
  - Table-Driven Implementation of the Exponential Function in IEEE Floating-Point Arithmetic: http://soc.knu.ac.kr/video_lectures/15_4.pdf
  - http://users.kymp.net/p501474a/Orbiter/CmpExpLog.pdf
  - Formal Verif: http://gappa.gforge.inria.fr/doc/ch05s02.html
  - Another formal Verif: https://www.collectionscanada.gc.ca/obj/s4/f2/dsk3/ftp05/MQ64057.pdf
- Schraudolf approximation using the logarithmic nature of IEEE754 floating points: https://nic.schraudolph.org/pubs/Schraudolph99.pdf
- Cody & Waite algorithm: http://arith.cs.ucla.edu/publications/LogExp-TC04.pdf
  - Formal Proof: http://fastrelax.gforge.inria.fr/files/fastrelax2015GuillaumeMelquiond.pdf
- CORDIC algorithm
  - http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm
- Twofold exp and log:
  - http://soc.knu.ac.kr/video_lectures/15_4.pdf
  - https://arxiv.org/pdf/1502.05216.pdf

## Reference discussions

- https://math.stackexchange.com/questions/55830/how-to-calculate-ex-with-a-standard-calculator
- https://stackoverflow.com/questions/6984440/approximate-ex

### with SIMD code:
- https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
- https://stackoverflow.com/questions/47025373/fastest-implementation-of-exponential-function-using-sse

## Libraries
- http://gruntthepeon.free.fr/ssemath/
- http://software-lisc.fbk.eu/avx_mathfun/
- https://github.com/herumi/fmath
- https://github.com/jhjourdan/SIMD-math-prims
  and http://gallium.inria.fr/blog/fast-vectorizable-math-approx/
- [A new open-source SIMD vector libm fully implemented
with high-level scalar C](https://hal.archives-ouvertes.fr/hal-01511131/document), 2016, Lauter
  https://gitlab.com/cquirin/vector-libm
- https://github.com/shibatch/sleef
- https://github.com/gnuradio/volk
- https://github.com/VcDevel/Vc
- https://github.com/p12tic/libsimdpp
- https://github.com/projectNe10/Ne10

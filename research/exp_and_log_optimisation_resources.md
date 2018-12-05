Optimising exponential and logarithm functions is critical for machine learning as many activations and loss functions are relying on those especially:

  - Negative log-likelihood and cross-entropy loss
  - sigmoid
  - softmax
  - softmax cross-entropy (using the log-sum-exp techniques)

The default implementation in `<math.h>` are very slow. The usual way to implement them is via polynomial approximation.

## Literature

- Taylor Expansion: https://en.wikipedia.org/wiki/Taylor_series#Exponential_function
- Remez algorithm and Chebyshev approximation: https://en.wikipedia.org/wiki/Remez_algorithm
- Euler's continued fractions: https://en.wikipedia.org/wiki/Euler%27s_continued_fraction_formula#The_exponential_function
- Padé approximant: https://en.wikipedia.org/wiki/Padé_table#An_example_–_the_exponential_function
- Using the fact that `e^x = 2^x/ln(2)` and split x into an integer portion to be computed with shifts and a fractional part
- Range reduction
  Example using the previous 2 to implement exp(x) on FPGA: http://perso.citi-lab.fr/fdedinec/recherche/publis/2005-FPT.pdf
- Schraudolf approximation using the logarithmic nature of IEEE754 floating points: https://nic.schraudolph.org/pubs/Schraudolph99.pdf
- Cody & Waite algorithm: http://arith.cs.ucla.edu/publications/LogExp-TC04.pdf
- CORDIC algorithm

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

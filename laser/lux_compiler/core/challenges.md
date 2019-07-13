# Challenges for Lux

This document highlights challenges where
- the desire for a high-level, ergonomic language
- the desired use-case
- the implementation "details"

are clashing, making Lux an interesting engineering challenge.

Here are some challenges for the DSL:

## Gauss-Seidel Smoother

Example in C, from [OpenMP 5.0 talk](images/gauss_seidel_smoother)

```C
// A is a NxN matrix.
for (int t = 0; t < TimeStep ; ++t)
  for (int i = 1; i < N - 1; ++i)
    for (int j = 1; j < N - 1; ++j)
      A[i][j] = 0.25 * (A[i][j-1] * // left
                        A[i][j+1] * // right
                        A[i-1][j] * // top
                        A[i+1][j]); // bottom
```

The naive way cannot specify the time loop.

```Nim
proc gauss_seidel_smoother(A: var Function) =
  A[i, j] = 0.25 * (A[i][j-1] * # left
                      A[i][j+1] * # right
                      A[i-1][j] * # top
                      A[i+1][j]); # bottom
```

A potential solution would be to introduce the ``recur`` proc to express recurrence

```Nim
proc gauss_seidel_smoother(A: var Function, T: Domain) =
  A[i, j] = 0.25 * (A[i][j-1] * # left
                      A[i][j+1] * # right
                      A[i-1][j] * # top
                      A[i+1][j]); # bottom
  A.recur(T)
```

## Reduced domain

From [MLIR simplified polyhedral dialect](https://github.com/tensorflow/mlir/blob/83ff81bfd9d382852d0302ab2a234feb2e938fc7/g3doc/RationaleSimplifiedPolyhedralForm.md#proposal-simplified-polyhedral-form)

From the base example
```C
void simple_example(...) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
        float tmp = X[i,j]    // S1
        A[i,j] = tmp + 1      // S2
        B[i,j] = tmp * 42     // S3
      }
  }
}
```

Make S3 ignore the 10 outer points i.e. in C pseudocode
```C
void reduced_domain_example(...) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float tmp = X[i,j]    // S1
      A[i,j] = tmp + 1      // S2
      if (10 <= i < N-10) || (10 <= j < N-10){
        B[i,j] = tmp * 42     // S3
      }
    }
  }
}
```

A potential solution would be to introduce the ``where`` proc to express a conditional definition

```nim
proc reduced_domain_example(A, B: var Func, X: Func) =
  var i, j: Domain
  A[i, j] = X[i, j] + 1
  B[i, j] = X[i, j] * 42
  B.where((10 <= i < N-10) and (10 <= j < N-10)) # Pseudo code, we can't do 0<x<10
```

## Domain of composed functions

Within a generator, we must distinguish iteration domains

Let's take a generator that initializes a square tensor
with its elements to the sum of their indices.

```Nim
proc foo(n: Func): Func =
  # Remember everything is a Func
  # This allows n to be the result of another Lux function
  # for composition

  var i = domain(0, n)
  var j = domain(0, n)

  result[i, j] = i+j
```

Now suppose someone else reuses this function, for they it's a blackbox,
also they prefer to use x and y for indexing.

They want to chain it with the following double function

```Nim
proc double(n: Func): Func =
  var x = domain(0, n)
  var y = domain(0, n)

  result[x, y] = n[x, y] * 2
```

to create

```Nim
let N = newImm(10) # create a function that always return 10 (immediate)
let dbl = double(foo(N))
```

and we want the compiler to auto-fuse the iteration domains into

```Nim
result[i, j] = (i+j) * 2
```
or
```Nim
result[x, y] = (x+y) * 2
```

So we need something akin to an equality functions that respects:
- i != j
- x != y
- i == x
- j == y
to handle this case.

## Recurrent neural network

Implementing RNN, and especially Fused Stacked RNN is very error prone.

For example this is the equation of a GRU cell:

```
r = sigmoid(W_{ir} x + b_{ir} + W_{hr} h + b_{hr})     // reset gate
z = sigmoid(W_{iz} x + b_{iz} + W_{hz} h + b_{hz})     // update gate
n = tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn}))  // candidate hidden state
h' = (1 - z) * n + z * h                               // new hidden state
```

with indices
```
ir: input-reset
hr: hidden-reset
in: input-candidate
hn: hidden-candidate
```
and W for Weight and b for bias

Hopefully in Lux DSL, it would not be too much more complicated than

```Nim
proc gru_cell(
        x: Func, hidden: var Func,
        Wir, bir, Whr, bhr: Func
        Wiz, biz, Whz, bhz: Func
        Win, bin, Whn, bhn: Func)
  # x of shape [batch_size, features]
  # hidden of shape [batch_size, hidden_size]
  # Gates Weights for input Wi of shape [hidden_size, features]
  # Recurrent Weights for hidden Wh of shape [hidden_size, hidden_size]
  # Biases of shape [hidden_size] and broadcasted

  # Note that W*x is in practice x*W.transpose
  # Frameworks do not have a standardized way to store weights
  # and order the linear/dense/affine/matrixmultiplication layer

  let b = domain(x.shape(0))
  let f = domain(x.shape(1))
  let h = domain(hidden.shape(1))
  let w = domain(hidden.shape(0)) # This also iterates on hidden size

  var r, z, n: Func

  r[b,h] = sigmoid(
            Wir[h,f] * x[b,f]      + bir[h] +
            Whr[w,h] * hidden[b,h] + bhr[h]
          )
  z[b,h] = sigmoid(
            Wiz[h,f] * x[b,f]      + biz[h] +
            Whz[w,h] * hidden[b,h] + bhz[h]
          )
  n[b,h] = tanh(
            Win[h,f] * x[b,f]      + bin[h] +
            r * (Whn[w,h] * hidden[b,h] + bhn[h]
          )
  h[b,h] = (1 - z[b,h]) * n[b,h] + z[b,h] * h[b,h]
```

And now we need to add a timestep/sequence and a layer dimension
for full blown GRU

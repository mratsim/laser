# Overview of Lux

Lux is a Domain-Specific Language (DSL) embedded in Nim.
It provides a high-level expressive language to represent
algorithms that work on multi-dimensional constructs,
scalars, vectors, matrices and their generalisations, ndarrays/tensors.

Lux allows you to schedule those algorithms and compose their schedules
to best use your hardware: parallelizing, vectorizing, unrolling, tiling, prefetching, ...

In the rest of the overview we will use the term tensor interchangeably with
  - vectors, 1d tensors indexed like A[i]
  - matrices, 2d tensors indexed like B[i, j]
  - ndarrays, n-d tensors indexed like C[i, j, k, l]

## Types and data structures

Let's dive into Lux from the high-level to the nitty gritty details.

But first we need to introduce `LuxNode`. `LuxNode` is a basic block in Lux,
the low-level language.
It can represent a scalar parameter, a tensor, an addition, anything that can
or will be used in Lux, be it a function or a loop or an integer. As an user,
LuxNodes are completely hidden from you.

Lux is separated in a front-end, a nice high-level DSL language that generates LuxNodes
and specifies scheduling directives (like parallelize this computation),
and a backend that compiles LuxNodes to a target, currently Nim code.

The high-level language is functional in nature: tensors, scalars and intermediate computations
are all symbolic functions. This allows composability.

Let's review how Lux code is organised

### Generators

At the top-level of Lux we have generators. Generators are Nim procedures with
abstract ``Function`` inputs and abstract ``Function`` output.

#### Note on Function vs Tensors

The equivalent of Lux function on concrete scalars or tensors is called a kernel.

For example, matrix transposition generator, function and kernels are is:

```Nim
proc transpose(A: Function): Function =
  # Generator for the transposition function
  var i, j: Domain
  var B: Function

  # Definition of the result function
  B[j, i] = A[i, j]

  return B # Transposition function

proc transpose(A: Tensor[float32]): Tensor[float32] =
  # Transposition kernel
  let dim0 = A.shape[0]
  let dim1 = A.shape[1]

  result = newTensor[float32](dim0, dim1)

  for j in 0 ..< dim1:
    for i in 0 ..< dim0:
      result[j, i] = A[i, j]
```

It may be confusing, so here are some properties of Lux `Function`:
  - They can be chained without intermediate tensors as they represent an abstract computation.
  - If they are not used, there will be no allocation

For example:
```Nim
proc divmod(A, B: Function): tuple[D, M: Function] =
  # Generator for element-wise division+modulo
  var i: Domain
  result.D[i] = A[i] div B[i]
  result.M[i] = A[i] mod M[i]

proc div(A, B: Function): Function =
  # Generator for element-wise division
  let (D, _) = divmod(A, B)
  result = D

proc mod(A, B: Function): Function =
  # Generator for element-wise division
  let (_, M) = divmod(A, B)
  result = M
```

If only ``mod`` is called the code for division will never be generated.
There will be no allocation for division result nor loop to compute it.

### Functions

We had a short preview of what is a function, let's dive in details into them

A function represents a symbolic computation done on tensor elements and scalars.
For composability, tensors are also represented by functions.

#### Function Definition

A function definition requires point/element coordinates as parameters and an expression.
For example, a function that set all elements of a 2d matrix to 1 would be

```Nim
A[i, j] = 1
```

Functions can be composed, for example to double them

```Nim
B[i, j] = A[i, j] * 2
```

The syntax is similar to Einstein summation and implicit for-loops are inferred
from the indices used.

For example a matrix multiplication would be:
```Nim
C[i, j] += A[i, k] * B[k, j]
```

The loops i, j, k are implicit.

### Iteration domain

The iteration variables are called the ``domain`` of the function.
i, j are part of the function iteration domain as they appear on the left-hand side.
k is a contraction domain as it doesn't appear on the left-hand side.

### In-place updates

Lux functions are able to represent in-place updates as well, any in-place operator can be used

For example:

```
A[i, j] = 1
A[i, j] += 10
```

Internally the A function will keep track of the multiple versions of its definition.
Each version are referred to as `Phase` and can be scheduled freely.

In-place updates can be set for specific points:

```Nim
A[i, j] = 1
A[10, 10] += 10
```

### Initialization

A function must be fully initialized on its domains in its first definition.
The following is not allowed:

```Nim
A[i, 0] = 1
A[0, j] = 2
```

For convenience, when using an in-place operator like `+=` or `*=`
The function will be automatically initialized with the appropriate
neutral element on it's whole domain, i.e. with `+=` it will be initialized to 0
and with `*=`, to 1

### Elementwise iteration

Element-wise operations on a whole tensor can use the wildcard symbol "_"
```Nim
C[_] = A[_] * B[_]
```

#### Function schedule

Automatic optimisations for the heavily nested-loop in deep learning
that takes into account architecture differences (cache size, NUMA sockets, SIMDS, CPUs, GPUs, ...)
is still difficult and often time-consuming
if offered by an application (polyhedral compilers and auto-tuners).

Traditional compilers are significantly lagging behind those applications or hand-written kernels.

Hand-written kernels are difficult to write and quite error-prone.
This is exacerbated in cases like writing function derivatives.

Lux offers a middle-ground:
  - a declarative language to easily define a function.
  - scheduling primitives that enable fearless loop optimizations:
      - nested parallelism, tiling, vectorization, loop fusion at any nest level, loop distribution, ...

TODO: scheduling language


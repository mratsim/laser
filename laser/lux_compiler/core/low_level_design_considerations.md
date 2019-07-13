
# Design

## Document purpose

This document is a scratchpad on why and how Lux arrived at the current low-level design and the alternative considered.
It may be outdated, as I progress and find inefficiencies or worse get stuck
when trying to implement the user interface I envision in the [README](README.md).

## Lux description

> The high-level, user focused description is available in the [README](README.md).

Lux is a DSL embedded in Nim. At the moment it is implemented as macro
and run at compile-time. It has been designed so that the front-end can be
trivially re-used at runtime and only a JIT backend would be needed for
run-time code generation.
In the future, it may also become a pluggable Nim compiler plugin with a DLL.

At a high-level Lux works on symbolic functions
that represent computation to be done on tensors elements and scalars.

The set of high-level functions can be extended and composed
by users and libraries (shallow embeddings).

Those functions use elemental operations represented as a data type (deep embedding):
  - Tensor indexing
  - Addition, multiplication, ...
  - sin, cos, exp, log, ...
to build an AST tree.

We assume that those elemental operations can be composed efficiently
and that we do not need to implement a coarse-grained operators
 to reach state-of-the art performance (for example matrix multiplication or convolution operator) and that they actually actively impeds us. This is a big shift from traditional deep learning frameworks.

To this endeavour the DSL is complemented with a schedule language,
that expresses the loop transformations used in handwritten high-performance-computing kernels
like: parallelization, vectorization, tiling, prefetching or unrolling ...

I.e. the DSL is functional, extensible and composable and generates efficient code.

In the future it should allow autodifferentiation as well.

## Implementation details

### AST

The AST uses sum types. Due to the compile-time requirement it does not
use generics as generic macros arguments don't work and methods/inheritance does
not work at compile-time.

### AST traversal

Furthermore several tree traversal or DSL abstractions are not possible
as they would require generic AST nodes:
- a common tree traversal technique in functional language is catamorphism
  which decouples the tree traversal from the operation applied to the tree node
  i.e. a `map` on a tree.
- An advanced AST design technique is object algebra which encodes the expression tree
  in the generic type of the AST nodes
- Another is tagless final which expresses "nodes" as functions over a generic type T
  that corresponds to an interpreter which can be "eval", "prettyprint", "compile"


i.e. generic macros are not possible:
```nim
macro doSomething[T](io: static varargs[LuxNode[T]]):
  proc matmul(C: var Tensor[float32], A, B: Tensor[float32])
```

### Particular feature:
Computation graph engines and autograds (and symbolic execution engines?)
are traditionally using a context (which can be a hidden global in some ML framework)
to record all the computation done, also called Tape, Trace or Wengert list.
In Lux there is no context, all computations needed for the results are carried
by the results and dead code eliminated by construction.

### AST transformation passes

1. A Function is defined on LuxNodes.
   A schedule is optionally attached to them.

2. If Nim rewrite-rules are defined at the LuxNode level like exp(ln(x)) or fused-multiply-add.
       They are applied automatically by the Nim compiler.

3. Function is then symbolically executed at compile-time (and in the future runtime).
   This eliminates dead code and construct a Lux AST for each output or mutated symbols.

4. The AST contains high-level representation of array access patterns.

5. An inference pass is done to infer tensor ranks and shapes.

6. A lowering pass transforms assignment into affine loops over the relevant iteration domains.

7. A schedule pass will apply the desired schedule on those loops.
   At the start, the schedule will not be checked for validity regarding data dependencies.

8. Nim AST will be generated from the low-level Lux AST at compile-time.
   In the future, LLVM IR or MLIR will be generated instead at runtime.

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ###########################################
#
#                    Design
#
# ###########################################

# Lux is a DSL embedded in Nim. At the moment it is implemented as macro
# and run at compile-time. It has been designed so that the front-end can be
# trivially re-used at runtime and only a JIT backend would be needed for
# run-time code generation.
# In the future, it may also become a pluggable Nim compiler plugin with a DLL.
#
# At a high-level Lux works on symbolic functions
# that represent computation to be done on tensors elements and scalars.
#
# The set of high-level functions can be extended and composed
# by users and libraries (shallow embeddings).
#
# Those functions use elemental operations represented as a data type (deep embedding):
#   - Tensor indexing
#   - Addition, multiplication, ...
#   - sin, cos, exp, log, ...
# to build an AST tree.
#
# We assume that those elemental operations can be composed efficiently
# and that we do not need to implement a coarse-grained operators
# to reach state-of-the art performance (for example matrix multiplication or convolution operator).
#
# To this endeavour the DSL is complemented with a schedule language,
# that expresses the loop transformations used in handwritten high-performance-computing kernels
# like: parallelization, vectorization, tiling, prefetching or unrolling ...
#
# I.e. the DSL is functional, extensible and composable
#
# ## Implementation details
#
# The AST uses sum types. Due to the compile-time requirement it does not
# use generics as generic macros arguments don't work and methods/inheritance does
# not work at compile-time.
#
# Furthermore several tree traversal or DSL abstractions are not possible
# as they would require generic AST nodes:
# - a common tree traversal technique in functional language is catamorphism
#   which decouples the tree traversal from the operation applied to the tree node
#   i.e. a `map` on a tree.
# - An advanced AST design technique is object algebra which encodes the expression tree
#   in the generic type of the AST nodes
# - Another is tagless final which expresses "nodes" as functions over a generic type T
#   that corresponds to an interpreter which can be "eval", "prettyprint", "compile"
#
# i.e. this is not possible:
#   macro doSomething[T](io: static varargs[LuxNode[T]]):
#     proc matmul(C: var Tensor[float32], A, B: Tensor[float32])
#
# Particular feature:
#   Computation graph engines and autograds (and symbolic execution engines?)
#   are traditionally using a context (which can be a hidden global in some ML framework)
#   to record all the computation done, also called Tape, Trace or Wengert list.
#   In Lux there is no context, all computations needed for the results are carried
#   by the results and dead code eliminated by construction.

# ###########################################
#
#         Internal Graph Representation
#
# ###########################################

type
  UnaryOpKind* = enum
    Ln
    Exp

  BinaryOpKind* = enum
    # Must return a scalar for scalar expr check
    Add
    Mul

  TernaryOpKind* = enum
    # Must return a scalar for scalar expr check
    Fma # FusedMultiplyAdd
    Mux # Multiplexer / Selector for example max(x, 0) == mux(x>0, x, 0)

  LuxNodeKind* = enum
    ## Computation Graph / Abstract Syntax Tree nodes
    ## that represents a Lux computation.
    ##
    ## ### Function definition
    ##
    ## A function definition uses Lux nodes.
    ##
    ## The function definition mixes iteration domains, tensor accesses and operations.
    ## It symbolically represents the computation done with a syntax
    ## similar to Einstein summation and implicit for-loops that are inferred from the indices used.
    ##
    ## For example a matrix multiplication would be:
    ##   C[i, j] := A[i, k] * B[k, j]
    ##
    ## The loops i, j, k are implicit.
    ##
    ## Operations are done on symbolic scalars.
    ## Tensors rank will be inferred from the access pattern.
    ##
    ## Element-wise operations on a whole tensor can use "_"
    ##   C[_] := A[_] * B[_]
    ##
    ## The operator `:=` is a shorthand for defining the tensors and indices used (if not defined)
    ## And then doing the symbolic computation.
    ##
    ## ### Function schedule
    ##
    ## Automatic optimisations for the heavily nested-loop in deep learning
    ## that takes into account architecture differences (cache size, NUMA sockets, SIMDS, CPUs, GPUs, ...)
    ## is still difficult and often time-consuming
    ## if offered by an application (polyhedral compilers, auto-tuners).
    ## Traditional compilers are significantly lagging behind those applications or hand-written kernels.
    ## Hand-written kernels are difficult to write and quite error-prone.
    ## This is exacerbated in cases like writing function derivatives.
    ##
    ## Lux offers a middle-ground:
    ##   - a declarative language to easily define a function.
    ##   - scheduling primitives that enable fearless loop optimizations:
    ##       - nested parallelism, tiling, vectorization, loop fusion at any nest level, loop distribution, ...
    ##
    ## ### AST transformation passes
    ##
    ##   0. Function is defined on LuxNodes.
    ##      A schedule is optionally attached to l-value symbols (on the left-side of an assignment).
    ##
    ##   0 bis. If rewrite-rules are defined at the LuxNode level like exp(ln(x)) or fused-multiply-add.
    ##          They are applied automatically by the Nim compiler
    ##
    ##   1. Function is then symbolically executed at compile-time (and in the future runtime).
    ##      This eliminates dead code and construct a Lux AST for each output or mutated symbols.
    ##
    ##   2. The AST contains high-level representation of array access patterns.
    ##
    ##   3. An inference pass is done to infer tensor ranks and shapes.
    ##
    ##   4. A lowering pass transforms assignment into affine loops over the relevant iteration domains.
    ##
    ##   5. A schedule pass will apply the desired schedule on those loops.
    ##      At the start, the schedule will not be checked for validity regarding data dependencies.
    ##
    ##   6. Nim AST will be generated from the low-level Lux AST at compile-time.
    ##      In the future, LLVM IR or MLIR will be generated instead at runtime.

    # ############################################
    #
    #             High-level AST
    #
    # ############################################

    # Scalar invariants
    IntImm        # Integer immediate (known at compile-time)
    FloatImm      # Float immediate (known at compile-time)
    IntParam      # Integer environment parameter (known at run-time, invariant during function execution)
    FloatParam    # Float environment parameter (known at run-time, invariant during function execution)

    # Mutable scalars
    IntMut
    FloatMut
    IntLVal
    FloatLVal

    # Scalar expressions built-ins
    BinOp       # Built-in binary operations

    # Tensor Symbols
    InTensor    # InTensor tensor node
    MutTensor   # Mutable output tensor node
    LValTensor  # Temporary allocated node

    # Tensor access and properties
    Access      # Tensor access
    Shape       # Tensor shape

    # Scalar statements
    Assign      # Assignment statement
    MutAccess   # `[]=` assignment

    # ############################################
    #
    #             Mid-level AST
    #
    # ############################################

    # Affine statements
    AffineFor   # Affine for loop
    AffineIf    # Affine if
    Domain      # Iteration Domain

    # Affine statements:
    # - for/if constraints are a linear expression of
    #   the function invariants and the iterator indices.
    #   with i and j iterator indices and M, N invariants,
    #   Here is an overview of affine/non-affine conditions:
    #   - for i in 0 ..< 10    is valid
    #   - for i in 0 ..< N     is valid
    #   - for i in j ..< 2j+10 is valid
    #   - for i in j ..< M*N   is valid
    #   - for i in j ..< j*j   is invalid, as it's quadratic (j is not an invariant)
    #   - if 2*i - 3*j == 0    is valid
    #   - if i*j < 20          is invalid, as it's quadratic
    #
    # Note: We may extend to "if" with modulo and division by a runtime invariant
    #       which will make it a quasi-affine statement.
    #       - A non-unit step in the for-loop is quasi-affine.
    #       - This will also allows branching depending of
    #         fraction if CPU/GPU charatectristics like cache or TLB size

    # ############################################
    #
    #             Low-level AST
    #
    # ############################################

    # ISA runtime characteristics
    CpuInfo     # CPUInfo function call

    # Control-Flow
    IfElifElse  # Restrict to function invariants?

  Id* = int

  LuxNode* = ref object
    id*: Id

    case kind*: LuxNodeKind
    of InTensor, IntParam, FloatParam:
      ast*: LuxNode               # If nil, it uses symId
      symId*: int
    of MutTensor, LValTensor, IntMut, FloatMut, IntLVal, FloatLVal:
      symLval*: string            # TODO MutTensor should probably use symId
      version*: int
      prev_version*: LuxNode      # Persistent data structure
    of IntImm:
      intVal*: int
    of FloatImm:
      floatVal*: float
    of Assign:
      lval*, rval*: LuxNode
      domains*: seq[LuxNode]      # Nested loops needed to construct this assignment
    of BinOp:
      binOpKind*: BinaryOpKind
      lhs*, rhs*: LuxNode
    of Access, MutAccess:
      tensorView*: LuxNode
      indices*: seq[LuxNode]
    of Shape:
      tensor*: LuxNode
      axis*: int
    of Domain:
      # Domain represents an iteration domain.
      # During execution its value corresponds to the iteration index value
      #
      # Example: "for i in 0 ..< 10:"
      # would be Domain(symbol: "i", start: 0, stop: 10, step: 1)
      # and will be replaced by the value of "i" at run-time
      #
      # Stop is exclusive.
      # We usually steps from 0 to N with N the dimension of a tensor axis.
      # This might change as Nim is inclusive and polyhedral representation
      # uses inclusive constraints.
      symDomain*: string
      start*, stop*, step*: LuxNode
    of AffineFor:
      # Represent a for loop
      domain*: LuxNode
    of AffineIf:
      constraint*: LuxNode
    of CpuInfo:
      # Extern function call
      # Only supports proc with no arguments
      # as it is only needed for CPUInfo
      symFunc*: string
    of IfElifElse:
      # Represent if f0: .. elif f1: .. elif ... else:
      ifBranches*: seq[LuxNode]

  ScheduleKind* = enum
    ScReduce
    ScParallel
    ScVectorize
    ScUnroll
    ScStoreLoc
    ScComputeLoc
    ScOrder
    ScPrefetch
    ScPipeline # Interleaves N successive loops to avoid pipeline stalls
               # Create extra accumulators for reductions
    # Directly modifies AST
    # ScFuse
    # ScTile
    # ScSplit / Distribute
    # ScInterchange
    # ScSkew
    # ScShift

# ###########################################
#
#                   TODO
#
# ###########################################

# ## User-facing
# - Support "ptr T" container
# - lineinfo debug - https://github.com/nim-lang/Nim/issues/11689
# - Compilation statistics on parallelization/vectorization/fusion, ...
# - Compilation logs
# - Booleans:
#     - invariant parameters
#     - tensors for masking
#
# ## Internal
# - Track alignment
# - Track contiguity/striding

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

type

  # ###########################################
  #
  #         Internal Graph Representation
  #
  # ###########################################

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
      domains*: seq[LuxNode]
        # Nested loops needed to construct this assignment
        # Approximatively ordered from outermost to innermost
        # Inner dimension of the lhs is always last.
        # As prefetching for write operations is more expensive.
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
      affineForBody*: LuxNode
      nestedLVal*: LuxNode # for codegen and assigning result
                           # we need the lval that required the loop
    of AffineIf:
      constraint*: LuxNode
      affineIfBody*: LuxNode
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
    #
    # To design
    # ScAffinity --> NUMA, multi-socket, multiGPU affinity

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

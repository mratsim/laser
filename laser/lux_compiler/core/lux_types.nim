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

    # Order is important to separate Expression from statements

    # ################### High-level #########################
    # Functions are the high-level concepts of Lux

    # Scalars, Tensors, Chained Functions
    Func        # Everything is a function
                # to promote composability

    # ################### Expressions #########################

    # Scalar invariants
    IntImm      # Integer immediate (known at compile-time)
    FloatImm    # Float immediate (known at compile-time)
    IntParam    # Integer environment parameter (known at run-time, invariant during function execution)
    FloatParam  # Float environment parameter (known at run-time, invariant during function execution)
    BoolParam   # Bool environment parameter (known at run-time, invariant during function execution)

    # Affine loop expression
    Domain      # Iteration Domain

    # Scalar expressions built-ins
    BinOp       # Built-in binary operations

    # Tensor/Function spatial indexing and properties
    Index       # Access a single element of a Tensor/Func
    Shape       # Tensor/Func shape

    # ISA runtime characteristics
    CpuInfo     # CPUInfo function call

    # ################### Statements #########################
    # Statements are generate when the high-level functional AST
    # is lowered to an AST that more closely match Nim's AST.

    # Scalar statements
    Assign

    # Affine loop statements
    AffineFor   # Affine for loop
    AffineIf    # Affine if

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


  Id* = int

  LuxNode* = ref object
    id*: Id

    case kind*: LuxNodeKind
    of IntImm:
      intVal*: int
    of FloatImm:
      floatVal*: float
    of IntParam, FloatParam:
      symParam*: string
    of Func:
      function: Function
    of BinOp:
      binOpKind*: BinaryOpKind
      lhs*, rhs*: LuxNode
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
      iter*: Iter
    of AffineFor:
      # Represent a for loop
      domain*: Iter
      affineForBody*: LuxNode
    of AffineIf:
      # Represent an if around an assignment
      # It should only be an affine combination of iterators
      # and run-time invariant IntParameters to allow
      # polyhedral optimizations, for example to schedule non-rectangular loops.
      constraint*: LuxNode
      affineIfBody*: LuxNode
    of CpuInfo:
      # Extern function call
      # Only supports proc with no arguments
      # as it is only needed for CPUInfo
      symFunc*: string

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

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

type

  # ###########################################
  #
  #         Internal AST Representation
  #
  # ###########################################

  UnaryOpKind* = enum
    Ln
    Exp

  BinaryOpKind* = enum
    # Must return a scalar for scalar expr check
    Add
    Mul
    Eq

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

    # ----- Kinds not using "children: seq[LuxNode]" field
    # Scalar invariants
    IntLit      # Integer immediate (known at compile-time)
    FloatLit    # Float immediate (known at compile-time)
    IntParam    # Integer environment parameter (known at run-time, invariant during function execution)
    FloatParam  # Float environment parameter (known at run-time, invariant during function execution)
    BoolParam   # Bool environment parameter (known at run-time, invariant during function execution)

    # Affine loop expression
    Domain      # Iteration Domain

    # Scalar expressions built-ins
    BinOpKind   # Built-in binary operations

    # ----- Kinds using "children: seq[LuxNode]" field
    BinOp       #

    # Tensor/Function spatial indexing and properties
    Access      # Access a single element of a Tensor/Func
    DimSize     # Get the size of one of the Tensor/Func dimensions

    # Function Calls
    ExternCall  # Extern function call like CPUInfo

    # ################### Statements #########################
    # Statements are generate when the high-level functional AST
    # is lowered to an AST that more closely match Nim's AST.

    # General statement
    StatementList

    # Scalar statements
    Assign
    Check

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
    of IntLit:
      intVal*: int
    of FloatLit:
      floatVal*: float
    of IntParam, FloatParam:
      symParam*: string
    of Func:
      fn*: Fn
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
    of BinOpKind:
      bopKind*: BinaryOpKind
    else:
      children*: seq[LuxNode]

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
  #         High-level data structures
  #
  # ###########################################

  Iter* = ref object
    domId*: Id # Shouldn't be necessary
               # But when passing to the backend
               # The reference is lost
    symbol*: string
    start*, stop*, step*: LuxNode

  InvariantKind* = enum
    ikInt
    ikFloat

  Invariant* = ref object
    ## Runtime invariant inputs once a kernel is called
    kind*: InvariantKind
    symbol*: string

  FnOutputKind* = enum
    fnScalar
    fnTensor

  Fn* = ref object
    ## Main datastructure of Lux
    ## A function can be redefined to represent mutation
    ## Each redefinition called a stage.
    ## A function can represent a Tensor, a Scalar,
    ## or a sequence of transformations applied to them.
    ## Those transformations can be scheduled at the global function level
    ## and at the stage level.
    ##
    ## The first initial stage must define a default value
    ## on the whole domain
    #
    # For symbolic execution, the initial input tensors
    # will be created as Function with no stages at all.
    symbol*: string
    stages*: seq[Stage]
    schedule*: FnSchedule
    outputKind*: FnOutputKind

  # ###########################################
  #
  #         Low-level data structures
  #
  # ###########################################

  Call* = ref object
    ## Object created when indexing a Function
    ## with A[i,j] or A[i,j] = expression
    # Must be ref object to live long enough for
    # assignation
    fn*: Fn
    params*: seq[LuxNode]

  Stage* = object
    ## A unique definition of a function
    definition*: LuxNode
    params*: seq[LuxNode]
      # domain or specific location to apply this phase
    recurrence*: seq[LuxNode]
      # Stage is repeated on a non-spatial domain, i.e.
      # the iteration domain does not appear on neither the LHS or RHS assignment
      # for example a Gauss-Seidel Smoother with the time domain t:
      # ------------------------------------
      #   for t in 0 ..< timeSteps:
      #     for i in 1 ..< N-1:
      #       for j in 1 ..< N-1:
      #         A[i][j] = 0.25 * (A[i][j-1] * # left
      #                           A[i][j+1] * # right
      #                           A[i-1][j] * # top
      #                           A[i+1][j])  # bottom
    condition*: LuxNode
      # wrapped in a
      # if(condition):
      #   A(i, j) = ...
    schedule*: StageSchedule

  StageSchedule* = object
  FnSchedule* = object

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

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random, tables

# ###########################################
#
#         Internal Graph Representation
#
# ###########################################

type
  IDom* = object
    ## Represents an iteration domain
    ## Example: "for i in 0 ..< 10:"
    ## would be Domain(symbol: "i", start: 0, stop: 10, step: 1)
    ##
    ## Stop is exclusive.
    ## We usually steps from 0 to N with N the dimension of a tensor axis.
    ## This might change as Nim is inclusive and polyhedral representation
    ## uses inclusive constraints.
    symbol: string
    start, stop, step: LuxNode

  LuxNodeKind* = enum
    ## Computation Graph / Abstract Syntax Tree nodes
    ## that represents a Lux computation.
    ##
    ## ### Function definition
    ##
    ## A function definition uses Lux nodes.
    ##
    ## The function definition mixes iteration domains, tensor accesses and operations.
    ## It symbolically represent the computation done with a syntax
    ## similar to Einstein summation and implicit for-loops, depending on indices used.
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

    # Scalar expressions
    Add         # Addition
    Mul         # Multiplication

    # Tensor Symbols
    InTensor    # InTensor tensor node
    MutTensor   # Mutable output tensor node
    LValTensor  # Temporary allocated node

    # Tensor access and properties
    Access      # Tensor access
    Shape       # Tensor shape

    # Scalar statements
    Assign      # Assignment statement

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
    # Note: We may extend to "if" affine modulo
    #       which will make it a quasi-affine statement.
    #       A non-unit step in the for-loop is quasi-affine.

  Id* = int

  LuxNode* = ref object
    id*: Id
    lineInfo*: tuple[filename: string, line: int, column: int]
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
    of Assign, Add, Mul:
      lhs*, rhs*: LuxNode
    of Access:
      tensorView*: LuxNode
      domains*: seq[LuxNode]
    of Shape:
      tensor*: LuxNode
      axis*: int
    of Domain, AffineFor:
      domain*: IDom
    of AffineIf:
      constraint*: LuxNode

  ScheduleKind* = enum
    ScReduce
    ScParallel
    ScVectorize
    ScUnroll
    ScStoreLoc
    ScComputeLoc
    ScOrder
    # Directly modifies AST
    # ScFuse
    # ScTile
    # ScSplit / Distribute
    # ScInterchange
    # ScSkew
    # ScShift



# ###########################################
#
#         Routine definitions
#
# ###########################################

var luxNodeRng {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for LuxNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

proc genId(): int =
  luxNodeRng.rand(high(int))

proc input*(id: int): LuxNode =
  when nimvm:
    LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: InTensor, symId: id
    )
  else: # TODO: runtime ID
    LuxNode(kind: InTensor, symId: id)

proc `+`*(a, b: LuxNode): LuxNode =
  when nimvm:
    LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: Add, lhs: a, rhs: b
    )
  else: # TODO: runtime ID
    LuxNode(kind: Add, lhs: a, rhs: b)

proc `*`*(a, b: LuxNode): LuxNode =
  when nimvm:
    LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: Mul, lhs: a, rhs: b
    )
  else:
    LuxNode(id: genId(), kind: Mul, lhs: a, rhs: b)

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  when nimvm:
    LuxNode(
        id: genId(), lineInfo: instantiationInfo(),
        kind: Mul,
        lhs: a,
        rhs: LuxNode(kind: IntImm, intVal: b)
      )
  else:
    LuxNode(
        kind: Mul,
        lhs: a,
        rhs: LuxNode(kind: IntImm, intVal: b)
      )

proc `+=`*(a: var LuxNode, b: LuxNode) =
  assert a.kind notin {InTensor, IntImm, FloatImm}

  # If LHS does not have a memory location, attribute one
  if a.kind notin {MutTensor, LValTensor}:
    a = LuxNode(
          id: genId(), lineInfo: instantiationInfo(),
          kind: LValTensor,
          symLVal: "localvar__" & $a.id, # Generate unique symbol
          version: 1,
          prev_version: LuxNode(
            id: a.id, lineInfo: a.lineinfo,
            kind: Assign,
            lhs: LuxNode(
              id: a.id, lineInfo: a.lineinfo, # Keep the hash
              kind: LValTensor,
              symLVal: "localvar__" & $a.id, # Generate unique symbol
              version: 0,
              prev_version: nil,
            ),
            rhs: a
          )
    )

  # Then update it
  if a.kind == MutTensor:
    a = LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: MutTensor,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id, lineinfo: a.lineinfo,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )
  else:
    a = LuxNode(
      id: genId(), lineinfo: instantiationInfo(),
      kind: LValTensor,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id, lineinfo: a.lineinfo,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )

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
  Domain* = object
    ## Represents an iteration or a reduction domain
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

    # Expressions
    IntImm      # Integer immediate
    FloatImm    # Float immediate
    Add         # Elementwise add
    Mul         # Elementwise mul

    # Symbols
    Input       # Input tensor node
    Output      # Mutable output tensor node
    LVal        # Temporary allocated node
    Assign      # Assignment statement

  Id* = int

  LuxNode* = ref object
    id*: Id
    lineInfo*: tuple[filename: string, line: int, column: int]
    case kind*: LuxNodeKind
    of Input:
      symId*: int
    of Output, LVal:
      symLval*: string
      version*: int
      prev_version*: LuxNode      # Persistent data structure
    of IntImm:
      intVal*: int
    of FloatImm:
      floatVal*: float
    of Assign, Add, Mul:
      lhs*, rhs*: LuxNode

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
      kind: Input, symId: id
    )
  else: # TODO: runtime ID
    LuxNode(kind: Input, symId: id)

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
  assert a.kind notin {Input, IntImm, FloatImm}
  if a.kind notin {Output, LVal}:
    a = LuxNode(
          id: genId(), lineInfo: instantiationInfo(),
          kind: LVal,
          symLVal: "localvar__" & $a.id, # Generate unique symbol
          version: 1,
          prev_version: LuxNode(
            id: a.id, lineInfo: a.lineinfo,
            kind: Assign,
            lhs: LuxNode(
              id: a.id, lineInfo: a.lineinfo, # Keep the hash
              kind: LVal,
              symLVal: "localvar__" & $a.id, # Generate unique symbol
              version: 0,
              prev_version: nil,
            ),
            rhs: a
          )
    )
  if a.kind == Output:
    a = LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: Output,
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
      kind: LVal,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )

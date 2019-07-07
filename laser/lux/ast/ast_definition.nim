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
  LuxNodeKind* = enum
    ## Elemental Op Kind

    # Expressions
    IntImm      # Mark an integer immediate that will be broadcasted
    FloatImm    # Mark a float immediate that will be broadcasted
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

# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  hashes, random, tables

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

  LuxNode* = ref object
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

    ctHash*: Hash                 # Compile-Time only Hash (TODO)

# ###########################################
#
#         Routine definitions
#
# ###########################################

var astNodeRng {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for LuxNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

proc genHash(): Hash =
  Hash astNodeRng.rand(high(int))

proc hash*(x: LuxNode): Hash {.inline.} =
  when nimvm:
    x.cthash
  else: # Take its address
    cast[Hash](x)

proc input*(id: int): LuxNode =
  when nimvm:
    LuxNode(ctHash: genHash(), kind: Input, symId: id)
  else:
    LuxNode(kind: Input, symId: id)

proc `+`*(a, b: LuxNode): LuxNode =
  when nimvm:
    LuxNode(ctHash: genHash(), kind: Add, lhs: a, rhs: b)
  else:
    LuxNode(kind: Add, lhs: a, rhs: b)

proc `*`*(a, b: LuxNode): LuxNode =
  when nimvm:
    LuxNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)
  else:
    LuxNode(ctHash: genHash(), kind: Mul, lhs: a, rhs: b)

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  when nimvm:
    LuxNode(
        ctHash: genHash(),
        kind: Mul,
        lhs: a,
        rhs: LuxNode(kind: IntScalar, intVal: b)
      )
  else:
    LuxNode(
        kind: Mul,
        lhs: a,
        rhs: LuxNode(kind: IntScalar, intVal: b)
      )

proc `+=`*(a: var LuxNode, b: LuxNode) =
  assert a.kind notin {Input, IntImm, FloatImm}
  if a.kind notin {Output, LVal}:
    a = LuxNode(
          ctHash: genHash(),
          kind: LVal,
          symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
          version: 1,
          prev_version: LuxNode(
            cthash: a.ctHash,
            kind: Assign,
            lhs: LuxNode(
              ctHash: a.ctHash, # Keep the hash
              kind: LVal,
              symLVal: "localvar__" & $a.ctHash, # Generate unique symbol
              version: 0,
              prev_version: nil,
            ),
            rhs: a
          )
    )
  if a.kind == Output:
    a = LuxNode(
      ctHash: genHash(),
      kind: Output,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        ctHash: a.ctHash,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )
  else:
    a = LuxNode(
      ctHash: genHash(),
      kind: LVal,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        ctHash: a.ctHash,
        kind: Assign,
        lhs: a,
        rhs: a + b
      )
    )

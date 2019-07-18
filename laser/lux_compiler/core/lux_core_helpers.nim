# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random,
  # Internal
  ./lux_types

# ###########################################
#
#         LuxNode unique identifiers
#
# ###########################################

var luxNodeRngCT {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for LuxNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

var luxNodeRngRT = initRand(0x42)
  ## Runtime ID

proc genId*(): int =
  when nimvm:
    luxNodeRngCT.rand(high(int))
  else:
    luxNodeRngRT.rand(high(int))

# ###########################################
#
#         LuxNode basic procs
#
# ###########################################

const LuxExpr* = {IntLit..DimSize}
const LuxStmt* = {StatementList..AffineIf}

proc `[]`*(node: LuxNode, idx: int): var LuxNode =
  node.children[idx]

proc `[]=`*(node: LuxNode, idx: int, val: LuxNode) =
  node.children[idx] = val

proc len*(node: LuxNode): int =
  node.children.len

proc add*(node: LuxNode, val: LuxNode) =
  node.children.add val

proc add*(node: LuxNode, vals: openarray[LuxNode]) =
  node.children.add vals

iterator items*(node: LuxNode): LuxNode =
  doAssert(
    node.kind notin {Func .. BinOpKind},
    "Invalid node kind for iteration: \"" & $node.kind & '\"'
  )
  for val in node.children:
    yield val

# ###########################################
#
#         LuxNode tree constructions
#
# ###########################################

proc newTree*(kind: LuxNodeKind, args: varargs[LuxNode]): LuxNode =
  new result
  result.id = genId()
  result.kind = kind
  result.children = @args

proc newLux*(fn: Fn): LuxNode =
  LuxNode(
    id: genId(),
    kind: Func, fn: fn
  )

proc newLux*(domain: Iter): LuxNode =
  LuxNode(
    id: genId(),
    kind: Domain, iter: domain
  )

# We don't need genId for the following variant types

proc newLux*(lit: int): LuxNode =
  LuxNode(kind: IntLit, intVal: lit)

proc newLux*(lit: float): LuxNode =
  LuxNode(kind: FloatLit, floatVal: lit)

proc newLux*(invariant: Invariant): LuxNode =
  case invariant.kind:
  of ikInt:
    LuxNode(kind: IntParam, symParam: invariant.symbol)
  of ikFLoat:
    LuxNode(kind: FloatParam, symParam: invariant.symbol)

proc newLux*(bopKind: BinaryOpKind): LuxNode =
  LuxNode(kind: BinOpKind, bopKind: bopKind)

proc newLuxStmtList*(): LuxNode =
  LuxNode(kind: StatementList)

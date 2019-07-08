# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  random,
  # Internal
  ./ast_types

# ###########################################
#
#         Lux Primitive Routines
#
# ###########################################

var luxNodeRngCT {.compileTime.} = initRand(0x42)
  ## Workaround for having no UUID for LuxNodes
  ## at compile-time - https://github.com/nim-lang/RFCs/issues/131

var luxNodeRngRT = initRand(0x42)
  ## Runtime ID

proc genId(): int =
  when nimvm:
    luxNodeRngCT.rand(high(int))
  else:
    luxNodeRngRT.rand(high(int))

const ScalarExpr = {
            IntImm, FloatImm, IntParam, FloatParam,
            IntMut, FloatMut, IntLVal, FloatLVal,
            BinOp,
            Access, Shape, Domain
          }

func checkScalarExpr(targetOp: string, input: LuxNode) =
  # TODO - mapping LuxNode -> corresponding function
  # for better errors as LuxNode are "low-level"
  if input.kind notin ScalarExpr:
    raise newException(
      ValueError,
      "Invalid scalar expression \"" & $input.kind & "\"\n" &
      "Only the following LuxNodes are allowed:\n    " & $ScalarExpr &
      "\nfor building a \"" & targetOp & "\" function."
      # "\nOrigin: " & $node.lineInfo
    )

func checkMutable(node: LuxNode) =
  # TODO
  discard

proc input*(paramId: int): LuxNode =
  LuxNode(
    id: genId(), # lineInfo: instantiationInfo(),
    kind: InTensor, symId: paramId
  )

proc `+`*(a, b: LuxNode): LuxNode =
  checkScalarExpr("Add", a)
  checkScalarExpr("Add", b)
  LuxNode(
    id: genId(), # lineInfo: instantiationInfo(),
    kind: BinOp, binOpKind: Add,
    lhs: a, rhs: b
  )

proc `*`*(a, b: LuxNode): LuxNode =
  checkScalarExpr("Mul", a)
  checkScalarExpr("Mul", b)
  LuxNode(
    id: genId(), # lineInfo: instantiationInfo(),
    kind: BinOp, binOpKind: Mul,
    lhs: a, rhs: b
  )

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  checkScalarExpr("Mul", a)
  LuxNode(
      id: genId(), # lineInfo: instantiationInfo(),
      kind: BinOp, binOpKind: Mul,
      lhs: a,
      rhs: LuxNode(kind: IntImm, intVal: b)
    )

proc `+=`*(a: var LuxNode, b: LuxNode) =
  checkScalarExpr("In-place addition", b)
  checkMutable(a)

  # If LHS does not have a memory location, attribute one
  if a.kind notin {MutTensor, LValTensor}:
    a = LuxNode(
          id: genId(), # lineInfo: instantiationInfo(),
          kind: LValTensor,
          symLVal: "localvar__" & $a.id, # Generate unique symbol
          version: 1,
          prev_version: LuxNode(
            id: a.id, # lineInfo: a.lineinfo,
            kind: Assign,
            lval: LuxNode(
              id: a.id, # lineInfo: a.lineinfo, # Keep the hash
              kind: LValTensor,
              symLVal: "localvar__" & $a.id, # Generate unique symbol
              version: 0,
              prev_version: nil,
            ),
            rval: a
          )
    )

  # Then update it
  if a.kind == MutTensor:
    a = LuxNode(
      id: genId(), # lineInfo: instantiationInfo(),
      kind: MutTensor,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id, # lineinfo: a.lineinfo,
        kind: Assign,
        lval: a,
        rval: a + b
      )
    )
  else:
    a = LuxNode(
      id: genId(), # lineinfo: instantiationInfo(),
      kind: LValTensor,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id, # lineinfo: a.lineinfo,
        kind: Assign,
        lval: a,
        rval: a + b
      )
    )

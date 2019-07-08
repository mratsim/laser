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
      kind: BinOp, binOpKind: Add,
      lhs: a, rhs: b
    )
  else: # TODO: runtime ID
    LuxNode(
      kind: BinOp, binOpKind: Add,
      lhs: a, rhs: b
    )

proc `*`*(a, b: LuxNode): LuxNode =
  when nimvm:
    LuxNode(
      id: genId(), lineInfo: instantiationInfo(),
      kind: BinOp, binOpKind: Mul,
      lhs: a, rhs: b
    )
  else: # TODO: runtime ID
    LuxNode(
      kind: BinOp, binOpKind: Mul,
      lhs: a, rhs: b
    )

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  when nimvm:
    LuxNode(
        id: genId(), lineInfo: instantiationInfo(),
        kind: BinOp, binOpKind: Mul,
        lhs: a,
        rhs: LuxNode(kind: IntImm, intVal: b)
      )
  else: # TODO: runtime ID
    LuxNode(
        kind: BinOp, binOpKind: Mul,
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
            lval: LuxNode(
              id: a.id, lineInfo: a.lineinfo, # Keep the hash
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
      id: genId(), lineInfo: instantiationInfo(),
      kind: MutTensor,
      symLVal: a.symLVal, # Keep original unique symbol
      version: a.version + 1,
      prev_version: LuxNode(
        id: a.id, lineinfo: a.lineinfo,
        kind: Assign,
        lval: a,
        rval: a + b
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
        lval: a,
        rval: a + b
      )
    )

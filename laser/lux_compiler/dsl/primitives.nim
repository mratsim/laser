# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ./primitives_helpers,
  ../../private/ast_utils,
  # Debug
  ../core/lux_print

# ###########################################
#
#         Lux Primitive Routines
#
# ###########################################

template newLuxMutTensor*(t: untyped) =
  t = LuxNode(
      id: genId(),
      kind: MutTensor,
      symLVal: t.astToStr
    )

template newLuxIterDomain*(index: untyped) =
  index = LuxNode(
      id: genId(),
      kind: Domain,
      symDomain: index.astToStr
    )

proc `+`*(a, b: LuxNode): LuxNode =
  checkScalarExpr("Add", a)
  checkScalarExpr("Add", b)
  LuxNode(
    id: genId(),
    kind: BinOp, binOpKind: Add,
    lhs: a, rhs: b
  )

proc `*`*(a, b: LuxNode): LuxNode =
  checkScalarExpr("Mul", a)
  checkScalarExpr("Mul", b)
  LuxNode(
    id: genId(),
    kind: BinOp, binOpKind: Mul,
    lhs: a, rhs: b
  )

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  checkScalarExpr("Mul", a)
  LuxNode(
      id: genId(),
      kind: BinOp, binOpKind: Mul,
      lhs: a,
      rhs: LuxNode(kind: IntImm, intVal: b)
    )

proc `+=`*(a: var LuxNode, b: LuxNode) =
  checkScalarExpr("In-place addition", b)
  checkMutable(a)

  # If LHS does not have a memory location, attribute one
  if a.kind notin {MutTensor, LValTensor}:
    lvalify(a)

  # Then update it
  if a.kind == LValTensor:
    a = LuxNode(
      id: genId(),
      kind: LValTensor,
      symLVal: a.symLVal,
      version: a.version + 1,
      prev_version: assign(
        lhs = a,
        rhs = a + b
      )
    )
  else:
    a = LuxNode(
      id: genId(),
      kind: MutTensor,
      symLVal: a.symLVal,
      version: a.version + 1,
      prev_version: assign(
        lhs = a,
        rhs = a + b
      )
    )

proc at(t: LuxNode, indices: varargs[LuxNode]): LuxNode =
  ## Access a tensor
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)

  checkTensor(t)
  LuxNode(
      id: genId(),
      kind: Access,
      tensorView: t,
      indices: @indices
  )

proc at(t: var LuxNode, indices: varargs[LuxNode]): var LuxNode =
  ## Access a tensor, returns a mutable element
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  ##
  ## Used for A[i, j] += foo(i, j)

  checkTensor(t)
  checkMutable(t)
  t = LuxNode(
      id: genId(),
      kind: MutAccess,
      tensorView: t,
      indices: @indices
  )
  return t

proc at_mut(t: var LuxNode, indices: varargs[LuxNode], expression: LuxNode) =
  ## Mutate a tensor element
  ## at specified indices
  ##
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  ##
  ## Used for A[i, j] = foo(i, j)

  checkTensor(t)
  checkMutable(t)

  # If LHS does not have a memory location, attribute one
  if t.kind notin {MutTensor, LValTensor}:
    lvalify(t)

  # Then update it
  if t.kind == LValTensor:
    t = LuxNode(
      id: genId(),
      kind: LValTensor,
      symLVal: t.symLVal,
      version: t.version + 1,
      prev_version: assign(
        LuxNode(
          id: genId(),
          kind: MutAccess,
          tensorView: t,
          indices: @indices
        ),
        expression
      )
    )
  else:
    t = LuxNode(
      id: genId(),
      kind: MutTensor,
      symLVal: t.symLVal,
      version: t.version + 1,
      prev_version: assign(
        LuxNode(
          id: genId(),
          kind: MutAccess,
          tensorView: t,
          indices: @indices
        ),
        expression
      )
    )
    echo t

macro `[]`*(t: LuxNode, indices: varargs[untyped]): untyped =
  # TODO
  # Handle the "_" joker for whole dimension
  result = newCall(bindSym"at", t)
  indices.copyChildrenTo result

macro `[]=`*(t: var LuxNode, indicesAndExpr: varargs[untyped]): untyped =
  # Handle varargs[untyped] consume everything
  var indices = indicesAndExpr
  let expression = indices.pop()

  # TODO
  # Handle the "_" joker for whole dimension

  result = newCall(bindSym"at_mut", t)
  indices.copyChildrenTo result
  result.add expression

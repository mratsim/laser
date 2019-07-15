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
#         Lux DSL Primitive Routines
#
# ###########################################

proc dim_size*(t: LuxNode, axis: int): LuxNode =
  LuxNode(
    kind: DimSize,
    tensor: t,
    axis: axis
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
  if a.kind notin {MutTensor, LValTensor, MutAccess}:
    lvalify(a)

  # Then create the new node
  var upd_a = LuxNode(id: genId())
  upd_a.kind = a.kind
  upd_a.symLVal = a.symLVal
  upd_a.version = a.version + 1
  upd_a.prev_version = assign(
    lhs = a,
    rhs = a + b
  )

  # And swap it
  a = upd_a

proc `[]`*(t: Function, indices: varargs[untyped]): untyped =
  ## Access a tensor
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  # TODO
  # Handle the "_" joker for whole dimension


proc `[]`*(t: var Function, indices: varargs[untyped]): untyped =
  ## Access a tensor
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  # TODO
  # Handle the "_" joker for whole dimension

proc `[]=`*(t: var Function, indicesAndExpr: varargs[untyped]): untyped =
  ## Mutate a Func/tensor element
  ## at specified indices
  ##
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  ##
  ## Used for A[i, j] = foo(i, j)
  # TODO
  # Handle the "_" joker for whole dimension

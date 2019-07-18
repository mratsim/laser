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

proc dim_size*(fn: Fn, axis: int): LuxNode =
  DimSize.newTree(
    newLux fn,
    newLux axis
  )

proc `+`*(a, b: LuxNode): LuxNode =
  assert(a.kind in LuxExpr)
  assert(b.kind in LuxExpr)
  BinOp.newTree(
    newLux Add,
    a, b
  )

proc `*`*(a, b: LuxNode): LuxNode =
  assert(a.kind in LuxExpr)
  assert(b.kind in LuxExpr)
  BinOp.newTree(
    newLux Mul,
    a, b
  )

proc `*`*(a: LuxNode, b: SomeInteger): LuxNode =
  assert(a.kind in LuxExpr)
  assert(b.kind in LuxExpr)
  BinOp.newTree(
    newLux Add,
    a, newLux b
  )

proc `+=`*(a: Call, b: LuxNode) =
  discard

proc at(fn: Fn, indices: varargs[Iter]): Call =
  ## Access a tensor/function
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  assert not fn.isNil
  new result
  result.fn = fn
  for iter in indices:
    result.params.add newLux(iter)

proc at(fn: var Fn, indices: varargs[Iter]): Call =
  ## Access a tensor/function
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  if fn.isNil:
    new fn
  result.fn = fn
  for iter in indices:
    result.params.add newLux(iter)

macro `[]`*(fn: Fn, indices: varargs[Iter]): untyped =
  # TODO
  # - Handle the "_" joker for whole dimension
  # - Handle combinations of Iter, LuxNodes, IntParams and literals
  result = newStmtList()
  let args = symFnAndIndices(result, fn, indices)
  result.add quote do:
    at(`fn`, `args`)

macro `[]`*(fn: var Fn, indices: varargs[Iter]): untyped =
  # TODO
  # - Handle the "_" joker for whole dimension
  # - Handle combinations of Iter, LuxNodes, IntParams and literals
  result = newStmtList()
  result.add quote do:
    if `fn`.isNil:
      new `fn`
  let args = symFnAndIndices(result, fn, indices)
  result.add quote do:
    at(`fn`, `args`)

proc at_mut(
        fn: var Fn,
        indices: varargs[Iter],
        expression: LuxNode) =
  ## Mutate a Func/tensor element
  ## at specified indices
  ##
  ## For example
  ##   - A[i, j, k] on a rank 3 tensor
  ##   - A[0, i+j] on a rank 2 tensor (matrix)
  ##
  ## Used for A[i, j] = foo(i, j)
  let stageId = fn.stages.len
  # if stageId = 0: assert that indices are the full function domain.
  fn.stages.setLen(stageId+1)
  for iter in indices:
    fn.stages[stageId].params.add newLux(iter)
  fn.stages[stageId].definition = expression

macro `[]=`*(
        fn: var Fn,
        indices: varargs[Iter],
        expression: LuxNode): untyped =
  # TODO
  # - Handle the "_" joker for whole dimension
  # - Handle combinations of Iter, LuxNodes, IntParams and literals
  result = newStmtList()
  result.add quote do:
    if `fn`.isNil:
      new `fn`
  let args = symFnAndIndices(result, fn, indices)
  result.add quote do:
    at_mut(`fn`, `args`, `expression`)

converter toLuxNode*(call: Call): LuxNode =
  # Implicit conversion of function/tensor indexing
  # to allow seamless:
  # A[i,j] = myParam + B[i,j]
  result = Access.newTree(
    newLux call.fn
  )
  result.add call.params

converter toLuxNode*(lit: int): LuxNode =
  result = newLux lit

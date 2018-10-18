# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  strformat, macros,
  ./metadata, ./tensor, ./utils, ./compiler_optim_hints,
  ./tensor_display

func bcShape(x, y: Metadata): Metadata =
  if x.len > y.len:
    result = x
    for i, idx in result.mpairs:
      if idx == 1:
        idx = y[i]
      # Shape compat check on non-singleton dimension should be done when actually broadcasting
  else:
    result = y
    for i, idx in result.mpairs:
      if idx == 1:
        idx = x[i]
      # Shape compat check on non-singleton dimension should be done when actually broadcasting

macro getBroadcastShape(x: varargs[typed]): untyped =
  assert x.len >= 2
  result = nnkDotExpr.newTree(x[0], ident"shape")
  for i in 1 ..< x.len:
    let xi = x[i]
    result = quote do: bcShape(`result`, `xi`.shape)

func bc[T](t: Tensor[T], shape: Metadata): Tensor[T] =
  ## Broadcast tensors
  result.shape = shape
  for i in 0 ..< t.rank:
    if t.shape[i] == 1 and shape[i] != 1:
      result.strides[i] = 0
    else:
      result.strides[i] = t.strides[i]
      if t.shape[i] != result.shape[i]:
        raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")
  result.offset = t.offset
  result.storage = t.storage

func bc[T: SomeNumber](x: T, shape: Metadata): T {.inline.}=
  ## "Broadcast" scalars
  x

macro broadcastImpl(output: untyped, inputs_body: varargs[untyped]): untyped =
  ## If output is empty node it will return a value
  ## otherwise, result will be assigned in-place to output
  let
    in_place = newLit output.kind != nnkEmpty

  var
    inputs = inputs_body
    body = inputs.pop()

  let
    shape = genSym(nskLet, "broadcast_shape__")
    coord = genSym(nskVar, "broadcast_coord__")

  var doBroadcast = newStmtList()
  var bcInputs = nnkArgList.newTree()
  for input in inputs:
    let broadcasted = genSym(nskLet, "broadcast_" & $input & "__")
    doBroadcast.add newLetStmt(
      broadcasted,
      newCall(bindSym"bc", input, shape)
    )
    bcInputs.add nnkBracketExpr.newTree(broadcasted, coord)

  body = body.replaceNodes(bcInputs, inputs)

  result = quote do:
    block:
      let `shape` = getBroadcastShape(`inputs`)
      let rank = `shape`.len
      withCompilerOptimHints()
      var `coord`{.align64.}: array[MAXRANK, int] # Current coordinates in the n-dimensional space
      `doBroadcast`

      when not `in_place`:
        var output = newTensor[type(`body`)](`shape`)
      else:
        assert `output`.shape == `shape`

      for _ in 0 ..< `shape`.product:
        # Assign for the current iteration
        when not `in_place`:
          output[`coord`] = `body`
        else:
          `output`[`coord`] = `body`

        # Compute the next position
        for k in countdown(rank - 1, 0):
          if `coord`[k] < `shape`[k] - 1:
            `coord`[k] += 1
            break
          else:
            `coord`[k] = 0

      # Now return the value
      when not `in_place`:
        output

macro broadcast(inputs_body: varargs[untyped]): untyped =
  getAST(broadcastImpl(newEmptyNode(), inputs_body))

macro materialize*(output: var Tensor, inputs_body: varargs[untyped]): untyped =
  getAST(broadcastImpl(output, inputs_body))

#################################################################################

import math
proc sanityChecks() =
  # Sanity checks

  let x = randomTensor([1, 2, 3], 10)
  let y = randomTensor([5, 2], 10)

  echo x
  echo y

  block: # Simple assignation
    echo "\nSimple assignation"
    let a = broadcast(x, y):
      x * y

    echo a

  block: # In-place, similar to Julia impl
    echo "\nIn-place, similar to Julia impl"
    var a = newTensor[int]([5, 2, 3])
    materialize(a, x, y):
      x * y

    echo a

  block: # Complex multi statement with type conversion
    echo "\nComplex multi statement with type conversion"
    let a = broadcast(x, y):
      let c = cos x.float64
      let s = sin y.float64

      sqrt(c.pow(2) + s.pow(2))

    echo a

  block: # Variadic number of types with proc declaration inside
    echo "\nVariadic number of types with proc declaration inside"
    var u, v, w, x, y, z = randomTensor([3, 3], 10)

    let c = 2

    let a = broadcast(u, v, w, x, y, z):
      # ((u * v * w) div c) mod (if not zero (x - y + z) else 42)

      proc ifNotZero(val, default: int): int =
        if val == 0: default
        else: val

      let uvw_divc = u * v * w div c
      let xmypz = x - y + z

      uvw_divc mod ifNotZero(xmypz, 42)

    echo a

  block: # Simple broadcasted addition test
    echo "\nSimple broadcasted addition test"
    var a = newTensor[int]([2, 3])
    a.storage.data = @[3, 2, 1, 1, 2, 3] # Ideally we should have arrays of arrays -> tensor conversion
    var b = newTensor[int]([1, 3])
    b.storage.data = @[1, 2, 3]

    let c = broadcast(a, b): a + b
    doAssert c.storage.data == @[4, 4, 4, 2, 4, 6]
    echo "✓ Passed"

#################################################################################

when isMainModule:
  sanityChecks()

# MIT License
# Copyright (c) 2018 Mamy André-Ratsimbazafy

import
  macros,
  ./tensor, ./tensor_display, ./metadata, ./utils, ./compiler_optim_hints

proc genNestedFor(shape, indices, innerBody: NimNode): NimNode =
  # shape is a NimNode representing a Metadata
  # indices is a compile-time array corresponding to the iteration indices

  result = innerBody
  for i in countdown(indices.len - 1, 0):
    let idx = indices[i]
    result = quote do:
      for `idx` in 0 ..< `shape`[`i`]:
        `result`

macro triotForEach*(args: varargs[untyped]): untyped =
  ## Please assign input tensor to a variable first
  ## If they result from a proc, the proc that generated the tensor
  ## will be called multiple time by the macro.
  ## Also there is no mutability check

  result = newStmtList()

  var params = args
  var loopBody = params.pop()

  var values = nnkBracket.newTree()
  var tensors = nnkArglist.newTree()

  template syntaxError() {.dirty.} =
    error "Syntax error: argument " & ($arg.kind).substr(3) & " in position #" & $i & " was unexpected."

  for i, arg in params:
    if arg.kind == nnkInfix:
      if eqIdent(arg[0], "in"):
        values.add arg[1]
        tensors.add arg[2]
    elif arg.kind == nnkStmtList:
      # In generic proc, symbols are resolved early
      # the "in" symbol will be transformed into an opensymchoice of "contains"
      # Note that arg order in "contains" is inverted compared to "in"
      if arg[0].kind == nnkCall and arg[0][0].kind == nnkOpenSymChoice and eqident(arg[0][0][0], "contains"):
        values.add arg[0][2]
        tensors.add arg[0][1]
      else:
        syntaxError()
    else:
      syntaxError()

  #### Initialization
  let tensor0 = tensors[0]
  var testShape = newStmtList()
  for i in 1 ..< tensors.len:
    let tensor_i = tensors[i]
    testShape.add quote do:
          assert `tensor0`.shape == `tensor_i`.shape

  #### Generate nested for loops
  # And a case statement to select the proper one
  # at runtime
  var triotCase = nnkCaseStmt.newTree(
    newCall(bindSym"rank", tensor0)
  )

  for rank in 0 .. MAXRANK:
    var indices = nnkBracket.newTree()
    var tensors_indexed = nnkBracket.newTree()
    for i in countdown(rank - 1, 0): # For indexing highest rank is on the left
      indices.add genSym(nskForVar, "iter_dim" & $i & "_")
    for tensor in tensors:
      tensors_indexed.add nnkBracketExpr.newTree(tensor, indices)
    let newInnerBody = loopBody.replaceNodes(
      replacements = tensors_indexed,
      to_replace = values
    )
    let branchStmt = genNestedFor(
      newDotExpr(tensor0, bindSym"shape"),
      indices,
      newInnerBody
    )
    triotCase.add nnkOfBranch.newTree(
      newLit rank, branchStmt
    )

  result = quote do:
    `testShape`
    `triotCase`

#########################################################

proc sanityChecks() =
  # Sanity checks

  var x = randomTensor([5, 3], 10)
  let y = randomTensor([5, 3], 10)

  echo x
  echo y

  block:
    triotForEach i in x, j in y:
      i += j
    echo x

  block:
    let z = randomTensor([3, 5], 10).transpose()
    doAssert: not z.is_C_contiguous()
    echo z
    triotForEach i in x, j in z:
      i += j
    echo x

when isMainModule:
  sanityChecks()

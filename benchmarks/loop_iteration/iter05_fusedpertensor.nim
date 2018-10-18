# MIT License
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy

import
  ./tensor, ./tensor_display, ./metadata,
  ./compiler_optim_hints, ./utils,
  macros

macro fusedForEach*(args: varargs[untyped]): untyped =
  ## Warning: there is no mutability check

  result = newStmtList()

  var params = args
  var loopBody = params.pop()

  var values = nnkBracket.newTree()
  var tensors = nnkBracket.newTree()

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

  ### Initialization
  # First we need to alias the tensors, in our macro scope.
  # This is to ensure that an input like x[0..2, 1] isn't called multiple times
  # With move semantics this shouldn't cost anything.
  # We also take a pointer to the data
  var aliasesStmt = newStmtList()
  var aliases = nnkBracket.newTree()
  var dataPtrsDecl = newStmtList()
  var dataPtrs = nnkBracket.newTree()

  dataPtrsDecl.add newCall(bindSym"withCompilerOptimHints")

  for i, tensor in tensors:
    let alias = genSym(nskLet, "alias" & $i & '_' & $tensor & '_')
    aliases.add alias
    aliasesStmt.add newLetStmt(alias, tensor)

    let data_i = genSym(nskLet, "data" & $i & '_')
    dataPtrsDecl.add quote do:
      let `data_i`{.restrict.} = `alias`.dataPtr
    dataPtrs.add data_i

  let alias0 = aliases[0]
  var testShape = newStmtList()
  for i in 1 ..< aliases.len:
    let alias_i = aliases[i]
    testShape.add quote do:
      assert `alias0`.shape == `alias_i`.shape

  #### Deal with contiguous case
  var test_C_Contiguous = newCall(ident"is_C_contiguous", alias0)
  for i in 1 ..< aliases.len:
    test_C_Contiguous = newCall(
                      ident"and",
                      test_C_Contiguous,
                      newCall(ident"is_C_contiguous", aliases[i])
                      )

  let contiguousIndex = genSym(nskForVar, "contiguousIndex_")
  var dataPtrs_contiguous = nnkBracket.newTree()
  for dataPtr in dataPtrs:
    dataPtrs_contiguous.add nnkBracketExpr.newTree(dataPtr, contiguousIndex)
  let contiguousBody = loopBody.replaceNodes(replacements = dataPtrs_contiguous, to_replace = values)

  #### Deal with non-contiguous case
  var coord = genSym(nskVar, "coord_")
  var iter_pos = nnkBracket.newTree()
  var init_strided_iteration = newStmtList()
  var increment_iter_pos = newStmtList()
  var apply_backstrides = newStmtList()

  ####
  init_strided_iteration.add quote do:
    # withCompilerOptimHints() <- done in dataPtrsDecl
    var `coord` {.align64.}: array[MAXRANK, int]

  let k = genSym(nskForVar)

  for i, alias in aliases:
    let iter_pos_i = gensym(nskVar, "iter_pos" & $i)
    iter_pos.add iter_pos_i
    init_strided_iteration.add newVarStmt(iter_pos_i, newLit 0)
    increment_iter_pos.add quote do:
      `iter_pos_i` += `alias`.strides[`k`]
    apply_backstrides.add quote do:
      `iter_pos_i` -= `alias`.strides[`k`] * (`alias`.shape[`k`]-1)

  var dataPtrs_strided = nnkBracket.newTree()
  for i, dataPtr in dataPtrs:
    dataPtrs_strided.add nnkBracketExpr.newTree(dataPtr, iter_pos[i])
  let stridedBody = loopBody.replaceNodes(replacements = dataPtrs_strided, to_replace = values)

  let size = genSym(nskLet, "size_")
  let noncontiguousBody = quote do:
    # Initialisation
    `init_strided_iteration`

    # Iterator loop
    for _ in 0 ..< `size`:
      # Apply computation
      `stridedBody`

      # Next position
      for `k` in countdown(`alias0`.rank - 1, 0):
        if `coord`[`k`] < `alias0`.shape[`k`] - 1:
          `coord`[`k`] += 1
          `increment_iter_pos`
          break
        else:
          `coord`[`k`] = 0
          `apply_backstrides`

  result = quote do:
    block:
      `aliasesStmt`
      `testShape`
      `dataPtrsDecl`
      let `size` = `alias0`.size

      if `test_C_Contiguous`:
        for `contiguousIndex` in 0 ..< `size`:
          `contiguousBody`
      else:
        `noncontiguousBody`

#########################################################

proc sanityChecks() =
  # Sanity checks

  var x = randomTensor([5, 3], 10)
  let y = randomTensor([5, 3], 10)

  echo x
  echo y

  block:
    fusedForEach i in x, j in y:
      i += j
    echo x

  block:
    let z = randomTensor([3, 5], 10).transpose()
    doAssert: not z.is_C_contiguous()
    echo z
    fusedForEach i in x, j in z:
      i += j
    echo x

when isMainModule:
  sanityChecks()

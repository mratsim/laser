# Apache v2
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy and the Arraymancer contributors

# Current iteration scheme in Arraymancer. Each tensor manages it's own loop
import
  macros,
  ./tensor, ./compiler_optim_hints, ./metadata, ./utils,
  ./tensor_display

template initStridedIteration(backstrides, iter_pos: untyped, t: Tensor): untyped =
  ## Iterator init
  var iter_pos = 0
  withCompilerOptimHints()
  var coord {.align64, noInit.}: array[MAXRANK, int]
  var backstrides {.align64, noInit.}: array[MAXRANK, int]
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

proc genNestedFor(shape, iter_pos, strides, backstrides, indices, innerBody: NimNode): NimNode =
  # shape is a NimNode representing a Metadata
  # indices is a compile-time array corresponding to the iteration indices

  result = innerBody
  for i in countdown(indices.len - 1, 0):
    let idx = indices[i]
    result = quote do:
      for `idx` in 0 ..< `shape`[`i`]:
        `iter_pos` += `strides`[`i`]
        `result`
      iter_pos -= `backstrides`[`i`]

macro triotPerTensorForEach*(args: varargs[untyped]): untyped =
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
  var dataPtrsDecl = newStmtList()

  var dataPtrs = nnkBracket.newTree()
  for i, tensor in tensors:
    let data_i = genSym(nskLet, "data" & $i)
    dataPtrsDecl.add quote do:
      let `data_i`{.restrict.} = `tensor`.dataPtr
    dataPtrs.add data_i

  let tensor0 = tensors[0]
  var testShape = newStmtList()
  for i in 1 ..< tensors.len:
    let tensor_i = tensors[i]
    testShape.add quote do:
          assert `tensor0`.shape == `tensor_i`.shape

  #### Deal with contiguous case
  var testContiguous = newCall(ident"is_C_contiguous", tensors[0])
  for i in 1 ..< tensors.len:
    let tensor_i = tensors[i]
    testContiguous = newCall(
                      ident"and",
                      testContiguous,
                      newCall(ident"is_C_contiguous", tensor_i)
                      )

  let contiguousIndex = genSym(nskForVar, "contiguousIndex_")
  var dataPtrs_contiguous = nnkBracket.newTree()
  for dataPtr in dataPtrs:
    dataPtrs_contiguous.add nnkBracketExpr.newTree(dataPtr, contiguousIndex)
  let contiguousBody = loopBody.replaceNodes(replacements = dataPtrs_contiguous, to_replace = values)

  #### Deal with non-contiguous case
  var backstrides = nnkBracket.newTree()
  var iter_pos = nnkBracket.newTree()
  var stridedInits = newStmtList()
  var advanceStrided = newStmtList()

  for i in 0 ..< tensors.len:
    # We don't gensym here, the initStridedIteration template will do that
    backstrides.add ident("backstrides_t" & $i)
    iter_pos.add ident("iter_pos_t" & $i)
    stridedInits.add newCall(
      bindSym"initStridedIteration",
      backstrides[^1], iter_pos[^1], tensors[i]
    )

  var dataPtrs_strided = nnkBracket.newTree()
  for i, dataPtr in dataPtrs:
    dataPtrs_strided.add nnkBracketExpr.newTree(dataPtr, iter_pos[i])
  let stridedBody = loopBody.replaceNodes(replacements = dataPtrs_strided, to_replace = values)

  ## Triot case statement + iteration
  ## to select the proper nested for loop at runtime
  var triotCases = nnkBracket.newTree()

  for i in 0 ..< tensors.len:
    var triotCase = nnkCaseStmt.newTree(
      newCall(bindSym"rank", tensors[i])
    )
    for rank in 0 .. MAXRANK:
      var indices = nnkBracket.newTree() # The indices are actually not used during iteration
      for i in countdown(rank - 1, 0): # For indexing highest rank is on the left
        indices.add genSym(nskForVar, "iter_dim" & $i & "_")
      let branchStmt = genNestedFor(
        newDotExpr(tensors[i], bindSym"shape"),
        iter_pos[i],
        newDotExpr(tensors[i], ident"strides"),
        backstrides[i],
        indices,
        stridedBody
      )
      triotCase.add nnkOfBranch.newTree(
        newLit rank, branchStmt
      )
    triotCases.add triotCase

  ## Unfinished, how to yield at the right time?

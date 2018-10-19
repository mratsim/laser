# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Strided parallel iteration for tensors

import
  macros,
  ../private/ast_utils, ../compiler_optim_hints,
  ../openmp/omp_parallel

template isVar[T: object](x: T): bool =
  ## Workaround due to `is` operator not working for `var`
  ## https://github.com/nim-lang/Nim/issues/9443
  compiles(addr(x))

proc initForEach(
        args: NimNode,
        params: var NimNode,
        loopBody: var NimNode,
        omp_params: var NimNode,
        values, aliases, raw_ptrs: var NimNode,
        aliases_stmt, raw_ptrs_stmt: var NimNode,
        test_shapes: var NimNode
      ) =

  ### Parse the input
  params = args
  loopBody = params.pop()
  values = nnkBracket.newTree()
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
    elif getType(arg) is tuple:
      omp_params = arg
    else:
      syntaxError()

  ### Initialization
  # First we need to alias the tensors, in our macro scope.
  # This is to ensure that an input like x[0..2, 1] isn't called multiple times
  # With move semantics this shouldn't cost anything.
  # We also take a pointer to the data
  aliases_stmt = newStmtList()
  aliases = nnkBracket.newTree()
  raw_ptrs_stmt = newStmtList()
  raw_ptrs = nnkBracket.newTree()

  raw_ptrs_stmt.add newCall(bindSym"withCompilerOptimHints")

  for i, tensor in tensors:
    let alias = newIdentNode($tensor & "_alias" & $i & '_')
    aliases.add alias
    aliases_stmt.add quote do:
      when isVar(`tensor`):
        var `alias` = `tensor`
      else:
        let `alias` = `tensor`

    let raw_ptr_i = genSym(nskLet, $tensor & "_raw_data" & $i & '_')
    raw_ptrs_stmt.add quote do:
      let `raw_ptr_i`{.restrict.} = `alias`.unsafe_raw_data()
    raw_ptrs.add raw_ptr_i

  let alias0 = aliases[0]
  test_shapes = newStmtList()
  for i in 1 ..< aliases.len:
    let alias_i = aliases[i]
    test_shapes.add quote do:
      assert `alias0`.shape == `alias_i`.shape

proc forEachContiguousImpl(
  values, raw_ptrs, size, loopBody, omp_params: NimNode,
  ): NimNode =
  # Build the parallel body of a contiguous iterator

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  let body = loopBody.replaceNodes(
                  replacements = elems_contiguous,
                  to_replace = values
                  )

  result = newStmtList()
  if omp_params.isNil:
    result.add quote do:
      omp_parallel_for_default(`index`, `size`):
          `body`
  else:
    let
      omp_threshold  = omp_params[0]
      omp_grain_size = omp_params[1]
      use_simd       = omp_params[2]
    result.add quote do:
      omp_parallel_for(
        `index`, `size`,
        `omp_threshold`, `omp_grain_size`, `use_simd`):
          `body`

proc forEachStridedImpl(
  values, aliases,
  raw_ptrs, size,
  loopBody, omp_params: NimNode,
  ): NimNode =
  # Build the parallel body of a strided iterator

  var iter_pos = nnkBracket.newTree()
  var init_strided_iteration = newStmtList()
  var iter_start_offset = newStmtList()
  var increment_iter_pos = newStmtList()
  var apply_backstrides = newStmtList()

  let
    alias0 = aliases[0]
    coord = genSym(nskVar, "coord_")
    j = genSym(nskForVar, "j_mem_offset_") # Setting the start offset of each tensor iterator during init
    k = genSym(nskForVar, "k_next_elem_")  # Computing the next element in main body loop
    chunk_offset = newIdentNode("chunk_offset_")
    chunk_size = newIdentNode("chunk_size_")

  init_strided_iteration.add quote do:
    var `coord` {.align_array.}: array[LASER_MEM_ALIGN, int]

  for i, alias in aliases:
    let iter_pos_i = gensym(nskVar, "iter" & $i & "_pos_")
    iter_pos.add iter_pos_i
    init_strided_iteration.add newVarStmt(iter_pos_i, newLit 0)
    iter_start_offset.add quote do:
      `iter_pos_i` += `coord`[`j`] * `alias`.strides[`j`]
    increment_iter_pos.add quote do:
      `iter_pos_i` += `alias`.strides[`k`]
    apply_backstrides.add quote do:
      `iter_pos_i` -= `alias`.strides[`k`] * (`alias`.shape[`k`]-1)

  # Now add the starting memory offset to the init
  init_strided_iteration.add quote do:
    if `chunk_offset` != 0:
      var z = 1
      for `j` in countdown(`alias0`.rank - 1, 0):
        `coord`[`j`] = (`chunk_offset` div z) mod `alias0`.shape[`j`]
        `iter_start_offset`
        z *= `alias0`.shape[`j`]

  var elems_strided = nnkBracket.newTree()
  for i, raw_ptr in raw_ptrs:
    elems_strided.add nnkBracketExpr.newTree(raw_ptr, iter_pos[i])

  let body = loopBody.replaceNodes(replacements = elems_strided, to_replace = values)
  let stridedBody = quote do:
    # Initialisation
    `init_strided_iteration`

    # Iterator loop
    for _ in 0 ..< `chunk_size`:
      # Apply computation
      `body`

      # Next position
      for `k` in countdown(`alias0`.rank - 1, 0):
        if `coord`[`k`] < `alias0`.shape[`k`] - 1:
          `coord`[`k`] += 1
          `increment_iter_pos`
          break
        else:
          `coord`[`k`] = 0
          `apply_backstrides`

  result = newStmtList()
  if omp_params.isNil:
    result.add quote do:
      omp_parallel_chunks_default(`size`, `chunk_offset`, `chunk_size`):
          `stridedBody`
  else:
    let
      omp_threshold  = omp_params[0]
      omp_grain_size = omp_params[1]
      use_simd       = omp_params[2]
    result.add quote do:
      omp_parallel_chunks(
        `size`, `chunk_offset`, `chunk_size`,
        `omp_threshold`, `omp_grain_size`, `use_simd`):
          `stridedBody`

macro forEachContiguous*(args: varargs[untyped]): untyped =
  ## Format:
  ## forEachContiguous x in a, y in b, z in c, (512, 1024, true):
  ##    x += y * z
  ## (512, 1024, true) corresponds to omp_threshold, omp_grain_size, use_simd
  ## from omp_parallel_for
  var
    params, loopBody, values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode
    omp_params: NimNode

  initForEach(
        args,
        params,
        loopBody,
        omp_params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  let parallel_body = forEachContiguousImpl(
    values, raw_ptrs, size, loopBody, omp_params
    )
  let alias0 = aliases[0]

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      `parallel_body`

macro forEachStrided*(args: varargs[untyped]): untyped =
  ## Format:
  ## forEachStrided x in a, y in b, z in c, (512, 1024, true):
  ##    x += y * z
  ## (512, 1024, true) corresponds to omp_threshold, omp_grain_size, use_simd
  ## from omp_parallel_for
  var
    params, loopBody, values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode
    omp_params: NimNode

  initForEach(
        args,
        params,
        loopBody,
        omp_params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  let parallel_body = forEachStridedImpl(
    values, aliases, raw_ptrs, size, loopBody, omp_params
  )
  let alias0 = aliases[0]

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      `parallel_body`

macro forEach*(args: varargs[untyped]): untyped =
  ## Format:
  ## forEach x in a, y in b, z in c, (512, 1024, true):
  ##    x += y * z
  ## (512, 1024, true) corresponds to omp_threshold, omp_grain_size, use_simd
  ## from omp_parallel_for
  ##
  ## The iteration strategy is selected at runtime depending of
  ## the tensors memory layout. If you know at compile-time that the tensors are
  ## contiguous or strided, use forEachContiguous or forEachStrided instead.
  ## Runtime selection requires duplicating the code body.
  var
    params, loopBody, values, aliases, raw_ptrs: NimNode
    aliases_stmt, raw_ptrs_stmt, test_shapes: NimNode
    omp_params: NimNode

  initForEach(
        args,
        params,
        loopBody,
        omp_params,
        values, aliases, raw_ptrs,
        aliases_stmt, raw_ptrs_stmt,
        test_shapes
  )

  let size = genSym(nskLet, "size_")
  let contiguous_body = forEachContiguousImpl(
    values, raw_ptrs, size, loopBody, omp_params
  )
  let strided_body = forEachStridedImpl(
    values, aliases, raw_ptrs, size, loopBody, omp_params
  )
  let alias0 = aliases[0]
  var test_C_Contiguous = newCall(ident"is_C_contiguous", alias0)
  for i in 1 ..< aliases.len:
    test_C_Contiguous = newCall(
                      ident"and",
                      test_C_Contiguous,
                      newCall(ident"is_C_contiguous", aliases[i])
                      )

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      if `test_C_Contiguous`:
        `contiguous_body`
      else:
        `strided_body`

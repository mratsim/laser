# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Strided parallel reduction for tensors
# This is generic and work on any tensor types as long
# as it implement the following interface:
#
# Tensor[T]:
#   Tensors data storage backend must be shallow copied on assignment (reference semantics)
#   the macro works on aliases to ensure that if the tensor is the result of another routine
#   that routine is only called once, for example x[0..<2, _] will not slice `x` multiple times.
#
# Routine and fields used, (routines mean proc, template, macros):
#   - rank, size:
#       routines or fields that return an int
#   - shape, strides:
#       routines or fields that returns an array, seq or indexable container
#       that supports `[]`. Read-only access.
#   - unsafe_raw_data:
#       rountine or field that returns a ptr UncheckedArray[T]
#       or a distinct type with `[]` indexing implemented.
#       The address should be the start of the raw data including
#       the eventual tensor offset for subslices, i.e. equivalent to
#       the address of x[0, 0, 0, ...]
#       Needs mutable access for var tensor.
#
# Additionally the reduceEach macro needs an `is_C_contiguous` routine

import
  macros,
  ./foreach_common,
  ../private/ast_utils,
  ../openmp
export omp_suffix # Pending https://github.com/nim-lang/Nim/issues/9365 or 9366

proc reduceContiguousImpl(
  nb_chunks, chunk_offset, chunk_size,
  values, raw_ptrs, size, loopBody: NimNode,
  use_openmp: static bool, omp_params: NimNode): NimNode =

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  var body = loopBody.replaceNodes(
                  replacements = elems_contiguous,
                  to_replace = values
                  )

  if use_openmp:
    body = quote do:
      for `index` in `chunk_offset` ..< `chunk_offset` + `chunk_size`:
        `body`

    if omp_params.isNil:
      result = quote do:
        omp_parallel_chunks_default(
              `size`, `nb_chunks`,
              `chunk_offset`, `chunk_size`):
            `body`
    else:
      let
        omp_grain_size = omp_params[0]
        use_simd       = omp_params[1]
      result = quote do:
        omp_parallel_chunks(
            `size`, `nb_chunks`,
            `chunk_offset`, `chunk_size`,
            `omp_grain_size`, `use_simd`):
          `body`
  else:
    result = quote do:
      `nb_chunks` = 1
      const `chunk_offset`{.inject.} = 0
      let `chunk_size`{.inject.} = `size`
      for index in `chunk_offset` ..< `chunk_offset` + `chunk_size`:
        `body`

proc reduceStridedImpl(
  nb_chunks, chunk_offset, chunk_size,
  values, aliases, raw_ptrs, size, loopBody: NimNode,
  use_openmp: static bool, omp_params: NimNode): NimNode =

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

  init_strided_iteration.add quote do:
    var `coord` {.align_variable.}: array[LASER_MEM_ALIGN, int]

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
      var accum_size = 1
      for `j` in countdown(`alias0`.rank - 1, 0):
        `coord`[`j`] = (`chunk_offset` div accum_size) mod `alias0`.shape[`j`]
        `iter_start_offset`
        accum_size *= `alias0`.shape[`j`]

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

  if use_openmp:
    let
      omp_grain_size =  if omp_params.isNil: newLit( # scale grain_size down for strided operation
                          OMP_MEMORY_BOUND_GRAIN_SIZE div OMP_NON_CONTIGUOUS_SCALE_FACTOR
                        ) else: newLit(
                          omp_params[0].intVal div OMP_NON_CONTIGUOUS_SCALE_FACTOR
                        )
      use_simd       = if omp_params.isNil: newLit true else: omp_params[1]
    result = quote do:
      omp_parallel_chunks(
            `size`, `nb_chunks`,
            `chunk_offset`, `chunk_size`,
            `omp_grain_size`, `use_simd`):
        `stridedBody`
  else:
    result = quote do:
      `nb_chunks` = 1
      const `chunk_offset`{.inject.} = 0
      let `chunk_size`{.inject.} = `size`
      for index in `chunk_offset` ..< `chunk_offset` + `chunk_size`:
        `stridedBody`

template reduceEachContiguousTemplate(use_openmp: static bool){.dirty.} =
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

  let
    size = genSym(nskLet, "size_")
    chunk_offset = newIdentNode("chunk_offset_")
    chunk_size = newIdentNode("chunk_size_")
  let contiguous_body = reduceContiguousImpl(
    nb_chunks, chunk_offset, chunk_size,
    values, raw_ptrs, size, loopBody,
    use_openmp, omp_params
  )
  let alias0 = aliases[0]

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      `contiguous_body`

template reduceEachStridedTemplate(use_openmp: static bool){.dirty.} =
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

  let
    size = genSym(nskLet, "size_")
    chunk_offset = newIdentNode("chunk_offset_")
    chunk_size = newIdentNode("chunk_size_")
  let strided_body = reduceStridedImpl(
    nb_chunks, chunk_offset, chunk_size,
    values, aliases, raw_ptrs, size, loopBody,
    use_openmp, omp_params
    )
  let alias0 = aliases[0]

  result = quote do:
    block:
      `aliases_stmt`
      `test_shapes`
      `raw_ptrs_stmt`
      let `size` = `alias0`.size
      `strided_body`

template reduceEachTemplate(use_openmp: static bool){.dirty.} =
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

  let
    size = genSym(nskLet, "size_")
    chunk_offset = newIdentNode("chunk_offset_")
    chunk_size = newIdentNode("chunk_size_")
  let contiguous_body = reduceContiguousImpl(
    nb_chunks, chunk_offset, chunk_size,
    values, raw_ptrs, size, loopBody,
    use_openmp, omp_params
  )
  let strided_body = reduceStridedImpl(
    nb_chunks, chunk_offset, chunk_size,
    values, aliases, raw_ptrs, size, loopBody,
    use_openmp, omp_params
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

macro reduceEachContiguous*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ##   var partial_sums{.align_variable.}: newSeq[int](omp_get_max_threads() * padding)
  ##   var nb_chunks: Natural
  ##   reduceEachContiguous nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      partial_sums[omp_get_thread_num() * padding] += y * z
  ##
  ## (1024, true) corresponds to omp_grain_size, use_simd
  ## from omp_parallel_for
  reduceEachContiguousTemplate(true)

macro reduceEachContiguousSerial*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ##   var nb_chunks: Natural
  ##   reduceEachContiguous nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      result += y * z
  ##
  ## OpenMP parameters will be ignored
  reduceEachContiguousTemplate(false)

macro reduceEachStrided*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ##   var partial_sums{.align_variable.}: newSeq[int](omp_get_max_threads() * padding)
  ##   var nb_chunks: Natural
  ##   reduceEachContiguous nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      partial_sums[omp_get_thread_num() * padding] += y * z
  ##
  ## The OpenMP minimal per-core grain size
  ## is always scaled down by OMP_NON_CONTIGUOUS_SCALE_FACTOR (4 by default)
  reduceEachStridedTemplate(true)

macro reduceEachStridedSerial*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ## Format:
  ##   var nb_chunks: Natural
  ##   reduceEachStridedSerial nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      result += y * z
  ##
  ## Strided iteration with serial execution. OpenMP params passed to it will be ignored
  reduceEachStridedTemplate(false)

macro reduceEach*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ##   var partial_sums{.align_variable.}: newSeq[int](omp_get_max_threads() * padding)
  ##   var nb_chunks: Natural
  ##   reduceEach nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      partial_sums[omp_get_thread_num() * padding] += y * z
  ##
  ## (1024, true) corresponds to omp_grain_size, use_simd
  ## from omp_parallel_for
  ##
  ## The iteration strategy is selected at runtime depending of
  ## the tensors memory layout. If you know at compile-time that the tensors are
  ## contiguous or strided, use reduceEachContiguous or reduceEachStrided instead.
  ## Runtime selection requires duplicating the code body.
  ##
  ## If the tensors are non-contiguous, the OpenMP minimal per-core grain size
  ## is scaled down by OMP_NON_CONTIGUOUS_SCALE_FACTOR (4 by default)
  reduceEachTemplate(true)

macro reduceEachSerial*(nb_chunks: var Natural, args: varargs[untyped]): untyped =
  ## Format:
  ##   var nb_chunks: Natural
  ##   reduceEachSerial nb_chunks, x in a, y in b, z in c, (1024, true):
  ##      partial_sums += y * z
  ##
  ## OpenMP parameters will be ignored
  ##
  ## The iteration strategy is selected at runtime depending of
  ## the tensors memory layout. If you know at compile-time that the tensors are
  ## contiguous or strided, use reduceEachContiguousSerial or reduceEachStridedSerial instead.
  ## Runtime selection requires duplicating the code body.
  reduceEachTemplate(false)

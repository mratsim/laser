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
# Additionally the forEach macro needs an `is_C_contiguous` routine

import
  macros,
  ./foreach_common,
  ../private/ast_utils,
  ../openmp/[omp_parallel, omp_tuning]
export omp_suffix # Pending https://github.com/nim-lang/Nim/issues/9365 or 9366

proc reduceContiguousImpl(
  chunk_offset, chunk_size,
  values, raw_ptrs, size, loopBody: NimNode): NimNode =

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  let body = loopBody.replaceNodes(
                  replacements = elems_contiguous,
                  to_replace = values
                  )

  result = quote do:
    for `index` in `chunk_offset` ..< `chunk_size`:
      `body`

proc reduceStridedImpl(
  chunk_offset, chunk_size,
  values, aliases, raw_ptrs, size, loopBody: NimNode): NimNode =

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
  result = quote do:
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

macro forEachReduce*(nb_chunks: var Natural, chunk_id: untyped, args: varargs[untyped]): untyped =
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
    chunk_offset, chunk_size, values, raw_ptrs, size, loopBody
  )
  let strided_body = reduceStridedImpl(
    chunk_offset, chunk_size, values, aliases, raw_ptrs, size, loopBody
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
        omp_parallel_chunks(
              `size`, `nb_chunks`,
              `chunk_id`, `chunk_offset`, `chunk_size`,
              omp_threshold = OMP_MEMORY_BOUND_THRESHOLD,
              omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
              use_simd = true):
          `contiguous_body`
      else:
        omp_parallel_chunks(
              `size`, `nb_chunks`,
              `chunk_id`, `chunk_offset`, `chunk_size`,
              omp_threshold = OMP_MEMORY_BOUND_THRESHOLD,
              omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
              use_simd = true):
          `strided_body`

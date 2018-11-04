# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# This file implements the forEachStaged macro which allows multi-stage parallel for loop
# on a variadic number of tensors

import
  macros,
  ./foreach_common,
  ../private/ast_utils,
  ../openmp
export omp_suffix

template omp_parallel_threshold(size, threshold: Natural, body: untyped) =
  {.emit: "#pragma omp parallel if (`size` < `threshold`)".}
  block: body

proc forEachStagedContiguousImpl(
  values, raw_ptrs, size, loopBody: NimNode,
  use_simd: static bool,
  ): NimNode =
  # Build the body of a contiguous iterator
  # Whether this is parallelized or not should be
  # handled at a higher level

  let index = newIdentNode("contiguousIndex_")
  var elems_contiguous = nnkBracket.newTree()
  for raw_ptr in raw_ptrs:
    elems_contiguous.add nnkBracketExpr.newTree(raw_ptr, index)

  let body = loopBody.replaceNodes(
                  replacements = elems_contiguous,
                  to_replace = values
                  )

  result = getAST(
    omp_for,
    index, size, use_simd, loopBody
  )

proc forEachStagedStridedImpl(
  values, aliases,
  raw_ptrs, size,
  loopBody: NimNode,
  use_openmp: static bool
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
    chunk_size =  if use_openmp: newIdentNode("chunk_size_")
                  else: size

  init_strided_iteration.add quote do:
    var `coord` {.align_variable.}: array[LASER_MEM_ALIGN, int]

  stridedVarsSetup()

  # Now add the starting memory offset to the init
  if use_openmp:
    init_strided_iteration.add stridedChunkOffset()

  var elems_strided = nnkBracket.newTree()
  for i, raw_ptr in raw_ptrs:
    elems_strided.add nnkBracketExpr.newTree(raw_ptr, iter_pos[i])

  let body = loopBody.replaceNodes(replacements = elems_strided, to_replace = values)
  let stridedBody = stridedBodyTemplate()

  if use_openmp:
    result = getAST(
      omp_chunks,
      size, chunk_offset, chunk_size,
      stridedBody
    )
  else:
    result = stridedBody

template forEachStagedSimpleTemplate(contiguous: static bool){.dirty.} =
  let body =  if contiguous:
                forEachStagedContiguousImpl(
                  values, raw_ptrs, size, in_loop_body, use_simd
                )
              else:
                forEachStagedStridedImpl(
                  values, aliases, raw_ptrs, size, in_loop_body, use_openmp
                )
  let alias0 = aliases[0]

  if use_openmp:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        `before_loop_body`
        `body`
        `after_loop_body`

  else:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        omp_parallel_threshold(size, omp_threshold):
          `before_loop_body`
          `body`
          `after_loop_body`

template forEachStagedTemplate(){.dirty.} =
  let contiguous_body = forEachStagedContiguousImpl(
                          values, raw_ptrs, size, in_loop_body, use_simd
                        )
  let strided_body =  forEachStagedStridedImpl(
                        values, aliases, raw_ptrs, size, in_loop_body, use_openmp
                      )

  let alias0 = aliases[0]
  var test_C_Contiguous = newCall(ident"is_C_contiguous", alias0)
  for i in 1 ..< aliases.len:
    test_C_Contiguous = newCall(
                          ident"and",
                          test_C_Contiguous,
                          newCall(ident"is_C_contiguous", aliases[i])
                        )
  if use_openmp:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        `before_loop_body`
        if `test_C_Contiguous`:
          `contiguous_body`
        else:
          `strided_body`
        `after_loop_body`

  else:
    result = quote do:
      block:
        `aliases_stmt`
        `test_shapes`
        `raw_ptrs_stmt`
        let `size` = `alias0`.size
        let is_C_contiguous = `test_C_Contiguous`
        omp_parallel_threshold(size, omp_threshold):
          `before_loop_body`
          if is_C_contiguous:
            `contiguous_body`
          else:
            `strided_body`
          `after_loop_body`

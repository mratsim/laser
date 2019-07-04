# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internals
  ../platforms,
  ../../private/align_unroller

proc vectorize*(
      funcName: NimNode,
      ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
      len: NimNode,
      arch: SimdArch, alignNeeded, unroll_factor: int): NimNode =
  ## Vectorizing macro
  ## Apply a SIMD function on all elements of an array
  ## This deals with:
  ##   - indexing
  ##   - unrolling
  ##   - alignment
  ##   - any number of parameters and result

  # It does the same as the following templates
  #
  # template vectorize(
  #       wrapped_func,
  #       funcname: untyped,
  #       arch: static SimdArch,
  #       alignNeeded,
  #       unroll_factor: static int) =
  #   proc funcname(dst, src: ptr UncheckedArray[float32], len: Natural) =
  #
  #     template srcAlign {.dirty.} = cast[ByteAddress](src[idx].addr) and (alignNeeded - 1)
  #     template dstAlign {.dirty.} = cast[ByteAddress](dst[idx].addr) and (alignNeeded - 1)
  #
  #     doAssert srcAlign == dstAlign
  #
  #     # Loop peeling, while not aligned to required alignment
  #     var idx = 0
  #     while srcAlign() != 0:
  #       dst[idx] = wrapped_func(src[idx])
  #       inc idx
  #
  #     # Aligned part
  #     {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  #     let srca {.restrict.} = assume_aligned cast[ptr UncheckedArray[float32]](src[idx].addr)
  #     let dsta {.restrict.} = assume_aligned cast[ptr UncheckedArray[float32]](dst[idx].addr)
  #
  #     let newLen = len - idx
  #     let unroll_stop = newLen.round_down_power_of_2(unroll_factor)
  #     for i in countup(0, unroll_stop - 1, unroll_factor):
  #       simd(
  #         arch, simdStoreA,
  #         dst[i].addr,
  #         wrapped_func(
  #           simd(arch, simdLoadA, src[i].addr)
  #         )
  #       )
  #
  #     # Unrolling remainder
  #     for i in unroll_stop ..< len:
  #       dst[i] = wrapped_func(src[i])

  template alignmentOffset(p: NimNode, idx: NimNode): untyped {.dirty.}=
    quote:
      cast[ByteAddress](`p`[`idx`].addr) and (`alignNeeded` - 1)

  result = newStmtList()

  block: # Alignment
    let align0 = alignmentOffset(ptrs.inParams[0], newLit 0)
    for i in 1 ..< ptrs.inParams.len:
      let align_i = alignmentOffset(ptrs.inParams[i], newLit 0)
      result.add quote do:
        doAssert `align0` == `align_i`
    for outparam in ptrs.outParams:
      let align_i =  alignmentOffset(outparam, newLit 0)
      result.add quote do:
        doAssert `align0` == `align_i`

  let idxPeeling = newIdentNode("idxPeeling_")


  proc elems(idx: NimNode, simd: bool): tuple[fcall, dst, dst_init, dst_assign: NimNode] =
    ## Note we need a separate ident node for each for loops
    ## otherwise the C codegen is wrong
    # Src params / Function call
    result.fcall = nnkCall.newTree()
    result.fcall.add funcName
    for p in ptrs.inParams:
      let elem = nnkBracketExpr.newTree(p, idx)
      if not simd:
        result.fcall.add elem
      else:
        result.fcall.add newCall(
          SimdTable[arch][simdLoadU], # Hack: should be aligned but no control over alignment in seq[T]
          newCall(
            newidentNode"addr",
            elem
          )
        )

    # Destination params
    # Assuming we have a function called the following way
    # (r0, r1) = foo(s0, s1)
    # We can use tuples for the non-SIMD part
    # but we will need temporaries for the SIMD part
    # before calling simdStore

    # temp variable around bug? can't use result.dst_init[0].add, type mismatch on tuple sig
    var dst_init = nnkVarSection.newTree(
      nnkIdentDefs.newTree()
    )
    result.dst_assign = newStmtList()

    if ptrs.outParams.len > 1:
      result.dst = nnkPar.newTree()
      for p in ptrs.outParams:
        let elem = nnkBracketExpr.newTree(p, idx)
        if not simd:
          result.dst.add elem
        else:
          let tmp = newIdentNode($p & "_simd")
          result.dst.add tmp
          dst_init[0].add nnkPragmaExpr.newTree(
            tmp,
            nnkPragma.newTree(
              newIdentNode"noInit"
              )
          )
          result.dst_assign.add newCall(
            SimdTable[arch][simdStoreU], # Hack: should be aligned but no control over alignment in seq[T]
            newCall(
              newidentNode"addr",
              elem
            ),
            tmp
          )
    elif ptrs.outParams.len == 1:
      let elem = nnkBracketExpr.newTree(ptrs.outParams[0], idx)
      if not simd:
        result.dst = elem
      else:
        let tmp = newIdentNode($ptrs.outParams[0] & "_simd")
        result.dst = tmp
        result.dst_assign.add newCall(
          SimdTable[arch][simdStoreU], # Hack: should be aligned but no control over alignment in seq[T]
          elem,
          tmp
        )

    dst_init[0].add SimdTable[arch][simdType]
    dst_init[0].add newEmptyNode()

    result.dst_init = dst_init

  block: # Loop peeling
    let idx = newIdentNode("idx_")
    result.add newVarStmt(idxPeeling, newLit 0)
    let whileTest = nnkInfix.newTree(
      newIdentNode"!=",
      alignmentOffset(ptrs.inParams[0], idxPeeling),
      newLit 0
    )
    var whileBody = newStmtList()
    let (fcall, dst, _, _) = elems(idx, simd = false)

    whileBody.add newLetStmt(idx, idxPeeling)
    if ptrs.outParams.len > 0:
      whileBody.add newAssignment(dst, fcall)
    else:
      whileBody.add fcall
    whileBody.add newCall(newIdentNode"inc", idxPeeling)

    result.add nnkWhileStmt.newTree(
      whileTest,
      whileBody
    )

  let unroll_stop = newIdentNode("unroll_stop_")
  block: # Aligned part
    let idx = newIdentNode("idx_")
    result.add quote do:
      let `unroll_stop` = round_step_down(
        `len` - `idxPeeling`, `unroll_factor`)

    let (fcall, dst, dst_init, dst_assign) = elems(idx, simd = true)
    if ptrs.outParams.len > 0:
      result.add dst_init

    var forStmt = nnkForStmt.newTree()
    forStmt.add idx
    forStmt.add newCall(
      newIdentNode"countup",
      idxPeeling,
      nnkInfix.newTree(
        newIdentNode"-",
        unroll_stop,
        newLit 1
      ),
      newLit unroll_factor
    )
    if ptrs.outParams.len > 0:
      forStmt.add nnkStmtList.newTree(
        newAssignment(dst, fcall),
        dst_assign
      )
    else:
      forStmt.add fcall
    result.add forStmt

  block: # Remainder
    let idx = newIdentNode("idx_")
    var forStmt = nnkForStmt.newTree()
    forStmt.add idx
    forStmt.add nnkInfix.newTree(
        newIdentNode"..<",
        unroll_stop,
        len
      )
    let (fcall, dst, _, _) = elems(idx, simd = false)
    if ptrs.outParams.len > 0:
      forStmt.add newAssignment(dst, fcall)
    else:
      forStmt.add fcall
    result.add forStmt

  # echo result.toStrLit

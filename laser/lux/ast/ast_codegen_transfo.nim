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

proc alignmentOffset(arch: SimdArch, p: NimNode, idx: NimNode): NimNode =
  let alignNeeded = SimdAlignment[arch]
  quote:
    cast[ByteAddress](`p`) and (`alignNeeded` - 1)

proc vecChecks(
        arch: SimdArch,
        ptrs: tuple[inParams, outParams: seq[NimNode]],
      ): NimNode =

  # TODO: we probably need those checks at runtime to avoid failure
  # or only allow vectorization where alignment is provable:
  #   - from infering from previous allocation
  #   - when iterating over a single tensor
  # Otherwise always use unaligned loads

  # TODO: check contiguous

  result = newStmtList()
  let align0 = arch.alignmentOffset(ptrs.inParams[0], newLit 0)
  for i in 1 ..< ptrs.inParams.len:
    let align_i = arch.alignmentOffset(ptrs.inParams[i], newLit 0)
    result.add quote do:
      doAssert `align0` == `align_i`
  for outparam in ptrs.outParams:
    let align_i =  arch.alignmentOffset(outparam, newLit 0)
    result.add quote do:
      doAssert `align0` == `align_i`

proc setupDstElems(
    funcName: NimNode,
    arch: SimdArch,
    T: NimNode,
    idx: NimNode,
    ptrs: tuple[inParams, outParams: seq[NimNode]],
    simd: bool
  ): tuple[fcall, dst, dst_init, dst_assign: NimNode] =
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
        SimdMap(arch, T, simdLoadA),
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
          SimdMap(arch, T, simdStoreA),
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
        SimdMap(arch, T, simdStoreA),
        elem,
        tmp
      )

  dst_init[0].add SimdMap(arch, T, simdType)
  dst_init[0].add newEmptyNode()

  result.dst_init = dst_init

  # echo "###########"
  # echo result.fcall.toStrLit
  # echo "-----------"
  # echo result.dst.toStrLit
  # echo "-----------"
  # echo result.dst_init.toStrLit
  # echo "-----------"
  # echo result.dst_assign.toStrLit

proc vecPrologue*(
      funcName: NimNode,
      arch: SimdArch,
      T: NimNode,
      idxPeeling: NimNode,
      ptrs: tuple[inParams, outParams: seq[NimNode]],
    ): NimNode =

  result = newStmtList()

  let idx = newIdentNode("idx_")
  result.add newVarStmt(idxPeeling, newLit 0)
  let whileTest = nnkInfix.newTree(
    newIdentNode"!=",
    arch.alignmentOffset(ptrs.inParams[0], idxPeeling),
    newLit 0
  )
  var whileBody = newStmtList()
  let (fcall, dst, _, _) = setupDstElems(funcName, arch, T, idx, ptrs, simd = false)

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

proc vecKernel*(
        funcName: NimNode,
        arch: SimdArch,
        T: NimNode,
        idxPeeling, unroll_stop, len: NimNode,
        ptrs: tuple[inParams, outParams: seq[NimNode]]
      ): NimNode =
  result = newStmtList()

  let idx = newIdentNode("idx_")
  let unroll_factor = ident"unroll_factor"
  result.add quote do:
    const `unroll_factor` = elemsPerVector(SimdArch(`arch`), `T`)
    let `unroll_stop` = round_step_down(
      `len` - `idxPeeling`, `unroll_factor`)

  let (fcall, dst, dst_init, dst_assign) = setupDstElems(funcName, arch, T, idx, ptrs, simd = true)
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
    unroll_factor
  )
  if ptrs.outParams.len > 0:
    forStmt.add nnkStmtList.newTree(
      newAssignment(dst, fcall),
      dst_assign
    )
  else:
    forStmt.add fcall
  result.add forStmt

proc vecEpilogue*(
        funcName: NimNode,
        arch: SimdArch,
        T: NimNode,
        unroll_stop, len: NimNode,
        ptrs: tuple[inParams, outParams: seq[NimNode]]
      ): NimNode =

  result = newStmtList()

  let idx = newIdentNode("idx_")
  var forStmt = nnkForStmt.newTree()
  forStmt.add idx
  forStmt.add nnkInfix.newTree(
      newIdentNode"..<",
      unroll_stop,
      len
    )
  let (fcall, dst, _, _) = setupDstElems(funcName, arch, T, idx, ptrs, simd = false)
  if ptrs.outParams.len > 0:
    forStmt.add newAssignment(dst, fcall)
  else:
    forStmt.add fcall
  result.add forStmt

proc vectorize*(
      funcName: NimNode,
      ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
      len: NimNode,
      arch: SimdArch, T: NimNode): NimNode =
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

  result = newStmtList()
  result = vecChecks(arch, ptrs)

  let idxPeeling = newIdentNode("idxPeeling_")
  result.add vecPrologue(funcName, arch, T, idxPeeling, ptrs)

  let unroll_stop = newIdentNode("unroll_stop_")
  result.add veckernel(funcName, arch, T, idxPeeling, unroll_stop, len, ptrs)

  result.add vecEpilogue(funcName, arch, T, unroll_stop, len, ptrs)

  # echo result.toStrLit

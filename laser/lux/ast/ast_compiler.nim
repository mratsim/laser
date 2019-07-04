# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ../platforms,
  ./ast_definition,
  ./ast_sigmatch,
  ./ast_codegen,
  ./ast_codegen_transfo,
  ./macro_utils

# TODO: Do we need both compile and generate?

proc initParams(
       procDef,
       resultType: NimNode
       ): tuple[
            ids: seq[NimNode],
            ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
            length: NimNode,
            initStmt: NimNode
          ] =
  # Get the idents from proc definition. We order the same as proc def
  # Start with non-result
  # We work at simd vector level
  result.initStmt = newStmtList()
  let type0 = newCall(
    newIdentNode"type",
    nnkBracketExpr.newTree(
      procDef[0][3][1][0],
      newLit 0
    )
  )

  for i in 1 ..< procDef[0][3].len: # Proc formal params
    let iddefs = procDef[0][3][i]
    for j in 0 ..< iddefs.len - 2:
      let ident = iddefs[j]
      result.ids.add ident
      let raw_ptr = newIdentNode($ident & "_raw_ptr")
      result.ptrs.inParams.add raw_ptr

      if j == 0:
        result.length = quote do: `ident`.len
      else:
        let len0 = result.length
        result.initStmt.add quote do:
          assert `len0` == `ident`.len
      result.initStmt.add quote do:
        let `raw_ptr` = cast[ptr UncheckedArray[`type0`]](`ident`[0].unsafeAddr)
      result.simds.inParams.add newIdentNode($ident & "_simd")

  # Now add the result idents
  # We work at simd vector level
  let len0 = result.length

  if resultType.kind == nnkEmpty:
    discard
  elif resultType.kind == nnkTupleTy:
    for i in 0 ..< resultType.len:
      let iddefs = resultType[i]
      for j in 0 ..< iddefs.len - 2:
        let ident = iddefs[j]
        result.ids.add ident
        let raw_ptr = newIdentNode($ident & "_raw_ptr")
        result.ptrs.outParams.add raw_ptr

        let res = nnkDotExpr.newTree(
                    newIdentNode"result",
                    iddefs[j]
                  )
        result.initStmt.add quote do:
          `res` = newSeq[`type0`](`len0`)
          let `raw_ptr` = cast[ptr UncheckedArray[`type0`]](`res`[0].unsafeAddr)

        result.simds.outParams.add newIdentNode($ident & "_simd")

macro compile(arch: static SimdArch, io: static varargs[LuxNode], procDef: untyped): untyped =
  # Note: io must be an array - https://github.com/nim-lang/Nim/issues/10691

  # compile([a, b, c, bar, baz, buzz]):
  #   proc foobar[T](a, b, c: T): tuple[bar, baz, buzz: T]
  #
  # StmtList
  #   ProcDef
  #     Ident "foobar"
  #     Empty
  #     GenericParams
  #       IdentDefs
  #         Ident "T"
  #         Empty
  #         Empty
  #     FormalParams
  #       TupleTy
  #         IdentDefs
  #           Ident "bar"
  #           Ident "baz"
  #           Ident "buzz"
  #           Ident "T"
  #           Empty
  #       IdentDefs
  #         Ident "a"
  #         Ident "b"
  #         Ident "c"
  #         Ident "T"
  #         Empty
  #     Empty
  #     Empty
  #     Empty

  # echo procDef.treerepr

  ## Sanity checks
  procDef.expectkind(nnkStmtList)
  assert procDef.len == 1, "Only 1 statement is allowed, the function definition"
  procDef[0].expectkind({nnkProcDef, nnkFuncDef})
  # TODO: check that the function inputs are in a symbol table?
  procDef[0][6].expectKind(nnkEmpty)

  let resultTy = procDef[0][3][0]
  let (ids, ptrs, simds, length, initParams) = initParams(procDef, resultTy)

  # echo initParams.toStrLit()

  let seqT = nnkBracketExpr.newTree(
    newIdentNode"seq", newIdentNode"float32"
  )

  # We create the inner SIMD proc, specialized to a SIMD architecture
  # In the inner proc we shadow the original idents ids.
  let simdBody = bodyGen(
    genSimd = true,
    arch = arch,
    io = io,
    ids = ids,
    resultType = resultTy
  )

  var simdProc =  procDef[0].replaceType(seqT, SimdTable[arch][simdType])

  simdProc[6] = simdBody   # Assign to proc body
  echo simdProc.toStrLit

  # We create the inner generic proc
  let genericBody = bodyGen(
    genSimd = false,
    arch = ArchGeneric,
    io = io,
    ids = ids,
    resultType = resultTy
  )

  var genericProc = procDef[0].replaceType(seqT, newIdentNode"float32")
  genericProc[6] = genericBody   # Assign to proc body
  echo genericProc.toStrLit

  # We vectorize the inner proc to apply to an contiguous array
  var vecBody: NimNode
  if arch == x86_SSE:
    vecBody = vectorize(
        procDef[0][0],
        ptrs, simds,
        length,
        arch, 4, 4    # We require 4 alignment as a hack to keep seq[T] and use unaligned load/store in code
      )
  else:
    vecBody = vectorize(
        procDef[0][0],
        ptrs, simds,
        length,
        arch, 4, 8    # We require 4 alignment as a hack to keep seq[T] and use unaligned load/store in code
      )

  result = procDef.copyNimTree()
  let resBody = newStmtList()
  resBody.add initParams
  resBody.add genericProc
  resBody.add simdProc
  resBody.add vecBody
  result[0][6] = resBody

  # echo result.toStrLit

macro generate*(ast_routine: typed, signature: untyped): untyped =
  let formalParams = signature[0][3]
  let ast = ast_routine.resolveASToverload(formalParams)

  # Get the routine signature
  let sig = ast.getImpl[3]
  sig.expectKind(nnkFormalParams)

  # Get all inputs
  var inputs: seq[NimNode]
  for idx_identdef in 1 ..< sig.len:
    let identdef = sig[idx_identdef]
    doAssert identdef[^2].eqIdent"LuxNode"
    identdef[^1].expectKind(nnkEmpty)
    for idx_ident in 0 .. identdef.len-3:
      inputs.add genSym(nskLet, $identdef[idx_ident] & "_")

  # Allocate inputs
  result = newStmtList()
  proc ct(ident: NimNode): NimNode =
    nnkPragmaExpr.newTree(
      ident,
      nnkPragma.newTree(
        ident"compileTime"
      )
    )

  for i, in_ident in inputs:
    result.add newLetStmt(
      ct(in_ident),
      newCall("input", newLit i)
    )

  # Call the AST routine
  let call = newCall(ast, inputs)
  var callAssign: NimNode
  case sig[0].kind
  of nnkEmpty: # Case 1: no result
    result.add call
  # Compile-time tuple destructuring is bugged - https://github.com/nim-lang/Nim/issues/11634
  # of nnkTupleTy: # Case 2: tuple result
  #   callAssign = nnkVarTuple.newTree()
  #   for identdef in sig[0]:
  #     doAssert identdef[^2].eqIdent"LuxNode"
  #     identdef[^1].expectKind(nnkEmpty)
  #     for idx_ident in 0 .. identdef.len-3:
  #       callAssign.add ct(identdef[idx_ident])
  #   callAssign.add newEmptyNode()
  #   callAssign.add call
  #   result.add nnkLetSection.newTree(
  #     callAssign
  #   )
  else: # Case 3: single return value
    callAssign = ct(genSym(nskLet, "callResult_"))
    result.add newLetStmt(
      callAssign, call
    )

  # Collect all the input/output idents
  var io = inputs
  case sig[0].kind
  of nnkEmpty:
    discard
  of nnkTupleTy:
    var idx = 0
    for identdef in sig[0]:
      for idx_ident in 0 .. identdef.len-3:
        io.add nnkBracketExpr.newTree(
          callAssign[0],
          newLit idx
        )
        inc idx
  else:
    io.add callAssign

  result.add quote do:
    compile(x86_SSE, `io`, `signature`)

  echo result.toStrlit

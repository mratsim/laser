# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ../core/lux_types,
  ../utils/macro_utils,
  ../platforms,
  # Compiler passes
  ./passes/pass_build_loops,
  ./lux_codegen,
  # Debug
  ../core/lux_print

# ###########################################
#
#            Compiler backend
#
# ###########################################

# Progressively lowers the computation graph AST to
# a low-level AST.
# This is then translated to Nim AST

proc initParams(
       procDef,
       resultType: NimNode
       ): tuple[
            ids, ids_baseType: seq[NimNode],
            ptrs, simds: tuple[inParams, outParams: seq[NimNode]],
            length: NimNode,
            initStmt: NimNode
          ] =
  # Get the idents from proc definition. We order the same as proc def
  # Start with non-result
  # We work at simd vector level
  result.initStmt = newStmtList()

  var shape0: NimNode
  var len0: NimNode

  for i in 1 ..< procDef[0][3].len: # Proc formal params
    let iddefs = procDef[0][3][i]
    for j in 0 ..< iddefs.len - 2:
      # Ident
      let ident = iddefs[j]
      # result.ids.add ident # unused

      # TODO - support var Tensor
      # Ident base type (without seq)
      if not iddefs[^2].isType"Tensor":
        result.ids_baseType.add iddefs[^2]
      else:
        result.ids_baseType.add iddefs[^2][1]
        # If Tensor take pointers

        # Raw ptr
        let raw_ptr = ident($ident & "_raw_ptr")
        result.ptrs.inParams.add raw_ptr

        # Init statement and iteration length
        if len0.isNil:
          len0 = ident"iter_len"
          shape0 = ident"shape0"
          result.initStmt.add quote do:
            let `shape0` = `ident`.shape
            let `len0` = `ident`.size()
          result.length = len0
        else:
          let len0 = result.length
          result.initStmt.add quote do:
            assert `len0` == `ident`.size()
        result.initStmt.add quote do:
          let `raw_ptr` = `ident`.unsafe_raw_data()

        # SIMD ident
        result.simds.inParams.add ident($ident & "_simd")

  # Now add the result idents
  # We work at simd vector level
  if resultType.kind == nnkEmpty:
    discard
  elif resultType.kind == nnkTupleTy:
    for i in 0 ..< resultType.len:
      let iddefs = resultType[i]
      for j in 0 ..< iddefs.len - 2:
        # Ident
        let ident = iddefs[j]
        # result.ids.add ident # unused

        # Ident base type (without seq)
        if not iddefs[^2].isType"Tensor":
          result.ids_baseType.add iddefs[^2]
        else:
          let baseType = iddefs[^2][1]
          result.ids_baseType.add baseType

          # Unused for result types
          # # Raw ptr
          # let raw_ptr = ident($ident & "_raw_ptr")
          # result.ptrs.outParams.add raw_ptr

          # # Init statement
          # let res = nnkDotExpr.newTree(
          #             ident"result",
          #             iddefs[j]
          #           )
          # result.initStmt.add quote do:
          #   `res` = newTensor[`baseType`](`shape0`)
          #   let `raw_ptr` = `res`.unsafe_raw_data()

          # # SIMD ident
          # result.simds.outParams.add ident($ident & "_simd")
  else:
    if resultType.isType"Tensor":
      let baseType = resultType[1]
      result.ids_baseType.add baseType
    else:
      result.ids_baseType.add resultType

macro compile*(fns: static varargs[Fn], procDef: untyped): untyped =
  ## Lux Compiler backend
  ## Accept an array of AST representing the computation
  ## and generate specialized code from it

  # Note: fns must be an array - https://github.com/nim-lang/Nim/issues/10691

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
  let (ids, ids_baseType, ptrs, simds, length, initParams) = initParams(procDef, resultTy)

  # echo fns.treerepr

  # Sanity check on AST produced
  # echo "\n############################"
  # echo "After loop generation\n"
  let kernel_ast = fns.passBuildLoops()
  # echo kernel_ast.treerepr()
  # echo "\n############################\n"

  let kernel = genKernel(
    arch = ArchGeneric,
    kernel_ast,
    fns,
    ids,
    ids_baseType,
    resultTy
  )

  # echo kernel.toStrLit()

  result = procDef.copyNimTree()
  let resBody = newStmtList()
  # resBody.add initParams

  # Quick hack
  let bar = ident"bar"
  resBody.add quote do:
    var `bar` = newTensor[a.T](a.shape[0], a.shape[1])

  resBody.add kernel

  result[0][6] = resBody

  echo "\n-------------"
  echo "\nCompiled proc"
  echo result.toStrLit

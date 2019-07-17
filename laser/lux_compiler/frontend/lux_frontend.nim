# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ../platforms,
  ../core/lux_types,
  ./lux_sigmatch,
  ./lux_symbolic_exec

from ../backend/lux_backend import compile

# ###########################################
#
#           Lux Compiler Frontend
#
# ###########################################

macro generate*(ast_routine: typed, signature: untyped): untyped =
  ## Lux Compiler frontend
  ##
  ## It will symbolically execute an ast_routine of LuxNodes
  ## that represent the computation to be done.
  ##
  ## The symbolic execution will create a computation graph,
  ## that will then be passed to the compiler backend.
  ##
  ## The compiler backend will be called to generate a new proc
  ## that matches the desired signature.
  ##
  ## The procs generated are specialized (they cannot be generic)
  ## but the same base ast_routine can be reused to generate all
  ## the desired specialized procs.
  ##
  ## The name of the generated proc does not need to be the same
  ## as the symbolic proc.
  ##
  ## The signature can include pragmas like {.exportc.}

  result = newStmtList()

  # TODO: canonicalize signature
  let formalParams = signature[0][3]
  let ast = ast_routine.resolveASToverload(formalParams)

  # Get the routine signature
  let sig = ast.getImpl[3]
  sig.expectKind(nnkFormalParams)

  # Get all inputs
  var inputSyms: seq[NimNode]
  for idx_identdef in 1 ..< sig.len:
    let identdef = sig[idx_identdef]
    doAssert identdef[^2].eqIdent"Fn" or
               identdef[^2].eqIdent"Invariant"
    identdef[^1].expectKind(nnkEmpty)
    for idx_ident in 0 .. identdef.len-3:
      inputSyms.add genSym(nskLet, $identdef[idx_ident] & "_")

  # Symbolic execution statement
  var outputSyms: NimNode
  var symExecStmt = newStmtList()
  symbolicExecStmt(
      ast,
      inputSyms,
      hasOut = sig[0].kind != nnkEmpty,
      outputSyms,
      symExecStmt
    )

  # Collect all the input/output idents
  var io = inputSyms
  case sig[0].kind
  of nnkEmpty:
    discard
  of nnkTupleTy:
    var idx = 0
    for identdef in sig[0]:
      for idx_ident in 0 .. identdef.len-3:
        io.add nnkBracketExpr.newTree(
          outputSyms[0],
          newLit idx
        )
        inc idx
  else:
    io.add outputSyms

  # Call the compilation macro
  result.add symExecStmt
  result.add quote do:
    compile(`io`, `signature`)

  echo result.toStrlit

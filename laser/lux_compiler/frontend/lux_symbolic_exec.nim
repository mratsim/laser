# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros,
  # Internal
  ../core/lux_types,
  ../core/lux_core_helpers,
  ../utils/macro_utils

# ###########################################
#
#              Symbolic execution
#
# ###########################################

proc inputTensor(paramId: int): LuxNode =
  LuxNode(
    id: genId(),
    kind: InTensor, symId: paramId
  )

proc symbolicExecStmt*(ast: NimNode, inputSyms: seq[NimNode], hasOut: bool, outputSyms, stmts: var NimNode) =
  # Allocate inputs
  for i, in_ident in inputSyms:
    stmts.add newLetStmt(
      ct(in_ident),
      newCall(bindSym"inputTensor", newLit i)
    )

  # Call the AST routine
  let call = newCall(ast, inputSyms)
  if not hasOut: # Case 1: no result
    stmts.add call
  else:
    outputSyms = ct(genSym(nskLet, "callResult_"))
    stmts.add newLetStmt(
      outputSyms, call
    )

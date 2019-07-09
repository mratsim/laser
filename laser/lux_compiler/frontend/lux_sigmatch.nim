# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # Standard library
  macros

# ###########################################
#
#            Signature matching
#
# ###########################################

proc matchAST(overload, signature: NimNode): bool =
  proc inspect(overload, signature: NimNode, match: var bool) =
    # echo "overload: ", overload.kind, " - match status: ", match
    if overload.kind in {nnkIdent, nnkSym} and overload.eqident("LuxNode"):
      # LuxNode match with any type
      # It should especially match with seq[T] which is of kind nnkBracketExpr
      return

    # Return early when not matching
    if overload.kind != signature.kind:
      match = false
    if overload.len != signature.len:
      match = false
    if match == false:
      return

    case overload.kind:
    of {nnkIdent, nnkSym}:
      match = eqIdent(overload, signature)
    of nnkEmpty:
      discard
    else:
      for i in 0 ..< overload.len:
        inspect(overload[i], signature[i], match)

  result = true
  inspect(overload, signature, result)

proc resolveASToverload*(overloads, formalParams: NimNode): NimNode =
  if overloads.kind == nnkSym:
    result = overloads
    result.getImpl[3].expectKind nnkFormalParams
    return
  else:
    overloads.expectKind(nnkClosedSymChoice)
    for o in overloads:
      let implSig = o.getImpl[3]
      implSig.expectKind nnkFormalParams
      let match = implSig.matchAST(formalParams)
      if match:
        return o
    raise newException(ValueError, "no matching overload found")

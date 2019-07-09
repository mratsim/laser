# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # standard library
  strutils,
  # internal
  ../core/lux_types

# ###########################################
#
#              Pretty Printer
#
# ###########################################

proc `$`*(ast: LuxNode): string =
  proc inspect(ast: LuxNode, indent: int): string =

    if ast.kind == Domain:
      # TODO support domain expression like i+1
      return ast.symDomain

    result.add '\n' & repeat(' ', indent) & $ast.kind & " (id: " & $ast.id & ')'
    let indent = indent + 2
    case ast.kind
    of InTensor:
      result.add '\n' & repeat(' ', indent) & "paramId \"" & $ast.symId & "\""
    of MutTensor, LValTensor:
      result.add '\n' & repeat(' ', indent) & "symLVal \"" & ast.symLVal & "\""
      result.add '\n' & repeat(' ', indent) & "version \"" & $ast.version & "\""
      if ast.prev_version.isNil:
        result.add '\n' & repeat(' ', indent) & "prev_version: nil"
      else:
        result.add repeat(' ', indent) & "⮢⮢⮢" &
          inspect(ast.prev_version, indent)
    of IntImm:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatImm:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of BinOp:
      result.add '\n' &  repeat(' ', indent) & $ast.binOpKind
      result.add repeat(' ', indent) & inspect(ast.lhs, indent)
      result.add repeat(' ', indent) & inspect(ast.rhs, indent)
    of Access, MutAccess:
      result.add '\n' &  repeat(' ', indent) & "indices " & $ast.indices
      result.add repeat(' ', indent) & inspect(ast.tensorView, indent)
    of Assign:
      result.add repeat(' ', indent) & inspect(ast.lval, indent)
      result.add repeat(' ', indent) & inspect(ast.rval, indent)
    else:
      raise newException(
        ValueError, "Pretty Printer for \"" &
                    $ast.kind & "\" is not implemented")

  result = inspect(ast, 0)

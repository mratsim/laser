# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # standard library
  strutils,
  # internal
  ./ast_definition

proc `$`*(ast: LuxNode): string =
  proc inspect(ast: LuxNode, indent: int): string =
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
        result.add repeat(' ', indent) & inspect(ast.prev_version, indent)
    of IntImm:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatImm:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of Assign, Add, Mul:
      result.add repeat(' ', indent) & inspect(ast.lhs, indent)
      result.add repeat(' ', indent) & inspect(ast.rhs, indent)

  result = inspect(ast, 0)

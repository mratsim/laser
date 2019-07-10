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
#              Pretty Printers
#
# ###########################################

proc toStrLit*(ast: LuxNode): string =
  if ast.isNil:
    return "nil"
  case ast.kind:
  of IntImm: return $ast.intVal
  of FloatImm: return $ast.floatVal
  of BinOp:
    case ast.binOpKind
    of Add: return ast.lhs.toStrLit & "+" & ast.rhs.toStrLit
    of Mul: return ast.lhs.toStrLit & "*" & ast.rhs.toStrLit
  of Domain:
    result = "Domain(iterator: \""
    result.add ast.symDomain
    result.add "\", from: "
    result.add ast.start.toStrLit
    result.add ", to: "
    result.add ast.stop.toStrLit
    result.add ", step: "
    result.add ast.step.toStrLit
  of InTensor:
    return "In" & $ast.symId
  of MutTensor, LValTensor:
    return ast.symLVal
  of Shape:
    result = ast.tensor.toStrLit
    result.add ".shape["
    result.add $ast.axis
    result.add ']'
  else:
    raise newException(
      ValueError, "Pretty Printer for \"" &
                  $ast.kind & "\" is not implemented")

proc toStrLit*(asts: openarray[LuxNode]): string =
  result = "["
  for i, ast in asts:
    if i != 0:
      result.add ", "
    result.add toStrLit(ast)
  result.add ']'

proc treeRepr*(ast: LuxNode): string =
  proc inspect(ast: LuxNode, indent: int): string =
    if ast.isNil:
      return '\n' & repeat(' ', indent) & "nil"

    result.add '\n' & repeat(' ', indent) & $ast.kind & " (id: " & $ast.id & ')'
    let indent = indent + 2
    case ast.kind
    of InTensor:
      result.add '\n' & repeat(' ', indent) & "paramId \"" & $ast.symId & '\"'
    of MutTensor, LValTensor:
      result.add '\n' & repeat(' ', indent) & "symLVal \"" & ast.symLVal & '\"'
      result.add '\n' & repeat(' ', indent) & "version \"" & $ast.version & '\"'
      if ast.prev_version.isNil:
        result.add '\n' & repeat(' ', indent) & "prev_version: nil"
      else:
        result.add repeat(' ', indent) & "⮢⮢⮢ (prev_version)" &
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
      result.add '\n' &  repeat(' ', indent) & "indices " & ast.indices.toStrLit
      result.add repeat(' ', indent) & inspect(ast.tensorView, indent)
    of Assign:
      result.add '\n' & repeat(' ', indent) & "domains " & ast.domains.toStrLit
      result.add repeat(' ', indent) & inspect(ast.lval, indent)
      result.add repeat(' ', indent) & inspect(ast.rval, indent)
    of Domain:
      result.add '\n' & repeat(' ', indent) & "symDomain \"" & $ast.symDomain & '\"'
      result.add '\n' & repeat(' ', indent) & "start \"" & ast.start.toStrLit & '\"'
      result.add '\n' & repeat(' ', indent) & "stop \"" & ast.stop.toStrLit & '\"'
      result.add '\n' & repeat(' ', indent) & "step \"" & ast.step.toStrLit & '\"'
    of AffineFor:
      result.add '\n' & repeat(' ', indent) & ast.domain.toStrLit
      result.add repeat(' ', indent) &
          inspect(ast.affineForBody, indent)
    else:
      raise newException(
        ValueError, "Pretty Printer for \"" &
                    $ast.kind & "\" is not implemented")

  result = inspect(ast, 0)

proc treeRepr*(asts: openarray[LuxNode]): string =
  result = "["
  for i, ast in asts:
    result.add "\n\n- Node " & $i & "--------\n"
    result.add treerepr(ast)
  result.add "\n- --------------\n]"

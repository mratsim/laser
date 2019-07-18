# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # standard library
  strutils,
  # internal
  ../core/[lux_types, lux_core_helpers]

# ###########################################
#
#              Pretty Printers
#
# ###########################################

proc toStrLit*(ast: LuxNode): string =
  if ast.isNil:
    return "nil"
  case ast.kind:
  of IntLit: return $ast.intVal
  of FloatLit: return $ast.floatVal
  of IntParam, FloatParam:
    return ast.symParam
  of Domain:
    if ast.iter.symbol != "":
      return ast.iter.symbol
    else:
      return "iter" & $ast.id
  of Func:
    return ast.fn.symbol
  of DimSize:
    assert ast[0].kind == Func
    assert ast[1].kind == IntLit
    return ast[0].fn.symbol &
             ".shape[" &
             $ast[1].intVal &
             ']'
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

proc `?`(node: LuxNode): string =
  if node.isNil:
    result = "nil"
  else:
    result = node.toStrLit

proc shortId(id: Id): string =
  if false:
    result = $id
    if result.len > 3:
      result = "\t\t(id: " & result[0..<3] & ')'
    else:
      result = "\t\t(id: " & result & ')'

proc shortDomain*(ast: LuxNode): string =
  assert ast.kind == Domain
  result = "symbol: \""
  if ast.iter.isNil:
    result.add "none\""
  else:
    result.add ast.iter.symbol
    result.add "\", domId: \""
    result.add ast.iter.domId
    result.add "\", start: "
    result.add ?ast.iter.start
    result.add ", stop: "
    result.add ?ast.iter.stop
    result.add ", step: "
    result.add ?ast.iter.step
  result.add ')'

proc treeRepr*(ast: LuxNode): string =
  proc inspect(ast: LuxNode, indent: int): string =
    if ast.isNil:
      return '\n' & repeat(' ', indent) & "nil"

    result.add '\n' & repeat(' ', indent) & $ast.kind & shortId(ast.id)
    let indent = indent + 2
    case ast.kind
    of Func:
      result.add '\n' & repeat(' ', indent) & "function \"" & $ast.fn.symbol & '\"'
    of IntLit:
      result.add '\n' & repeat(' ', indent) & $ast.intVal
    of FloatLit:
      result.add '\n' & repeat(' ', indent) & $ast.floatVal
    of IntParam, FloatParam:
      result.add '\n' & repeat(' ', indent) & "symbol \"" & $ast.symParam & '\"'
    of Domain:
      result.add '\n' & repeat(' ', indent) & shortDomain(ast)
    of BinOpKind:
      result.add '\n' & repeat(' ', indent) & $ast.bopKind
    else:
      for node in ast:
        result.add repeat(' ', indent) & inspect(node, indent)

  result = inspect(ast, 0)

proc treeRepr*(asts: openarray[LuxNode]): string =
  result = "["
  for i, ast in asts:
    result.add "\n\n- Node " & $i & "--------\n"
    result.add treerepr(ast)
  result.add "\n- --------------\n]"

proc treeRepr*(fns: openarray[Fn]): string =
  result = "["
  for i, fn in fns:
    result.add "\n\n===== Function " & $i & ", " & fn.symbol & "\t\t======="
    for j, stage in fn.stages:
      result.add "\n----- Function " & $i & ", " & fn.symbol & "\t\tStage " & $j & '\n'
      result.add stage.definition.treerepr()
      result.add "\n\n----- Function " & $i & ", " & fn.symbol & "\t\tStage " & $j & '\n'

  result.add "\n= ================\n]"
